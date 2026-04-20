"""
manual_model_test.py
Flask server for NaVILA interactive demo.

Usage:
    python examples/manual_model_test.py [--load_4bit] [--do_sample] [--temperature T] [--model_path PATH] [--port PORT]
"""

from llava.mm_utils import KeywordsStoppingCriteria
from llava.conversation import SeparatorStyle
import argparse
import base64
import io
import json
import re
from collections import deque

import torch
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="NaVILA interactive Flask demo")
parser.add_argument("--load_4bit", action="store_true", default=True, help="Enable 4-bit quantization (default: True)")
parser.add_argument("--no_load_4bit", dest="load_4bit", action="store_false", help="Disable 4-bit quantization (falls back to 8-bit)")
parser.add_argument("--model_path", type=str, default="a8cheng/navila-llama3-8b-8f", help="HuggingFace model path")
parser.add_argument("--port", type=int, default=5000, help="Port to run the server on (default: 5000)")
parser.add_argument("--demo_dir", type=str, default="demo", help="Path to demo static files")
parser.add_argument("--max_frames", type=int, default=8, help="Maximum number of historical frames to use (default: 8)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
print(f"[INFO] Loading model: {args.model_path}  (4bit={args.load_4bit})")
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=args.model_path,
    model_base=None,
    model_name=get_model_name_from_path(args.model_path),
    load_8bit=not args.load_4bit,   # fall back to 8-bit when 4-bit is disabled
    load_4bit=args.load_4bit,       # 4-bit quantization
    device_map="auto",
    device="cuda",
    torch_dtype=torch.float16,
)
print("[INFO] Model loaded.")

# ---------------------------------------------------------------------------
# Frame history  (max 8 frames, FIFO)
# ---------------------------------------------------------------------------
frame_history: deque[Image.Image] = deque(maxlen=args.max_frames)

# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------
NAV_PROMPT_TEMPLATE = """\
Imagine you are a robot programmed for navigation tasks.
You have been given a video of historical observations: {hist_image_tokens}
and current observation: {curr_image_token}
Your assigned task is: {instruction}
Analyze this series of images to decide your next move,
which could involve turning left or right by a specific degree,
moving forward a certain distance, or stop if the task is completed.\
"""


def run_inference(instruction: str, images: list[Image.Image]) -> str:
    image_tensor = process_images(images, image_processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = [img.to(dtype=torch.float16, device="cuda") for img in image_tensor]
    else:
        image_tensor = [image_tensor.to(dtype=torch.float16, device="cuda")]

    # Create historical image tokens
    hist_image_tokens = "\n".join([DEFAULT_IMAGE_TOKEN for _ in range(len(images) - 1)]) if len(images) > 1 else ""

    prompt_text = NAV_PROMPT_TEMPLATE.format(
        hist_image_tokens=hist_image_tokens,    
        curr_image_token=DEFAULT_IMAGE_TOKEN,
        instruction=instruction,
    )

    conv = conv_templates["llama_3"].copy()
    conv.append_message(conv.roles[0], prompt_text)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to("cuda")

    attention_mask = torch.ones_like(input_ids)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            attention_mask=attention_mask,
            pad_token_id=pad_token_id,
            do_sample=False,
            temperature=0,
            top_p=None,
            num_beams=1,
            max_new_tokens=512,
            use_cache=True,
            stopping_criteria=[stopping_criteria]
        )

    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


def crop_and_resize(img: Image.Image, size=384) -> Image.Image:
    w, h = img.size
    min_dim = min(w, h)
    left = (w - min_dim) / 2
    top = (h - min_dim) / 2
    right = (w + min_dim) / 2
    bottom = (h + min_dim) / 2
    img = img.crop((left, top, right, bottom))
    return img.resize((size, size), Image.Resampling.BICUBIC)


def parse_action_output(raw_text: str) -> dict | None:
    """Extract action from the expected text output format."""
    match = re.search(r"The next action is\s+(.*)", raw_text, re.IGNORECASE)
    if match:
        action_str = match.group(1).strip()
        # Remove any surrounding quotes or punctuation if needed
        action_str = action_str.strip('."\'')
        return {"action": action_str}
    
    # Fallback
    fallback_match = re.search(r"(forward|backward|left|right|stop).*?(?:[0-9]+)?", raw_text, re.IGNORECASE)
    if fallback_match:
        return {"action": raw_text.strip()}

    return None


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder=args.demo_dir)


@app.route("/")
def index():
    return send_from_directory(args.demo_dir, "index.html")


@app.route("/api/capture", methods=["POST"])
def capture():
    """Receive a base64-encoded JPEG frame, add it to history."""
    data = request.get_json(force=True)
    image_b64: str = data.get("image", "")
    if not image_b64:
        return jsonify({"error": "No image data"}), 400

    # Strip data-URL header if present
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]

    img_bytes = base64.b64decode(image_b64)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    
    img = crop_and_resize(img, 384)
    
    frame_history.append(img)

    return jsonify({"frame_count": len(frame_history), "max_frames": args.max_frames})


@app.route("/api/infer", methods=["POST"])
def infer():
    """Run NaVILA inference on current frame history."""
    data = request.get_json(force=True)
    instruction: str = data.get("instruction", "").strip()

    if not instruction:
        return jsonify({"error": "Instruction is empty"}), 400
    if len(frame_history) == 0:
        return jsonify({"error": "No frames in history. Capture at least one frame first."}), 400

    images = list(frame_history)

    # Pad to args.max_frames by repeating the first frame if necessary
    while len(images) < args.max_frames:
        images.insert(0, images[0])

    raw_result = run_inference(instruction, images)
    parsed_result = parse_action_output(raw_result)
    
    return jsonify({
        "raw": raw_result,
        "parsed": parsed_result
    })


@app.route("/api/reset", methods=["POST"])
def reset():
    """Clear the frame history."""
    frame_history.clear()
    return jsonify({"status": "ok", "frame_count": 0})


@app.route("/api/status", methods=["GET"])
def status():
    return jsonify({"frame_count": len(frame_history), "max_frames": args.max_frames})


if __name__ == "__main__":
    print(f"[INFO] Starting server on http://0.0.0.0:{args.port}")
    print(f"[INFO] Open http://localhost:{args.port} in your browser")
    app.run(host="0.0.0.0", port=args.port, debug=False)

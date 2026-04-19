"""
manual_model_test.py
Flask server for NaVILA interactive demo.

Usage:
    python examples/manual_model_test.py [--load_4bit] [--do_sample] [--temperature T] [--model_path PATH] [--port PORT]
"""

import argparse
import base64
import io
import json
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
parser.add_argument("--do_sample", action="store_true", default=False, help="Enable sampling-based decoding (default: False = greedy)")
parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature, only used when --do_sample is set (default: 0.7)")
parser.add_argument("--model_path", type=str, default="a8cheng/navila-llama3-8b-8f", help="HuggingFace model path")
parser.add_argument("--port", type=int, default=5000, help="Port to run the server on (default: 5000)")
parser.add_argument("--demo_dir", type=str, default="demo", help="Directory containing index.html (default: examples/demo)")
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
MAX_FRAMES = 8
frame_history: deque[Image.Image] = deque(maxlen=MAX_FRAMES)

# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------
NAV_PROMPT_TEMPLATE = """\
You are a navigation robot.

## Rules:
- Output the description of the scene.
- Explain purpose of your next action.
- Output the next action.
- description: free text
- purpose: free text
- action: choose one action from below.
    - forward <d> cm: move forward by d cm
    - backward <d> cm: move backward by d cm
    - left <d> deg: turn left by d degrees
    - right <d> deg: turn right by d degrees
    - stop: stop
- <d> is natural number.
- don't forget to use the tags below.

## Format
```
{{
"description": "description text",
"purpose": "purpose text",
"action": "action text"
}}
```

Task: {instruction}
"""


def run_inference(instruction: str, images: list[Image.Image]) -> str:
    image_tensor = process_images(images, image_processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = [img.to(dtype=torch.float16, device="cuda") for img in image_tensor]
    else:
        image_tensor = [image_tensor.to(dtype=torch.float16, device="cuda")]

    nav_prompt = NAV_PROMPT_TEMPLATE.format(instruction=instruction)
    prompt_text = f"{DEFAULT_IMAGE_TOKEN}\n{nav_prompt}"

    conv = conv_templates["llama_3"].copy()
    conv.append_message(conv.roles[0], prompt_text)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to("cuda")

    attention_mask = torch.ones_like(input_ids)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            attention_mask=attention_mask,
            pad_token_id=pad_token_id,
            do_sample=args.do_sample,
            temperature=args.temperature if args.do_sample else None,
            max_new_tokens=1000,
            use_cache=True,
        )

    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


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
    frame_history.append(img)

    return jsonify({"frame_count": len(frame_history), "max_frames": MAX_FRAMES})


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

    # Pad to MAX_FRAMES by repeating the first frame if necessary
    while len(images) < MAX_FRAMES:
        images.insert(0, images[0])

    result = run_inference(instruction, images)
    return jsonify({"result": result})


@app.route("/api/reset", methods=["POST"])
def reset():
    """Clear the frame history."""
    frame_history.clear()
    return jsonify({"status": "ok", "frame_count": 0})


@app.route("/api/status", methods=["GET"])
def status():
    return jsonify({"frame_count": len(frame_history), "max_frames": MAX_FRAMES})


if __name__ == "__main__":
    print(f"[INFO] Starting server on http://0.0.0.0:{args.port}")
    print(f"[INFO] Open http://localhost:{args.port} in your browser")
    app.run(host="0.0.0.0", port=args.port, debug=False)

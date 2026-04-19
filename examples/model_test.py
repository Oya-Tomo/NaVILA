import argparse
import torch
from PIL import Image

from llava.model.builder import load_pretrained_model
from llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token
)
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

parser = argparse.ArgumentParser(description="NaVILA model test")
parser.add_argument("--load_4bit", action="store_true", default=True, help="Enable 4-bit quantization (default: True)")
parser.add_argument("--do_sample", action="store_true", default=False, help="Enable sampling-based decoding (default: False = greedy)")
parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature, only used when --do_sample is set (default: 0.7)")
parser.add_argument("--model_path", type=str, default="a8cheng/navila-llama3-8b-8f", help="Model path (default: a8cheng/navila-llama3-8b-8f)")
args = parser.parse_args()

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=args.model_path,
    model_base=None,
    model_name=get_model_name_from_path(args.model_path),
    load_8bit=not args.load_4bit, # fall back to 8-bit when 4-bit is disabled
    load_4bit=args.load_4bit,     # 4-bit quantization
    device_map="auto",
    device="cuda",
    torch_dtype=torch.float16
)

image_frame_paths = [
    "examples/images/scene1/frame1.jpg", # older frame
    "examples/images/scene1/frame2.jpg",
    "examples/images/scene1/frame3.jpg",
    "examples/images/scene1/frame4.jpg",
    "examples/images/scene1/frame5.jpg",
    "examples/images/scene1/frame6.jpg",
    "examples/images/scene1/frame7.jpg",
    "examples/images/scene1/frame8.jpg", # newer frame
]

def crop_and_resize(img: Image.Image, size=384) -> Image.Image:
    w, h = img.size
    min_dim = min(w, h)
    left = (w - min_dim) / 2
    top = (h - min_dim) / 2
    right = (w + min_dim) / 2
    bottom = (h + min_dim) / 2
    img = img.crop((left, top, right, bottom))
    return img.resize((size, size), Image.Resampling.BICUBIC)

instruction = "Go to the elevator and use it."

images = [crop_and_resize(Image.open(p).convert("RGB"), 384) for p in image_frame_paths]

image_tensor = process_images(images, image_processor, model.config)
if isinstance(image_tensor, list):
    image_tensor = [img.to(dtype=torch.float16, device='cuda') for img in image_tensor]
else:
    image_tensor = [image_tensor.to(dtype=torch.float16, device='cuda')]

conv = conv_templates["llama_3"].copy()

hist_image_tokens = "\n".join([f"{DEFAULT_IMAGE_TOKEN}" for _ in range(len(image_frame_paths) - 1)])

NAVIGATION_PROMPT = f"""
You are a navigation robot.

## Task
{instruction}

## Environment Information

a video of historical observations:
{hist_image_tokens}
current observation:
{DEFAULT_IMAGE_TOKEN}

## Format rules

- Output a JSON object that follows the template. Other type of output is forbidden.
- Choose one action at a time.
  - forward d cm
  - backward d cm
  - left d deg
  - right d deg
  - stop

## Template
{{
  "action": "<action>"
}}
"""

prompt_text = NAVIGATION_PROMPT
conv.append_message(conv.roles[0], prompt_text)
conv.append_message(conv.roles[1], None)
full_prompt = conv.get_prompt()

input_ids = tokenizer_image_token(
    full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
).unsqueeze(0).to('cuda')

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
        use_cache=True
    )

response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
print(response)
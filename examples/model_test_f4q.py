import torch

from llava.model.builder import load_pretrained_model
from llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token
)
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

model_path = "a8cheng/navila-llama3-8b-8f"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path),
    load_8bit=False,
    load_4bit=True, # 4bit quantization
    device_map="auto",
    device="cuda",
    torch_dtype=torch.float16
)

image_frame_paths = [
    "examples/images/scene1/frame1.jpg",
    "examples/images/scene1/frame2.jpg",
    "examples/images/scene1/frame3.jpg",
    "examples/images/scene1/frame4.jpg",
    "examples/images/scene1/frame5.jpg",
    "examples/images/scene1/frame6.jpg",
    "examples/images/scene1/frame7.jpg",
    "examples/images/scene1/frame8.jpg",
]

# ナビゲーション命令
instruction = "Go to the elevator and use it."

# 画像前処理
image_tensor = process_images(image_frame_paths, image_processor, model.config)
if isinstance(image_tensor, list):
    image_tensor = [img.to(dtype=torch.float16, device='cuda') for img in image_tensor]
else:
    image_tensor = [image_tensor.to(dtype=torch.float16, device='cuda')]

# プロンプト構築
conv = conv_templates["llama_3"].copy()

NAV_PROMPT = f"""You are a navigation robot.
Given the current camera image, output ONLY the next navigation action as ONE line.

Allowed formats (choose exactly one):
- move forward <d> cm
- turn left <theta> degrees
- turn right <theta> degrees
- stop

Rules:
- Output only one line. No explanation.
- Use integers for <d> and <theta>.
- If the goal is reached or you are unsure, output: stop
- Make decisions based on the surrounding circumstances.

Task: "{instruction}"
The next action is:
"""

prompt_text = f"{DEFAULT_IMAGE_TOKEN}\n{NAV_PROMPT}"
conv.append_message(conv.roles[0], prompt_text)
conv.append_message(conv.roles[1], None)
full_prompt = conv.get_prompt()

input_ids = tokenizer_image_token(
    full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
).unsqueeze(0).to('cuda')

with torch.inference_mode():
    output_ids = model.generate(
        input_ids,
        images=image_tensor,
        do_sample=False,
        max_new_tokens=50,
        use_cache=True
    )

response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
print(response)
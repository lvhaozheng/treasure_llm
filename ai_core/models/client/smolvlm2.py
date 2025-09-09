import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers.image_utils import load_image

# Load local image
# image = Image.open("image.png")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize processor and model
# 使用本地模型路径 - 更新为新的相对路径
local_model_path = "../../smolvlm2_small_model"  # 相对于当前文件的路径
processor = AutoProcessor.from_pretrained(local_model_path)
model = AutoModelForImageTextToText.from_pretrained(
    local_model_path,
    dtype=torch.bfloat16,
    _attn_implementation="eager",  # 使用eager避免flash_attn依赖
).to(DEVICE)

# Create input messages
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": f''' 你是一个专业的古董鉴定专家，具有丰富的古董知识和鉴定经验。
        你的主要职责包括：
        1. 分析古董的特征、年代、材质等信息
        2. 评估古董的真伪和收藏价值
        3. 提供古董相关的历史背景和文化信息
        4. 回答用户关于古董的各种问题
        
        请始终保持专业、客观的态度，基于事实进行分析和回答。
        如果不确定某些信息，请明确说明并建议用户咨询更专业的鉴定机构。请问青花瓷是什么？'''}
        ]
    },
]

# Prepare inputs
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, return_tensors="pt")
inputs = inputs.to(DEVICE)

# Generate outputs
generated_ids = model.generate(**inputs, max_new_tokens=500)
generated_texts = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)

print(generated_texts[0])
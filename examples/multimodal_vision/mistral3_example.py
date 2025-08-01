import json
import os

import requests
import torch
from PIL import Image
from transformers import AutoProcessor, Mistral3ForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.utils import dispatch_for_generation

# Load model.
model_id = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
model = Mistral3ForConditionalGeneration.from_pretrained(model_id, torch_dtype="auto")
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# Use a custom calibration chat template, rather than the overly-verbose default
file_path = os.path.join(os.path.dirname(__file__), "mistral3_chat_template.json")
with open(file_path, "r") as file:
    processor.chat_template = json.load(file)["chat_template"]

# Oneshot arguments
DATASET_ID = "flickr30k"
DATASET_SPLIT = "test"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048


# Define a oneshot data collator for multimodal inputs.
def data_collator(batch):
    assert len(batch) == 1
    return {
        key: torch.tensor(value)
        if key != "pixel_values"
        else torch.tensor(value, dtype=model.dtype)
        for key, value in batch[0].items()
    }


# Recipe
recipe = [
    GPTQModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=["re:.*lm_head", "re:vision_tower.*", "re:multi_modal_projector.*"],
    ),
]

# Perform oneshot
oneshot(
    model=model,
    tokenizer=model_id,
    dataset=DATASET_ID,
    splits={"calibration": f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]"},
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    trust_remote_code_model=True,
    data_collator=data_collator,
    sequential_targets=["MistralDecoderLayer"],
)

# Confirm generations of the quantized model look sane.
print("========== SAMPLE GENERATION ==============")
dispatch_for_generation(model)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Please describe the animal in this image\n"},
            {"type": "image"},
        ],
    },
]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
image_url = "http://images.cocodataset.org/train2017/000000231895.jpg"
raw_image = Image.open(requests.get(image_url, stream=True).raw)

inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to("cuda")
inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)  # fix dtype
output = model.generate(**inputs, max_new_tokens=100)
print(processor.decode(output[0], skip_special_tokens=True))
print("==========================================")

# Save to disk compressed.
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-W4A16-G128"
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)

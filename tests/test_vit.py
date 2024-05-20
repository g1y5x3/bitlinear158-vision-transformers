import torch
from datasets import load_dataset
from transformers import AutoImageProcessor, ViTForImageClassification

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
model.to("cuda")
print(model)

inputs = image_processor(image, return_tensors="pt").to("cuda")

with torch.no_grad():
  logits = model(**inputs).logits

predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
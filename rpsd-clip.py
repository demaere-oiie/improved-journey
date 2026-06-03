import torch
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# 1. Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def classify(image_path):
    image = Image.open(image_path)
    classes = ["a photo of a fist", "a photo of a flat hand", "a photo of a victory sign", "a photo of a thing"]

    inputs = processor(text=classes, images=image, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

        # Calculate probabilities (logits)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=-1)

    r, p, s, d = probs[0]
    if d > 0.05: cl = "D"
    elif r >= 0.5: cl = "R"
    elif s >= 0.5: cl = "S"
    elif p >= 0.5 and s <= 0.35: cl = "P"
    else: cl = "S"

    print(f"{cl} {image_path} {r:.2f}:{p:.2f}:{s:.2f}:{d:.2f}")

p = Path("../../raw/rock")
for file in p.iterdir():
    classify(file)
p = Path("../../raw/paper")
for file in p.iterdir():
    classify(file)
p = Path("../../raw/scissors")
for file in p.iterdir():
    classify(file)
p = Path("../../raw/rc")
for file in p.iterdir():
    classify(file)

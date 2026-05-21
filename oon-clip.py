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
    classes = ["a photo of a person", "a photo of an octopus"]

    inputs = processor(text=classes, images=image, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

        # Calculate probabilities (logits)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=-1)

    print(image_path)
    print("Label probabilities:", probs)
    for i, class_name in enumerate(classes):
        print(f"{class_name}: {probs[0][i]*100:.2f}%")
    print()

classify("../../raw/cobian_headshot.jpg")
p = Path("../../raw/rc")
for file in p.iterdir():
    classify(file)
p = Path("../../raw/oct")
for file in p.iterdir():
    classify(file)

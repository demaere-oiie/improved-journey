import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# 1. Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 2. Load and preprocess your image
# Replace with the path to your image
image_path = "../../raw/rc/prof.jpeg" 
image = Image.open(image_path)

# 3. Define your custom categories (prompts)
# You can use simple class names or descriptive prompts for better accuracy
classes = ["a photo of a person", "a photo of an octopus"]

inputs = processor(text=classes, images=image, return_tensors="pt", padding=True).to(device)

# 4. Run the model
with torch.no_grad():
    outputs = model(**inputs)

    # Calculate probabilities (logits)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=-1)

# 5. Print the results
print("Label probabilities:", probs)
for i, class_name in enumerate(classes):
    print(f"{class_name}: {probs[0][i]*100:.2f}%")

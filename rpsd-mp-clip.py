import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# 1. Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# STEP 2: Create an HandLandmarker object.
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

def classify(image_path):
  # STEP 3: Load the input image.
  image = mp.Image.create_from_file(str(image_path))

  # STEP 4: Detect hand landmarks from the input image.
  detection_result = detector.detect(image)

  if not detection_result.hand_landmarks:
    return classify2(image_path, image_path)

  mx,my = 1,1
  xx,xy = 0,0
  for l in detection_result.hand_landmarks[0]:
    if l.x < mx: mx=l.x
    if l.y < my: my=l.y
    if l.x > xx: xx=l.x
    if l.y > xy: xy=l.y

  sm, sx = (.8,1.2)
  mx = max(int(mx*sm*image.width),0)
  my = max(int(my*sm*image.height),0)
  xx = min(int(xx*sx*image.width),image.width)
  xy = min(int(xy*sx*image.height),image.height)
  #print((mx,xx,my,xy))

  sc = ((image.width*image.height) / (xx-mx)*(xy-my))
  if (image.width/(xx-mx))>3 or  (image.height/(xy-my))>3:

      image = cv2.imread(image_path)
      cropped = image[my:xy, mx:xx]
      cv2.imwrite('/tmp/crop.png',cropped)
      return classify2(Path('/tmp/crop.png'),image_path)

  else:
      return classify2(image_path,image_path)

def classify2(image_path,origin):
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
    else: cl = "D"

    print(f"{cl} {origin} {r:.2f}:{p:.2f}:{s:.2f}:{d:.2f}")

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

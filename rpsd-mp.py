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

# https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker
def classify_geom(lm):
    palm = (lm[9].x - lm[0].x, lm[9].y - lm[0].y)
    indx = (lm[8].x - lm[6].x, lm[8].y - lm[6].y)
    ring = (lm[16].x - lm[14].x, lm[16].y - lm[14].y)

    dpi = palm[0]*indx[0] + palm[1]*indx[1]
    dpr = palm[0]*ring[0] + palm[1]*ring[1]

    def sd(x): return x[0]*x[0] + x[1]*x[1]

    print(f"{sd(palm)} {sd(indx)} {sd(ring)}")

    #if abs(dpi)<1e-2 and abs(dpr)<1e-2:
    #    return (True, False)

    return (dpi<0,dpr<0)

    # True True         Rock
    # False False       Paper
    # False True        Scissors

def classify(image_path):
  # STEP 3: Load the input image.
  image = mp.Image.create_from_file(str(image_path))

  # STEP 4: Detect hand landmarks from the input image.
  detection_result = detector.detect(image)

  if not detection_result.hand_landmarks:
    return "D"

  dpi,dpr = classify_geom(detection_result.hand_landmarks[0])

  if (dpi,dpr) == (True,True): cl="R"
  elif (dpi,dpr) == (False,False): cl="P"
  elif (dpi,dpr) == (False,True): cl="S"
  else: cl="D"

  print(f"{cl} {dpi} {dpr} {image_path}")

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

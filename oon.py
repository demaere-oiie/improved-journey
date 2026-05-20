from keras.preprocessing import image
import cv2 as cv
from skimage.io import imread
from pathlib import Path
print("dependencies loaded")

train = []
p = Path("../../raw/rc")
for file in p.iterdir():
    print(file)
    img = imread(file)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.resize(img, (32, 32), interpolation=cv.INTER_AREA)
    img = img / 255.
    train.append(img)
p = Path("../../raw/oct")
for file in p.iterdir():
    print(file)
    img = imread(file)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.resize(img, (32, 32), interpolation=cv.INTER_AREA)
    img = img / 255.
    train.append(img)

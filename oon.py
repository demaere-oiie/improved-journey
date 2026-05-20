import numpy as np
from keras.preprocessing import image
import cv2 as cv
import matplotlib.pyplot as plt
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
    train.append(img)
p = Path("../../raw/oct")
for file in p.iterdir():
    print(file)
    img = imread(file)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.resize(img, (32, 32), interpolation=cv.INTER_AREA)
    train.append(img)

plt.figure(figsize=(10, 10))
for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(np.array(train[i]).astype("uint8"))
        plt.axis("off")

plt.show()

plt.figure(figsize=(10, 10))
for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(np.array(train[-(i+1)]).astype("uint8"))
        plt.axis("off")

plt.show()

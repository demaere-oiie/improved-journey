import numpy as np
from keras.preprocessing import image
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from skimage.io import imread
from pathlib import Path
print("dependencies loaded")

train = []
p = Path("../../raw/rc")
for file in p.iterdir():
    img = imread(file)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.resize(img, (32, 32), interpolation=cv.INTER_AREA)
    train.append(img)
p = Path("../../raw")
for file in p.iterdir():
    if "obian" not in str(file): continue
    print(file)
    img = imread(file)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.resize(img, (32, 32), interpolation=cv.INTER_AREA)
    train.append(img)
p = Path("../../raw/oct")
for file in p.iterdir():
    img = imread(file)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.resize(img, (32, 32), interpolation=cv.INTER_AREA)
    train.append(img)

X_train = np.array(train)
X_train = np.reshape(X_train, (49, 32*32))
print(X_train.shape)
y0 = np.zeros(25)
y1 = np.ones(24)
y_train = np.concatenate((y0,y1), axis=0)

knn = KNeighborsClassifier(n_neighbors=5)

print(knn.fit(X_train, y_train))

y_pred = knn.predict(X_train)

if 1:
  knn = KNeighborsClassifier(n_neighbors=5)
  knn.fit(X_train, y_train)
  print(knn.predict_proba(None)[:25])
  print(knn.predict_proba(None)[25:])

if 1:
        plt.figure(figsize=(10, 10))
        for j in range(1):
            ax = plt.subplot(1, 1, j+1)
            plt.imshow(np.array(train[24]).astype("uint8"))
            plt.axis("off")
        plt.show()

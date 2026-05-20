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
p = Path("../../raw/oct")
for file in p.iterdir():
    img = imread(file)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.resize(img, (32, 32), interpolation=cv.INTER_AREA)
    train.append(img)

if 0:
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

X_train = np.array(train)
X_train = np.reshape(X_train, (48, 32*32))
print(X_train.shape)
y0 = np.zeros(24)
y1 = np.ones(24)
y_train = np.concatenate((y0,y1), axis=0)

knn = KNeighborsClassifier(n_neighbors=5)

print(knn.fit(X_train, y_train))

y_pred = knn.predict(X_train)

for i,t in enumerate(y_train):
    X_red = np.concatenate((X_train[:i],X_train[i+1:]), axis=0)
    y_red = np.concatenate((y_train[:i],y_train[i+1:]), axis=0)
    knn.fit(X_red,y_red)
    p = knn.predict(X_train[i:i+1])[0]
    if t != p:
        print(i)
        plt.figure(figsize=(10, 10))
        for j in range(1):
            ax = plt.subplot(1, 1, j+1)
            plt.imshow(np.array(train[i]).astype("uint8"))
            plt.axis("off")
        plt.show()

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

X_train = np.array(train)
X_train = np.reshape(X_train, (48, 32*32))
print(X_train.shape)
y0 = np.zeros(24)
y1 = np.ones(24)
y_train = np.concatenate((y0,y1), axis=0)


import matplotlib.pyplot as plt

# unused but required import for doing 3d projections with matplotlib < 3.2
#import mpl_toolkits.mplot3d  # noqa: F401

from sklearn.decomposition import PCA

fig = plt.figure(1, figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d", elev=90, azim=0)

X_reduced = PCA(n_components=3).fit_transform(X_train)
scatter = ax.scatter(
    X_reduced[:, 0],
    X_reduced[:, 1],
    X_reduced[:, 2],
    c=y_train,
    s=40,
)

ax.set(
    title="First three principal components",
    xlabel="1st Principal Component",
    ylabel="2nd Principal Component",
    zlabel="3rd Principal Component",
)
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])

# Add a legend
legend1 = ax.legend(
    scatter.legend_elements()[0],
    ["recurser","octopus"],
    loc="upper right",
    title="Classes",
)
ax.add_artist(legend1)

plt.show()

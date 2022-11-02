import matplotlib.pyplot as plt
import numpy as np


from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog 

url = 'http://github.com/dknife/ML/raw/main/data/Proj2/faces/'

face_images = []

for i in range(15):
  file = url + 'img{0:02d}.jpg'.format(i+1)
  img = imread(file)
  img = resize(img, (64,64))
  face_images.append(img)

def plot_images(nRow, nCol, img):
  fig = plt.figure()
  fig, ax = plt.subplots(nRow, nCol, figsize = (nCol, nRow))
  for i in range(nRow):
    for j in range(nCol):
      if nRow <= 1: axis = ax[j]
      else: axis = ax[i, j]
      axis.get_xaxis().set_visible(False)
      axis.get_yaxis().set_visible(False)
      axis.imshow(img[i*nCol+j])

plot_images(3,5, face_images)

face_hogs = []
face_features = []

for i in range(15):
  hog_desc, hog_image = hog(face_images[i], orientations=8,
                            pixels_per_cell=(16, 16), cells_per_block=(1, 1),
                            visualize=True, multichannel=True)
  face_hogs.append(hog_image)
  face_features.append(hog_desc)

plot_images(3,5, face_hogs)

print(face_features[0].shape)

fig = plt.figure()
fig, ax = plt.subplots(3,5, figsize = (10,6))
for i in range(3):
  for j in range(5):
    ax[i, j].imshow( resize(face_features[i*5+j], (128,16)))
    
url = 'https://github.com/dknife/ML/raw/main/data/Proj2/animals/'
animal_images = []

for i in range(15):
  file = url + 'img{0:02d}.jpg'.format(i+1)
  img = imread(file)
  img = resize(img, (64,64))
  animal_images.append(img)

plot_images(3,5, animal_images)

animal_hogs = []
animal_features = []

for i in range(15):
  hog_desc, hog_image = hog(animal_images[i], orientations=8,
                            pixels_per_cell=(16, 16), cells_per_block=(1, 1),
                            visualize=True, multichannel=True)
  animal_hogs.append(hog_image)
  animal_features.append(hog_desc)

plot_images(3,5, animal_hogs)

fig = plt.figure()
fig, ax = plt.subplots(3,5, figsize = (10,6))
for i in range(3):
  for j in range(5):
    ax[i, j].imshow(resize(animal_features[i*5+j], (128,16)))
    
X, y = [], []

for feature in face_features:
  X.append(feature)
  y.append(1)
for feature in animal_features:
  X.append(feature)
  y.append(0)
print(y)

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

polynomial_svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(C=1, kernel = 'poly', degree=5, coef0=10.0))
])
polynomial_svm_clf.fit(X, y)

url = 'https://github.com/dknife/ML/raw/main/data/Proj2/test_data/'

test_images = []

for i in range(10):
  file = url + 'img{0:02d}.jpg'.format(i+1)
  img = imread(file)
  img = resize(img, (64,64))
  test_images.append(img)

plot_images(2,5, test_images)

test_features = []
for i in range(10):
  hog_desc, hog_image = hog(test_images[i], orientations=8,
                            pixels_per_cell=(16, 16), cells_per_block=(1, 1),
                            visualize=True, multichannel=True)
  test_features.append(hog_desc)

test_result = polynomial_svm_clf.predict(test_features)
print(test_result)

from scipy.ndimage import interpolation
fig = plt.figure()
fig, ax = plt.subplots(2,5, figsize = (10,4))
for i in range(2):
  for j in range(5):
    ax[i, j].get_xaxis().set_visible(False)
    ax[i, j].get_yaxis().set_visible(False)
    if test_result[i*5+j] == 1:
      ax[i, j].imshow(test_images[i*5+j],interpolation='nearest')

url = 'https://github.com/maroonix/202216718/raw/main/sec_image/'

sec_images = []

for i in range(10):
  file = url + 'img{0:02d}.jpg'.format(i+1)
  img = imread(file)
  img = resize(img, (64,64))
  sec_images.append(img)

plot_images(2,5, sec_images)

sec_hogs = []
sec_features = []

for i in range(10):
  hog_desc, hog_image = hog(sec_images[i], orientations=8,
                            pixels_per_cell=(16, 16), cells_per_block=(1, 1),
                            visualize=True, multichannel=True)
  sec_hogs.append(hog_image)
  sec_features.append(hog_desc)

plot_images(2,5, sec_hogs)

fig = plt.figure()
fig, ax = plt.subplots(2,5, figsize = (10,4))
for i in range(2):
  for j in range(5):
    ax[i, j].imshow(resize(sec_features[i*5+j], (128,16)))
    
sec_features = []
for i in range(10):
  hog_desc, hog_image = hog(sec_images[i], orientations=8,
                            pixels_per_cell=(16, 16), cells_per_block=(1, 1),
                            visualize=True, multichannel=True)
  sec_features.append(hog_desc)

sec_result = polynomial_svm_clf.predict(sec_features)
print(sec_result)

fig = plt.figure()
fig, ax = plt.subplots(2,5, figsize = (10,4))
for i in range(2):
  for j in range(5):
    ax[i, j].get_xaxis().set_visible(False)
    ax[i, j].get_yaxis().set_visible(False)
    if sec_result[i*5+j] == 1:
      ax[i, j].imshow(sec_images[i*5+j],interpolation='nearest')

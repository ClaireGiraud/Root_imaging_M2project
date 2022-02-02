# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 09:53:41 2022

This script tests the thresholding and K-means segmentation methods.

@author: Claire Giraud
"""
### Import packages ##########################

from PIL import Image
from skimage import segmentation
from skimage import filters
from skimage.util import img_as_int
from skimage.filters import try_all_threshold
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
from PIL import ImageEnhance
import numpy as np
import cv2

### Script ##########################
### Thresholding

# Grayscale import of the image with PIL
img = Image.open("00.Datasets/Initial/Plant_2/Plant_2_2020-10-26.jpg") # open colour image
imgGray = img.convert('1') # convert image to black and white
plt.imshow(imgGray)
plt.show()

# Conversion of the PIL object to a numpy array
img_array = np.array(imgGray, dtype=float)

## Different thresholding methods are tried
img_array = 255 * img_array
img = img_array.astype(np.uint8)

fig, ax = try_all_threshold(img, figsize=(10, 8), verbose=False)
plt.show()

## Try adaptive

# Reimport the photo in the right format
img = Image.open("00.Datasets/Initial/Plant_2/Plant_2_2020-10-26.jpg") # open colour image
imgGray = img.convert('1') # convert image to black and white

# Conversion of the PIL object to a numpy array
img_array = np.array(imgGray, dtype=float)
adaptive_threshold = filters.threshold_local(img_array, 151)
adaptive_threshold = img_as_int(adaptive_threshold)
clear_image = segmentation.clear_border(adaptive_threshold)
plt.imshow(adaptive_threshold)
plt.show()

###############################################################################
###############################################################################
### Segmentation with K-means

# Importation
img2 = Image.open('00.Datasets/Initial/Plant_2/Plant_2_2020-10-26.jpg')

# Pretreatment
img2 = img2.rotate(90, expand=True) # Rotation
width, height = img2.size #Get the size
area = (1 * width / 4.5, 0, 3.5 * width / 4, height)  # left top right bottom
img2 = img2.crop(area)# crop the image around the frame
enh = ImageEnhance.Contrast(img2) # Change contrast
img2 = enh.enhance(1.5) # Parameters that changes the contrast

# Show the picture
plt.imshow(img2)
plt.show()

# Preparation for k-means
new_img = np.array(img2)
# Shaping into a 2D image of float32
twoDimage = new_img.reshape((-1, 3))
twoDimage = np.float32(twoDimage)

# Criteria for kmeans: stopping, maximum iteration number, number of classes

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 4
attempts = 10

# The labels can be retrieved but it's a bit long for a single image.
ret, label, center = cv2.kmeans(twoDimage, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)

center = np.uint8(center)
res = center[label.flatten()]

# Creating a new image with Kmeans results
result_image = res.reshape((new_img.shape))

# Show the result
plt.imshow(result_image)
plt.title('Results k-means')
plt.show()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 14:35:40 2021
Script implementing segmentation by having an autoencoder denoise the images
@author: tom-h
"""

### Packages ##################################################################
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import cv2
import os

### Functions #################################################################
def preprocess(array, width, height):
    """
    Normalizes the supplied array and reshapes it into the appropriate format.
    Taken fr
    """

    array = array.astype("float32") / 255.0 #conversion en type float 32 et normalisation
    # Valeur des pixels comprise entre 0 et 1.
    
    array = np.reshape(array, (len(array), width, height, 3))
   
    return array
def display(array1, array2, width, height, title=False):
    '''
    Displays ten random images from each one of the supplied arrays.
    Taken from https://keras.io/examples/vision/autoencoder/
    But modified

    Parameters
    ----------
    array1 : np.ndarray
        Array of the first row of images to display
    array2 : np.ndarray
        Array of the second row of images to display
    width : int
        images' width
    height : TYPE
        images' height
    title : string, optional
        list of strings (titles to give to each row of images).
        The default is False (no titles)

    '''
    
    n = 10
    
    images1 = array1[:n]
    images2 = array2[:n]

    plt.figure(figsize=(20, 4))
    for i, (image1, image2) in enumerate(zip(images1, images2)):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(image1.reshape(width, height, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        if title:
            plt.title(title[0])

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(image2.reshape(width, height, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        if title:
            plt.title(title[1])

    plt.show()

def is_test(string, names_test):
    '''

    Parameters
    ----------
    string : string
        an Image filename
    names_test : list
        list of all photos' number that are to be in the test dataset

    Returns
    -------
    is_test : boolean
        Indicates whether the current image (whose number is string) shall be 
        put in the test dataset or not (train)

    '''
    
    i=0
    is_test = False #init to False
    
    #Going through names_test and stopping when our string is found in that list
    while not is_test and i<len(names_test):
        is_test = (names_test[i] in string)
        i+=1
        
    return is_test

### Script ####################################################################

### DATA IMPORTATION
# Empty list for data importation
photos, masques, filenames = [],[],[]

#defining image size
im_size = (360, 360)

path = '00.Datasets/modified/blackroots/'

#iterating over images
for filepath in os.listdir(path+'Photo'):
    
    #reading images
    img=cv2.imread(path+f'Photo/{filepath}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # resizing so that it can run on this laptop.
    res = cv2.resize(img, im_size, interpolation=cv2.INTER_AREA)
    
    #appending to data lists
    photos.append(res)
    filenames.append(filepath)
    
#same as above for masks
for filepath in os.listdir(path+'Masque'):
    img=cv2.imread(path+f'Masque/{filepath}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # resizing so that it can run on this laptop.
    res = cv2.resize(img, im_size, interpolation=cv2.INTER_AREA)
    masques.append(res)

# we collect the dimensions of the np array (without the number of line)
width, height = im_size[:]

#list to np array with number image equals number of lines
photos = np.asarray(photos)
masques = np.asarray(masques)

photos = preprocess(photos, width, height)
masques = preprocess(masques, width, height)

photos_train, photos_test = photos[:55], photos[55:]
masques_train, masques_test = masques[:55], masques[55:]


### DEFINING THE MODEL
# Input layer : images dimensions width, height
input_ = layers.Input(shape=(width, height, 3))

# Encoder
x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input_)
x = layers.MaxPooling2D((2, 2), padding="same")(x)

# Decoder
x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2D(3, (3, 3), activation="sigmoid", padding="same")(x)

# Autoencoder
autoencoder = Model(input_, x)
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
autoencoder.summary()


### 1st TRAINING PHASE : verifying that the autoencoder knows how to reproduce
#the input images
history = autoencoder.fit(
    x=photos_train,
    y=photos_train,
    epochs=50,
    batch_size=12,
    shuffle=True,
    validation_data=(photos_test, photos_test),
)

#plotting loss graph
plt.figure()
plt.plot(history.history['loss'])
plt.title('Evolution of loss across epochs')
plt.ylabel('loss')
plt.ylim(0,1)
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#predictions
predictions = autoencoder.predict(photos_test)

#display
display(photos_test, predictions, width, height,
        title=['Photo de départ', 'Prédiction'])


### 2ND TRAINING PHASE : teaching the autoencoder to denoise the input images

#initialisation of the lists
testImages, testMasks, trainImages, trainMasks = [],[],[],[]

#List of the image files' original numbers that should be in the test dataset
# (in order to have a common test dataset between our different segmentation
# method (comparison))
names_test = ['5564', '5598', '5622', '5625', '5627', '5630', '5653',
              '5655', '5658', '5664']

for i in range(len(photos)):
    
    #thanks to this call to the 'is_test' function, the user can manually choose
    #which photos are to be on the test sample (based on the original image nb)
    if is_test(filenames[i]):
        testImages.append(photos[i])
        testMasks.append(masques[i])
        
    else:
        trainImages.append(photos[i])
        trainMasks.append(masques[i])
        
#preprocessing
testImages, testMasks = np.asarray(testImages), np.asarray(testMasks)
trainImages, trainMasks = np.asarray(trainImages), np.asarray(trainMasks)

#fitting the autoencoder
history = autoencoder.fit(
    x=trainImages,
    y=trainMasks,
    epochs=50,
    batch_size=12,
    shuffle=True,
    validation_data=(testImages, testMasks),
)

#plotting loss graph
plt.figure()
plt.plot(history.history['loss'])
plt.title('Evolution of loss across epochs')
plt.ylabel('loss')
plt.ylim(0,1)
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#predictions
predictions = autoencoder.predict(testImages)

#display
display(testImages, predictions, width, height,
        title=['Photo de départ', 'Prédiction'])

#writing the 10 predictions and their corresponding masks, images
path = '07.Compare_F1_set/res_autoencoder/'
for i in range(10):
    cv2.imwrite(path+f'photos/photo{i}.jpg', cv2.cvtColor(testImages[i]*255,
                                                                  cv2.COLOR_RGB2BGR))
    cv2.imwrite(path+f'labels/label{i}.jpg', cv2.cvtColor(testMasks[i]*255,
                                                                  cv2.COLOR_RGB2BGR))
    cv2.imwrite(path+f'pred/pred{i}.jpg', cv2.cvtColor(predictions[i]*255,
                                                                  cv2.COLOR_RGB2BGR))


### FINAL PREDICTIONS 

#path to the dataset (sorted images)
path_final_pred = '05.Autoencoder/predictions/'

#iterating over the plant/rhizobox folders
for i,dir_ in enumerate(os.listdir(path_final_pred)):
    print(f'\nProcessing {dir_} {i+1}/{len(os.listdir(path_final_pred))}...')
    
    #refreshing the path
    path_img = path_final_pred+'/'+dir_+'/Photos/'

    try:
        os.listdir(path_img)
    except:
        path_img = path_final_pred+'/'+dir_+'/Photo/'

    #iterating over the images in the plant directory
    for k, photo, in enumerate(os.listdir(path_img)):
        print(f'--- Processing {photo} {k+1}/{len(os.listdir(path_img))}...')
        
        #preprocessing
        im_size = (360, 360)
        
        #reading the image
        img = cv2.imread(path_img+photo)
        
        #preprocessing to feed the autoencoder
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img = cv2.resize(img, im_size, interpolation=cv2.INTER_AREA)
        img = np.asarray([img])
        width, height = im_size[:]
        img = preprocess(img, width, height)
        
        #prediction
        pred = autoencoder.predict(img)
        
        _, width, height, _ = pred.shape
        pred = pred.reshape(width, height, 3)
        
        #post-processing bedore saving the image
        pred = cv2.cvtColor(pred*255,cv2.COLOR_RGB2BGR)
        
        #saving the image to the appropriate location
        path_save = path_final_pred+'/'+dir_+'/Mask_pred/'
        pred_name = 'pred_'+photo

        cv2.imwrite(path_save+pred_name, pred)
        
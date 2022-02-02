# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 15:45:06 2021
Method 3 for the automatic dataset ordering (reading the rhizobox's number)
@author: tom-h
"""

### Importing packages #######################################################
import cv2
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os

### Function #################################################################

def preprocessing(path, dest_dir):
    '''

    Parameters
    ----------
    path : string
        path to the directory containing the images
    dest_dir : string
        path to the directory in which save the pre-processed images

    Returns
    -------
    None.

    '''
    
    #if the destination directory doesnt exist, it is created
    if not os.path.exists(f'{dest_dir}'):
        os.mkdir(f'{dest_dir}')
    
    #looping over the image files in the directory
    for image_file in os.listdir(path):
    
        #importing initial image
        img_init = cv2.imread(path+'/'+image_file)
        img_init = cv2.cvtColor(img_init, cv2.COLOR_BGR2GRAY)
    
        #systematic crop
        width, height, _ = img_init.shape
        x1, x2, y1, y2 = 0,  int(height/5), 0, int(width/3)
        img_cropped = img_init[x1:x2, y1:y2]
        
        #normalisation
        img_norm = img_cropped/255
        
        #binarization
        img_t = cv2.threshold(img_norm, 0.5, 1, cv2.THRESH_BINARY)[1]
        
        #saving the new processed image
        cv2.imwrite(f'{dest_dir}/{image_file}', img_t*255)
    
def make_model(input_shape):
    '''builds a CNN model, based on an input shape'''

    model = tf.keras.Sequential([
        
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.5),
        
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.5),

        layers.Flatten(),
        
        layers.Dense(80, activation='relu'),        
        layers.Dropout(0.5),

        layers.Dense(56, activation='softmax')
        
        ])
    
    model.build(input_shape=input_shape)
    model.summary()

    return model

##### data augmentation layer 
data_augmentation = tf.keras.Sequential(

    [
        # layers.Rescaling(1./255)
        # layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        # layers.experimental.preprocessing.RandomRotation(0.1),
        # layers.GaussianNoise(1),
        # layers.experimental.preprocessing.RandomZoom(0.2),
        # layers.experimental.preprocessing.RandomContrast(0.5)
    ]
)

### Main Script ##############################################################

##### 1. Generating the train dataset by processing the raw images
path = '00.Datasets/Sample_not_sorted'
dest_dir = '' #to decide!
preprocessing(path, dest_dir)

##### 2. Training phase
image_size = (512,512)
batch_size = 32

input_shape = (batch_size,) + image_size + (1,)

#importing the pre-processed data
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dest_dir,
    validation_split=0.25,
    seed=123,
    subset="training",
    image_size=image_size,
    batch_size=batch_size,
    color_mode = 'grayscale',
    crop_to_aspect_ratio = True
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dest_dir,
    validation_split=0.25,
    seed=123,
    subset="validation",
    image_size=image_size,
    batch_size=batch_size,
    color_mode = 'grayscale',
    crop_to_aspect_ratio = True
)

    
##### visualisation of the image data
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"), 
                   cmap='gray',vmin=0, vmax=255)
        plt.title(int(labels[i]))
        plt.axis("off")
        
        
##### building the model
model = make_model(input_shape=input_shape)

#compiling the model
model.compile(optimizer='adam',
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

#fitting the model
epochs = 10
history = model.fit(train_ds, epochs=epochs, validation_data=val_ds, verbose=1)

#summarizing the training's history for accuracy
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.ylim(0,1)
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

print(model.evaluate(val_ds))

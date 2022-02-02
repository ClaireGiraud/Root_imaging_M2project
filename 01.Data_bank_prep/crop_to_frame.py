# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 15:15:43 2022
Script implementing a method to automatically crop the image's zones outside of
the rhizobox
@author: tom-h
"""

### Importing packages #######################################################
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import imageio as io
import cv2
import tensorflow as tf
from tensorflow.keras import layers, metrics

### Functions ################################################################

def generate_bb_annotations(path):
  
    '''
    Parameters
    ----------
    path : string
        Path to the directory containing the images
        
    Function used to annotate the whole dataset. The user is asked to click on:
        1: the top left corner of the rhizobox
        2: the top right corner
        3: the bottom right corner
        4: the bottom left corner
    The program then automatically fills an excel with those coordinates for each
    rhizobox. ( ! This is a long process)
    '''
    
    #if the excel doesn't exist, it is created
    try:
        annot_filename = '01.Data_bank_pretreat/Results/annot_frame_bb.xlsx'
        annot = pd.read_excel(annot_filename)
    except FileNotFoundError:
        header = pd.DataFrame(columns=["image_name","bounding_box"])
        header.to_excel(annot_filename, index=False)
                
    #looping over all raw images
    for img_name in os.listdir(path):
        
        #reading the image
        img = io.imread(path+img_name)
        
        #displaying the image
        plt.imshow(img,interpolation='none')
        plt.axis('image')
        
        #the user is asked to click 4 times
        [lx,ly],[tx,ty],[rx,ry],[bx,by]=plt.ginput(4,timeout=0)
        
        #displaying the clicks
        plt.plot(lx,ly,'xr')
        plt.plot(tx,ty,'xr')
        plt.plot(rx,ry,'xr')
        plt.plot(bx,by,'xr')

        plt.pause(0.1)
        plt.close()
        
        #getting coordinates from the click data
        topleft, topright, botright, botleft = (round(lx),round(ly)), 
        (round(tx),round(ty)),(round(rx),round(ry)),(round(bx),round(by))
        
        #append the information to the df
        annot = annot.append({"image_name":img_name,
                              "bounding_box":[topleft, topright, 
                                              botright, botleft]},
                             ignore_index=True)
    
    #converting the df to excel
    annot.to_excel(annot_filename, index=False)
    return

def loading_data(im_size):
    '''

    Parameters
    ----------
    im_size : tuple
        tuple containing the dimensions of the image (width, height)

    Returns
    -------
    data : list
        Images (x)
    targets : list
        Targets annotations (y)
    filenames : list
        Filenames corresponding to the images

    '''
    
    #initialisation of the lists
    data, targets, filenames = [],[],[]
    
    #reading the annotation excel file
    df = pd.read_excel('Results/annot_frame_bb.xlsx')
    
    #iterating over the excel's rows
    for i, row in df.iterrows():
        
        print(f' Loading data : {i}/{len(df)}')
        
        #getting filename in the excel
        filename = row[0]
        
        #reading the corresponding image
        image = cv2.imread(f'data1/{filename}')
        #resizing to the im_size input parameter
        image = cv2.resize(image, im_size, interpolation = cv2.INTER_AREA)

        #getting bounding box from the excel (annotation)
        bb = eval(row[1])
        
        #simplifying the coordinates : left, top, right, bottom
        topleft, topright, botright, botleft = bb[:]
        left, top, right, bottom = min(topleft[0],botleft[0]), min(topleft[1],topright[1]),max(topright[0],botright[0]),max(botleft[1],botright[1])
        
        #converting the coordinates to proportions of the im_size
        h, w = image.shape[:2]
        
        left = int(left)/w
        top = int(top)/h
        right = int(right)/w
        bottom = int(bottom)/h
        
        #appending objects to the corresponding lists
        filenames.append(filename)
        data.append(image)
        targets.append([left,top,right,bottom])

    return data, targets, filenames

def make_model(input_shape):
    '''builds a CNN model, based on an input shape'''

    model = tf.keras.Sequential([
                
        layers.Conv2D(6, 5, activation='relu', padding='same'),
        layers.AveragePooling2D((2,2)),

        layers.Conv2D(16, 5, activation='relu', padding='same'),
        layers.AveragePooling2D((2,2)),

        layers.Conv2D(32, 5, activation='relu', padding='same'),
        layers.AveragePooling2D((2,2)),
        layers.Dropout(0.5),
        
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.AveragePooling2D((2,2)),
        layers.Dropout(0.5),
        

        layers.Flatten(),
        
        layers.Dense(120, activation='relu'),        
        layers.Dense(84, activation='relu'),
        layers.Dropout(0.5),

        layers.Dense(4, activation='sigmoid')
        
        ])
    
    model.build(input_shape=input_shape)
    model.summary()

    return model

### Script ###################################################################


### 1. Generate the train dataset : raw images + annotations of the bounding
### box defining the rhizobox

path = '00.Datasets/Sample_not_sorted'

#call to a function allowing the user to manually annotate all raw images
generate_bb_annotations(path)

### 2. Training the model

#parameters
image_size = (10,10)
batch_size = 10
input_shape = (batch_size,) + image_size + (3,)

#loading data by calling the corresponding function
data, targets, filenames = loading_data(image_size)

split = train_test_split(data, targets, filenames, test_size=0.10, random_state=42)
(trainImages, testImages) = split[:2]
(trainTargets, testTargets) = split[2:4]
(trainFilenames, testFilenames) = split[4:]

#visualisation of the data
plt.imshow(trainImages[0])

#making the model
model = make_model(input_shape=input_shape)

#compiling the model
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=[
                  metrics.MeanSquaredError()
                  ]
              )

#fitting the model
val_ds = [(x,y) for x in testImages for y in testTargets]
epochs = 10
history = model.fit(x=trainImages, y=trainTargets, epochs=epochs, 
                    validation_data=val_ds)

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

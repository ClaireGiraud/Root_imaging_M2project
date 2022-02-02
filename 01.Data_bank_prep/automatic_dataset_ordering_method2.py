# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 10:26:43 2022
Method 2 for the automatic dataset ordering (reading the rhizobox's number)
@author: tom-h
"""

### Importing modules ########################################################
import cv2
import numpy as np
from keras.models import model_from_json
import matplotlib.pyplot as plt
import os
import pytesseract
import re
import sys

#path to the tesseract executable, after installation -> you have to modify
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\tom-h\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

#adding path to the Yolo directory
sys.path.insert(0, 'Text-Detection-using-Yolo-Algorithm-in-keras-tensorflow-master')

#importing the Utils module from Text-Detection-using-Yolo-Algorithm-in-keras-tensorflow-master
from Utils import *

### Functions ################################################################

### FUNCTIONS taken from Text-Detection-using-Yolo-Algorithm-in-keras-tensorflow-master
def load_model(strr): 
    '''taken from Text-Detection-using-Yolo-Algorithm-in-keras-tensorflow-master'''
    json_file = open(strr, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    return loaded_model
    
def predict_func(model , inp , iou , name):
    '''taken from Text-Detection-using-Yolo-Algorithm-in-keras-tensorflow-master'''
    '''predicts the boxes' locations and optionally saves the image'''

    ans = model.predict(inp)
    boxes = decode_boxes(ans[0] , img_w , img_h , iou)
    
    img = ((inp + 1)/2)
    img = img[0]
    
    for i in boxes:

        i = [int(x) for x in i]
        img = cv2.rectangle(img , (i[0] ,i[1]) , (i[2] , i[3]) , 
                            color = (0,255,0) , thickness = 2)

    if name:
        cv2.imwrite(os.path.join('Results' , str(name) + '.jpg') , img*255.0)
   
def predict_boxes(model, inp, iou):
    '''taken from Text-Detection-using-Yolo-Algorithm-in-keras-tensorflow-master'''
    ''' predicts boxes containing text from an image'''
    ans = model.predict(inp)
    boxes = decode_boxes(ans[0] , img_w , img_h , iou)
    return boxes

def plot_boxes(img, boxes):
    '''taken from Text-Detection-using-Yolo-Algorithm-in-keras-tensorflow-master'''
    '''plots the big box on the image'''
    
    img = ((img + 1)/2)
    img = img[0]
    
    for box in boxes:
    
        i = [int(x) for x in box]
    
        img = cv2.rectangle(img , (i[0] ,i[1]) , (i[2] , i[3]) , 
                            color = (0,0,255) , thickness = 2)
    plt.figure()
    plt.imshow(img)

### Self-written functions
def get_big_box(boxes, valx=130, valy=100):
    """
    generates a frame containing the whole tag from text boxes
    
    Parameters
    ----------
    boxes : list
        list of the boxes' coordinates
    valx : int, optional
        width of the big box. The default is 130.
    valy : int, optional
        height of the big box. The default is 100.

    Returns
    -------
    big_box
        list of the coordinates of the big box, englobing the whole tag.

    """
    
    L_centers = [] #initialisation of the L_centers
    for box in boxes: #coordinates of the boxes's centers
        L_centers.append(((box[2]+box[0])/2, (box[3]+box[1])/2))
        
    #calculating the average center of the boxes
    x,y = [sum(ele) / len(L_centers) for ele in zip(*L_centers)]
    
    #defining the big box
    big_box = [x-valx,y-valy,x+valx,y+valy]
    
    #taking potential coordinates outside the image's dimensions
    if x-valx<0:
        big_box[0] = 0
        big_box[2] = valx*2
        
    if y-valy<0:
        big_box[1] = 0
        big_box[3] = valy*2
        
    return tuple(big_box)
        
def plot_bbox(img, box):
    '''plots the big box on the image'''
    
    img = ((img + 1)/2)
    img = img[0]
    
    i = [int(x) for x in box]

    img = cv2.rectangle(img , (i[0] ,i[1]) , (i[2] , i[3]) , 
                        color = (0,0,255) , thickness = 2)
    plt.figure()
    plt.imshow(img)

def detect_boxes_from_image(img_init, image_file, display):
    """

    Parameters
    ----------
    img_init : np.ndarray
        Input image to read boxes from
    image_file : string
        Image filename
    display : boolean
        Optional display of the processed images.

    Returns
    -------
    boxes : list or NoneType
        List of tuples containing the boxes coordinates on the image or None if
        no boxes were found
    tag_image : np.ndarray
        Corresponding image

    """
    
    #width and height of the initial image
    width_init, height_init, _ = img_init.shape
    
    #resizing to 512² to feed the YOLO model
    img = cv2.resize(img_init,(512,512))
    img = (img - 127.5)/127.5    
    
    #predicting boxes to get an average idea of where the tag is located
    boxes = predict_boxes(model, np.expand_dims(img,axis= 0) , 0.5)
    
    #optional display
    if display:
        plot_boxes(np.expand_dims(img,axis= 0), boxes)
    
    tag_image = None #initilisation to None
    
    if boxes == []:
        print('empty boxes')
        
    #if at least one box was detected
    else:
        # *** : this choice of valx and valy assures that the final crop 
        # size will be 512²
        valx=(512**2)/(2*height_init)
        valy=(512**2)/(2*width_init)
        
        #once we have the approximate position of the tag on the image, we will 
        #be getting a "big box" that frames the whole tag
        big_box = get_big_box(boxes, valx, valy)
        
        #optional display
        if display:
            plot_bbox(np.expand_dims(img,axis= 0),big_box)
        
        #converting coordinates (left, top, right, bottom) to int
        big_box = [int(e) for e in big_box]
            
        #multiplying coefficients to calculate the tag's position on the 
        #original image
        ty, tx= width_init/512, height_init/512
        
        #getting big box that frames the tag on the whole image (initial image)
        x1,y1,x2,y2 = big_box
        big_box_init = [int(x1*tx),int(y1*ty),int(x2*tx),int(y2*ty)]
        
        #cropping the original image to the frame -> its size will be 512² 
        #(see the *** step)
        x1,y1,x2,y2 = big_box_init
        tag_img_init = img_init[y1:y2, x1:x2]
        
        #resizing to 512² nonetheless (in case the value changed slightly 
        #bc of rounding up)
        tag_image = cv2.resize(tag_img_init,(512,512))
        tag_image = (tag_image - 127.5)/127.5   
        
        #optional display
        if display:
            plt.figure()
            plt.imshow(tag_image)
        
        #detecting text boxes on this new image
        boxes = predict_boxes(model, np.expand_dims(tag_image,axis= 0),
                              0.5)
        #converting to int
        boxes_int = []
        for box in boxes:
            box_int = [int(e) for e in box]
            boxes_int.append(box_int)
        boxes = boxes_int
        
        #optional display
        if display:
            plot_boxes(np.expand_dims(tag_image,axis= 0), boxes)

    return boxes, tag_image

def filter_digits(L): 
    
    '''
    
    Parameters
    ----------
    L : list
        A list containing strings to sort in order to find out potential digits

    Returns
    -------
    nb : string
        The digit(s) found.

    '''
    
    #looking for potential single digits in the list (ex: 1)
    L_1digit = [re.findall(r'\d', str(e)) for e in L if e!='' or e==None]
    L_1digit = [int(e[0]) for e in L_1digit if e != [] ]

    #looking for potential double digits in the list (ex: 11)
    L_2digits = [re.findall(r'\d\d', str(e)) for e in L if e!='' or e==None]
    L_2digits = [int(e[0]) for e in L_2digits if e != [] ]

    nb = None     #initialisation to None
    
    if L_1digit != []:
        
        #the number with the most occurences in the list
        nb = max(L_1digit,key=L_1digit.count) 
        
        if L_2digits != []:
            #there are more chances that the nb is composed of two digits, so 
            #we replace the former value of nb to the following (see below)
            
            #the number with the most occurences in the list
            nb = max(L_2digits,key=L_2digits.count)
            
    return nb

def determine_nb(image, display=False):   
    '''

    Parameters
    ----------
    image : numpy.ndarray
        The image from which the number will be read by OCR.
    display : boolean, optional
        Option to display some of the processed images. The default is False.

    Returns
    -------
    string : string
        The number found on the image.

    '''
    
    #converting the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    #calculating the mean of all pixels in the image
    mean = round(cv2.mean(gray)[0], 1)
    
    #list of "thresholded images", based on the mean. The threshold varies from
    #mean - 0.2 to mean + 0.2. Doing so allows the predictions to be more precise, 
    #otherwise it varies too much on the threshold
    threshs = [cv2.threshold(gray, i, np.max(gray), cv2.THRESH_BINARY)[1]
              for i in [mean-0.2, mean-0.1, mean, mean+0.1, mean+0.2]]
    
    #reading the "thresholded images" with the Tesseract engine
    strings = [pytesseract.image_to_string(t, config='--psm 7').strip().replace('\n','')
              for t in threshs]
    
    #filtering digits from the list of strings
    string = filter_digits(strings)
    
    #optional display
    if display:
        plt.figure()
        plt.imshow(image)
        
        plt.figure()
        plt.imshow(threshs[1], cmap='gray')
    
    return string

def read_nb_from_boxes(img, boxes, display=False):
    """

    Parameters
    ----------
    img : np.ndarray
        input image to read the number from
    boxes : list
        list of the coordinates of the boxes
    display : boolean, optional
        Optional display of the processed images. The default is False.

    Returns
    -------
    nb : string
        The number read from the image.

    """
    
    res = [] #initialisation of the list of results
    for box in boxes:
        
        #for each box, we crop the corresponding portion of the image
        #we double the crop's length, to have higher chances of getting the nb
        crop = img[box[1]:box[3], box[0]:(2*box[2]-box[0])]         
        crop = crop.astype('uint8')*255
        
        #determining the nb
        res.append(determine_nb(crop, display))
        
    #filtering digits
    nb = filter_digits(res)
    
    return nb

def method2(img, image_file, display=False):
    """
    
    Parameters
    ----------
    img : np.ndarray
        The image on which to apply method 2
    image_file : string
        Filename of the processed image
    display : boolean, optional
        Optinal display of the processed images. The default is False.
    
    Returns
    -------
    nb : string
        The tag number read from the rhizobox image.
    
    """
    
     #converting from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    #first crop
    width, height, _ = img.shape
    x1, x2, y1, y2 = 0,  int(height/4), 0, int(width/3)
    img_cropped = img[x1:x2, y1:y2]
     
    #optional display
    if display:
        plt.figure()
        plt.imshow(img)
         
        plt.figure()
        plt.imshow(img_cropped)
     
    #First step : detecting text boxes from the image
    boxes, img = detect_boxes_from_image(img_cropped, image_file, display)
      
    if type(img) == np.ndarray: #if some text boxes were found
 
        #reading the number from the boxes by OCR
        nb = read_nb_from_boxes(img, boxes, display)
   
    else: #if no text boxes were found
        nb = 'empty boxes'
    
    return nb

def main(path_dir, model, display=False):
    '''

    Parameters
    ----------
    path_dir : string
        Path to the directory containing the images
    model : YOLO model
    display : boolean, optional
        Optional display of the processed images. The default is False.

    Returns
    -------
    dic : dict
        Dictionary containing the results

    '''
    
    dic = {} #initialisation of the dict object
    
    #loop over all image files in the directory
    for i, image_file in enumerate(os.listdir(path_dir)):
        # print(f'Processing {image_file} -- {i+1}/{len(os.listdir(path_dir))}')
        
        #importing initial image
        img_init = cv2.imread(os.path.join(path_dir,image_file))
        
        #reading number with method 2
        nb = method2(img_init, image_file, display)
        
        #Appending the results to the dictionary
        dic[image_file] = nb
        
        print(nb)
            
    return dic
  
    
### Script ####################################################################
#image size for the YOLO model
img_w = 512
img_h = 512

#loading YOLO model and weights
model = load_model('Text-Detection-using-Yolo-Algorithm-in-keras-tensorflow-master/model/text_detect_model.json')
model.load_weights('Text-Detection-using-Yolo-Algorithm-in-keras-tensorflow-master/model/text_detect.h5')
  
if __name__ == '__main__':
    
    #call to the main function  
    dic2 = main('00.Datasets/Sample_not_sorted', model)
    

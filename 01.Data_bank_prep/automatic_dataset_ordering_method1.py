# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 14:59:29 2021
Method 1 for the automatic dataset ordering (reading the rhizobox's number)
@author: tom-h
"""

### Importing packages #######################################################
import numpy as np
import os
import matplotlib.pyplot as plt
import re
import pytesseract
import cv2

#path to the tesseract executable, after installation
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\tom-h\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

### Functions ################################################################

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

def method1(img, display=False):
    """

    Parameters
    ----------
    img : np.ndarray
        The image on which to apply method 1
    display : boolean, optional
        Optinal display of the processed images. The default is False.

    Returns
    -------
    nb : string
        The tag number read from the rhizobox image.

    """
    
    #converting from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #first systematic crop
    width, height, _ = img.shape
    x1, x2, y1, y2 = 0,  int(height/4), 0, int(width/3)
    img_cropped = img[x1:x2, y1:y2]
    
    #optional display
    if display:
        plt.figure()
        plt.imshow(img)
        
        plt.figure()
        plt.imshow(img_cropped)
    
    #method 1 : the image is scanned by a window ([x1:x2, y1:y2]) and the OCR
    #Tesseract engine is applied each time
    width, height, _ = img_cropped.shape
    px=50   #step for the x loop
    py=height//5    #step for the y loop
    res = [] #initialisation of the list of the results
    
    #double loop on the image's dimensions
    for x in range(0,width,px):
        for y in range(0,height,py):  
            
            #an image is cropped from the scanning window
            x1, x2, y1, y2 = x,x+60,y,y+py
            crop = img_cropped[x1:x2, y1:y2]
            
            #appending results to the list
            res.append(determine_nb(crop, display))
         
    #filtering digits from the results of all scanning windows
    nb = filter_digits(res)
    
    return nb

def main(path_dir, display=False):
    '''

    Parameters
    ----------
    path_dir : string
        path to the directory containing the images
    display : boolean, optional
        Optional display of the processed images. The default is False.

    Returns
    -------
    dic : dict
        A dictionary containing the results for all image files.

    '''
    
    dic = {}    #initialisation of the results
    
    #loop over all image files in the target directory
    for i, image_file in enumerate(os.listdir(path_dir)):
        # print(f'Processing {image_file} -- {i+1}/{len(os.listdir(path_dir))}')
        
        #importing initial image
        img_init = cv2.imread(os.path.join(path_dir,image_file))
        
        #Call to the method1 function
        nb = method1(img_init, display)
        
        #appending to the dictionnary
        dic[image_file] = nb
        
        print(nb)
            
    return dic

if __name__ == '__main__':
    
    #call to the main function  
    dic = main('00.Datasets/Sample_not_sorted', display=True)

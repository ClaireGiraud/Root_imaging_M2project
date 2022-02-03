# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 12:09:05 2022
Script allowing the user to convert images to grayscale in a given folder (no
matter the folder's inner architecture). We used this script to convert Unet's
input images to grayscale.
@author: tom-h
"""

### Importing packages #######################################################
import os
import cv2

### Functions ################################################################

def convert_to_gray(path):
    '''

    Parameters
    ----------
    path : string
        Path to the folder in which to convert the images

    '''
    
    #going through all of the folder's inner structure
    for root, dir_, files in os.walk(path):
        
        #for every file encountered
        for name in files:
            
            #definition of the path to the file
            path_file = os.path.join(root,name)

            #there are some hidden directories called '.DS_Store', that we won't
            #take into account
            if '.DS_Store' not in path_file:
                
                #reading the image in grayscale mode
                im = cv2.imread(path_file, cv2.IMREAD_GRAYSCALE)
                
                #writing the new image (replacing the old one)
                cv2.imwrite(path_file, im)
        
    return

### Script ###################################################################

path = '00.Datasets/blackroots/'
# path = '00.Datasets/blackroots_sorted/'

#call to the function
convert_to_gray(path)
 
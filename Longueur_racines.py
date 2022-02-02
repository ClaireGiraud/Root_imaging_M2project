# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 09:53:41 2022

This script computes the length of roots with two different methods
and saves the results as excel files.

@author: Claire Giraud
"""

### Import packages ##########################
from PIL import Image
import os
import pandas as pd
import openpyxl

### Functions ##########################
def Pretreat(path_rep, origine_mask):
    '''
    Import of masks predicted with a model specified
    in the origin_mask argument.
    From a folder containing the pictures,
    this function returns a list of PIl images with their names.
    '''

    # Importing the names of the images into a list in chronological order
    # For a rhizobox create a list with all the pictures
    # Photo 1 is the photo taken on day 1
    # Photo 2 is the photo taken on day 2, etc.
    
    input_img_paths = sorted(
        [
            os.path.join(path_rep, fname)
            for fname in os.listdir(path_rep)
            if fname.endswith(".png")
        ]
    )
    
    L=[]
    # Image binarisation
    if origine_mask in ['U_net','K_means'] : #Different thresholds depending on the origin of the mask
        thresh = 240
    else:
        thresh = 140
        
    fn = lambda x : 255 if x > thresh else 0
    for photo in input_img_paths :
        L.append(Image.open(photo).convert('L').point(fn, mode='1')) #Importing the greyscale image and binarisingn

    # Resize images to 720x720
    L_res = []
    for photo2 in L:
        
        if origine_mask=='U_net' : #Rotation of images if necessary according to the origin of the mask
            photo2=photo2.rotate(90)
            
        if origine_mask=='K_means' : #Rotation of images if necessary according to the origin of the mask
            photo2=photo2.rotate(180)
        
        L_res.append(photo2.resize((720, 720)))
        
    L_res[0].show()
           
    names = [name[-14:-4] for name in input_img_paths]
    return L_res, names


def Calculate_length_pix(path_rhizo, origine_mask):
    '''
    This function calculates the number of black pixels (considered as roots).
    It returns a dictionary with the names of the pictures for each rhizobox and
    the number of black pixels for each image.
    '''
    # Importing the various images
    L, names = Pretreat(path_rhizo, origine_mask)

    width, height = L[1].size
    time = len(L)

    # Definition of the list that will contain all the root lengths of the rhizobox i for each day

    length_roots=[]
    
    for k in range(0, time):
        px = L[k].load() # Upload the pixels
        nb_black_px = 0
        print('photo n°¸', k)
        for i in range(width):
            for j in range(height):
                if px[i,j] != 255:
                    nb_black_px = nb_black_px + 1
        length_roots.append(nb_black_px)
        print('photo n°¸', k, 'la longueur des racines est de', nb_black_px, 'pixels')
        
    return dict(zip(names, length_roots))


def Calculate_length_diff(path_rhizo, origine_mask):
    '''
    This function calculates the difference between the highest and lowest black pixel (considered the root)
    the highest and the lowest pixel on the image.
    It returns a dictionary with the names of the pictures for each rhizobox and
    the number of pixels of difference for each image.
    '''
    
    L, names = Pretreat(path_rhizo, origine_mask)
    
    width, height = L[1].size
    time = len(L)
    
    length_roots=[]
    
    for k in range(0, time):
        px = L[k].load() # Upload the pixels
                
        print('photo n°¸', k)
        
        i = 0
        pix_blanc = True
        
        # We look for the first black pixel
        while pix_blanc and i<width:
            for j in range(height):
                pix = px[i,j]
                if pix != 255:
                    # print('lll',pix, i,j)
                    pix_blanc *= bool(pix == 255)
                
            i+=1

        min_ = i-1
        
        # We look for the last black pixel
        for i in range(width):
            for j in range(height):
                
                if px[i,j] != 255:
                    max_ = i
        
        print(max_,min_)
        length = max_-min_
        length_roots.append(length)
        print('photo n°¸', k, 'la longueur des racines est de', length, 'pixels')

    return dict(zip(names, length_roots))

def export_to_excel(path, method, origine_mask):
    '''
    This function exports an excel file with the root lengths
    for each image of each rhizobox present in the initial file.
    It takes the method argument which specifies which of the two methods defined
    defined above should be used to calculate the root length
    '''
    # Step 1: calculate the length with one of the two methods
    dic = dict()
    for folder in os.listdir(path):
        
        if method == 'pix':
            L =Calculate_length_pix(path+'/'+folder+'/Mask_pred', origine_mask)
        if method == 'diff':
            L =Calculate_length_diff(path+'/'+folder+'/Mask_pred', origine_mask)

        dic[f'rhizo_{folder}'] = L
    
    # Management of missing values
    L_len = [len(dic[rhizo].keys()) for rhizo in dic]
    max_ = max(L_len) #Search for the maximum number of measurements (number of photos) for a rhizobox

    # This rhizobox is considered complete
    index_max = L_len.index(max_) 
    rhizo_complet = list(dic.keys())[index_max]
    L_dates = list(dic[rhizo_complet].keys()) # Finding dates in a dictionary from the complete rhizobox

    # Searching for missing dates in the different rhizoboxes
    for rhizo in dic: 
        if len(dic[rhizo]) < max_:
            
            for date in L_dates:
                if date not in list(dic[rhizo].keys()):
                    # If the date is missing, it is added to the dictionary
                    dic[rhizo][date] = ''
                            
    df_length = pd.DataFrame.from_dict(dic)  # he dictionary is transformed into a dataframe
    df_length.to_excel(f'Roots_length_{origine_mask}_{method}.xlsx', index=L_dates) # Export to excel
    
### Script ##########################
# To be changed according to the origin of the predictions
path = '04.2D_U-net/predictions'
# ori_mask : ['U_net','K_means', 'Autoencoder']
ori_mask = 'U_net' # Model/algo that produced the prediction

export_to_excel(path, 'diff', ori_mask)

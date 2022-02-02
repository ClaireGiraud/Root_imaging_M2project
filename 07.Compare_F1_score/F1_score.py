# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 09:53:41 2022

This script compares the predicted masks and the lablels for each model.

@author: Claire Giraud
"""
### Import packages ############################
from PIL import Image
import os
import pandas as pd
import openpyxl

### Functions ############################
def Pretreat_to_compare(path_label, path_mask):
    '''
    Importing masks and labels.
    From a folder containing the pictures,
    this function returns a list of PIL images with their names.
    '''
    
    # import of the names of the images in a list in the order of the folder
    
    input_mask_paths = sorted(
        [
            os.path.join(path_mask, fname)
            for fname in os.listdir(path_mask)
            if fname.endswith(".png") or fname.endswith(".jpg")
        ]
    )
    
    input_label_paths = sorted(
        [
            os.path.join(path_label, fname)
            for fname in os.listdir(path_label)
            if fname.endswith(".png") or fname.endswith(".jpg")
        ]
    ) 
    
    L_label=[]
    L_mask=[]
    
    # Image binarisation
    thresh = 250
        
    fn = lambda x : 255 if x > thresh else 0
    
    for photo in input_label_paths :
        L_label.append(Image.open(photo).convert('L').point(fn, mode='1')) # import of the greyscale image and binarisation
    
    for img in input_mask_paths :
        L_mask.append(Image.open(img).convert('L').point(fn, mode='1')) # import of the greyscale image and binarisation
    
    # Resize images to 720x720
    L_res_label = []
    L_res_mask = []
    
    for photo2 in L_label:
        L_res_label.append(photo2.resize((720, 720)))
    
    for img2 in L_mask:
        L_res_mask.append(img2.resize((720, 720)))
        
    L_res_label[1].show()
    L_res_mask[1].show()
           
    return L_res_label, L_res_mask


def Build_F1_score(path_label, path_mask):
    '''
    Construction of the confusion matrix for each image and calculation of the F1 score.
    '''
    L_lab, L_pred = Pretreat_to_compare(path_label, path_mask)
    
    width, height = 720, 720
    nb_img = len(L_lab)
    
    L_F1 = []
    
    #For each image we load the pixels
    for k in range(0, nb_img) : 
        pix_lab = L_lab[k].load()
        pix_pred = L_pred[k].load()
        
        F1 = 0 #F1 score for the photo k
        TP = 0 #True positive (root)
        TN = 0 #True negtive (background)
        FN = 0 #False negative (should be root, ex : the edge of the roots)
        FP = 0 #False positive (should be background, ex : the frame)
        
        #We compare every pixel from the mask to the label
        for i in range (width):
            for j in range (height):
                if pix_lab[i,j] == pix_pred[i,j] == 0:
                    TP = TP + 1
                if pix_lab[i,j] == pix_pred[i,j] == 255:
                    TN = TN + 1
                if pix_lab[i,j] == 0 and pix_pred[i,j] == 255:
                    FN = FN + 1
                if pix_lab[i,j] == 255 and pix_pred[i,j] == 0:
                    FP = FP + 1
        print('TP:', TP, 'FN:', FN, 'FP:', FP, 'TN:', TN)
        
        F1 = TP/(TP+0.5*(FN+FP)) #Compute F1 Score
        L_F1.append(F1) #create a list with every F1 score for a set of images from the same model
        
    return L_F1


def create_excel(F1_tot):
    '''
    This function exports an excel file with the F1 scores for each model.
    '''
    dic = {}
    names = ['F_autoencoder', 'F_unet', 'F_kmeans', 'F_keras']

    for i, L in enumerate(F1_tot):
        dic[names[i]] = L

    df = pd.DataFrame.from_dict(dic)  # The dictionary is transformed into a dataframe
    df.to_excel('F1_scores.xlsx')  # Export to excel


### Script ############################
F1_tot = [] #list with all the F1 scores

#Model = autoencoder
path_label_auto = '07.Compare_F1_score/res_autoencoder/labels'
path_mask_auto = '07.Compare_F1_score/res_autoencoder/pred'

F_autoencoder = Build_F1_score(path_label_auto, path_mask_auto)
F1_tot.append(F_autoencoder)

#Model = 2D U_net
path_label_unet = '07.Compare_F1_score/res_2D_unet/labels'
path_mask_unet = '07.Compare_F1_score/res_2D_unet/pred'

F_unet = Build_F1_score(path_label_unet, path_mask_unet)
F1_tot.append(F_unet)

#Model = kmeans
path_label_kmeans = '07.Compare_F1_score/res_k-means/labels'
path_mask_kmeans = '07.Compare_F1_score/res_k-means/pred'

F_kmeans = Build_F1_score(path_label_kmeans, path_mask_kmeans)
F1_tot.append(F_kmeans)

#Model = 3D U_net keras
path_label_keras = '07.Compare_F1_score/res_3D_unet/labels'
path_mask_keras = '07.Compare_F1_score/res_3D_unet/pred'

F_keras = Build_F1_score(path_label_keras, path_mask_keras)
F1_tot.append(F_keras)

create_excel(F1_tot)

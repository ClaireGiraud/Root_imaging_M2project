# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 10:19:23 2022
Comparison of the methods 1 and 2 for the lecture of the tag's number
@author: tom-h
"""

### Importing modules ########################################################
from automatic_dataset_ordering_method1 import *
from automatic_dataset_ordering_method2 import *
import time
import pandas as pd

### Functions ################################################################

def comparison(path_dir, model):
    """

    Parameters
    ----------
    path_dir : string
        Path to the directory where the sorted images are
    model : YOLO model

    Returns
    -------
    dic : dict
        Dictionary containing the results
    L_time1 : list
        List of the times of executions of method 1
    L_time2 : list
        List of the times of executions of method 2

    """
    
    #initialisation of the results containers
    dic = {}
    L_time1, L_time2 = [], []
        
    #loop over all directories
    for k, dir_ in enumerate(os.listdir(path_dir)):
        
        #extracting the rhizobox's "true" number from the filename
        true_nb = int(dir_[6:])
        
        #defining the path to the image
        path_image = os.path.join(path_dir,dir_)
        
        #loop over the images inside the directory
        for i, image_file in enumerate(os.listdir(path_image)[:3]):
            print(f'Processing {image_file} -- {i+1}/{len(os.listdir(path_image))}')
            
            #importing initial image
            full_path = os.path.join(path_image, image_file)
            
            #reading image
            img_init = cv2.imread(full_path)
            
            #rotating image
            img_init = cv2.rotate(img_init, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            #method 1 : calculating time of execution
            t0 = time.time()
            nb1 = method1(img_init)
            time1 = time.time() - t0
            L_time1.append(time1)
            
            #method 2 : calculating time of execution
            t0 = time.time()
            nb2 = method2(img_init,image_file)
            time2 = time.time() - t0
            L_time2.append(time2)

            print([true_nb, nb1, nb2])
            print(time1, time2)
            
            #appending results to the dictionary
            dic[image_file] = [true_nb, nb1, nb2]
                
    return dic, L_time1, L_time2

### Script ####################################################################
  
### Saving the results
#extractings results from the function call
D, L_time1, L_time2 = comparison('00.Datasets/Initial', model)
    
#re-creating a dic with all the results
plant_name, true_nb, nb1, nb2 = [], [], [], []
for plant in D:
    true_nb.append(D[plant][0])
    plant_name.append(plant)
    nb1.append(D[plant][1])
    nb2.append(D[plant][2])
    
dic = {'plant_name':plant_name,
       'true_nb':true_nb,
       'nb1':nb1,
       'nb2':nb2, 
       'time1':L_time1, 
       'time2':L_time2}

#converting dic to pandas dataframe
df = pd.DataFrame.from_dict(dic)

#exporting dataframe to excel file
df.to_excel('01.Data_bank_pretreat/Results/res_comparison_tag.xlsx', 
            index=False)
    
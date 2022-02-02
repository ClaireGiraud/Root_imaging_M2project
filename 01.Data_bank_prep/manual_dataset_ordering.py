# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 10:09:37 2021
Manually ordering our raw dataset (from a file) into named folders and renaming 
images according to the plant number and the date the picture was taken
@author: tom-h
"""

### Importing packages #######################################################
import os
from PIL import Image
import re
import pandas as pd

### Functions ################################################################
def get_date_taken(img):
    '''

    Parameters
    ----------
    img : PIL image
        The PIL Image to read the date from (metadata)

    Returns
    -------
    The date the picture was taken

    '''
    return img._getexif()[36867]

### Main Script ##############################################################

#importing file connecting image name to plant number
#in our case, this file was manually filled, but we explore some methods (1,2,3)
#to automate this process
df = pd.read_excel('01.Data_bank_pretreat/no_rizhobox.xlsx')

# paths to the root and destination directories
root_dir = '00.Datasets/Sample_not_sorted'
dest_dir = '00.Datasets/Initial'

#if the dest dir doesnt exist, it is created
if not os.path.exists(dest_dir):
    os.mkdir(dest_dir)

#iterating over rows of the excel
for index, row in df.iterrows():
    
    print(f'{index}/{len(df)} sorted pictures...')
    
    #extracting info from row
    name, no, flou, a_suppr = row[:]
    path_image = f'{root_dir}/{name}.JPG'
    
    #if the image must be deleted (info specified in the excel)
    #ex : some images are duplicates / corrupted
    if pd.isna(a_suppr):
            
        try:
            #opening image
            img = Image.open(path_image)
            
            #getting date from metadata
            date = get_date_taken(img).split(' ')[0].replace(':','-')

            #if the plant number is not readable from the picture
            if pd.isna(no) or re.findall(r'\d', str(no))==[]:
                image_name = f'Plant_NA_{date}.jpg'

                #creating folder in case it does not exist
                if not os.path.exists(f'{dest_dir}/pas_de_no'):
                    os.mkdir(f'{dest_dir}/pas_de_no')

                #saving image to the appropriate location    
                img.save(f'{dest_dir}/pas_de_no/'+image_name)

            else:
                no = round(no)
                image_name = f'Plant_{no}_{date}.jpg'
                
                ### if the plant number is blurred
                if not pd.isna(flou):
                                        
                    #creating folder in case it does not exist
                    if not os.path.exists(f'{dest_dir}/flou'):
                        os.mkdir(f'{dest_dir}/flou')
                    
                    #saving image to the appropriate location...  
                    img.save(f'{dest_dir}/flou/'+image_name)

                #creating folder in case it does not exist
                if not os.path.exists(f'{dest_dir}/Plant_{no}'):
                    os.mkdir(f'{dest_dir}/Plant_{no}')
                
                ### if the image is a duplicate
                if image_name in os.listdir(f'{dest_dir}/Plant_{no}'):
                    
                    #creating folder in case it does not exist
                    if not os.path.exists(f'{dest_dir}/duplicates'):
                        os.mkdir(f'{dest_dir}/duplicates')
                        
                    #saving image to the appropriate location...  
                    img.save(f'{dest_dir}/duplicates/'+image_name)
                
                ### if the image is 'normal'
                else:
                    #saving image to the appropriate location... based on no!
                    img.save(f'{dest_dir}/Plant_{no}/'+image_name)
                    
        #handling potential errors
        except Exception as e:
            print(e)
            
        



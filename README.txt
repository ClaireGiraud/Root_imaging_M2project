Root_imaging_M2project

Description
This project was carried out as an exercise in the framework of the Master 2 Data Science for Agronomy and Agro-food. 
From a database of images of rhizoboxes, we wanted to automatically extract the length of the roots. For this purpose we tested different methods and models. 
The summary of the project and the approach is available in the document "roots_project_presentation.pdf". 

Datasets and results
The data and results are available on a Google drive at the following address : https://drive.google.com/drive/folders/1plpLPGGV3T9rnVbaQmOP5xwqIPj7qpwK?usp=sharing

The architecture of this repro github is the same as the Google Drive. To use our data with our codes directly: 
- Download the google drive folder 
- Download the GitHub repro 
- Merge the two while keeping the architecture (put the scripts in the same folders). 

The initial unsorted dataset is not fully available for storage reasons (00.Datasets/Sample_not_sorted/). All photos are available in the folder (00.Datasets/Initial/) they have just been sorted. 

For the structuration of the dataset part, you will need to download the YOLO Algorithm :  https://github.com/Neerajj9/Text-Detection-using-Yolo-Algorithm-in-keras-tensorflow and put it at the root of the project.

In order to get OCR working : 
1. Download and install Tesseract OCR on your computer https://sourceforge.net/projects/tesseract-ocr/
2. Specify the path of the exe in the .py file (for example in : 'automatic_dataset_ordering_method1.py')
3. pip install pytesseract
4. import pytesseract

Links, codes and articles related to the project :

1.Yang, G., Pennington, J., Rao, V., Sohl-Dickstein, J. & Schoenholz, S. S. A Mean Field Theory of Batch Normalization. (2019). 
2.Yu, E. M., Iglesias, J. E., Dalca, A. V. & Sabuncu, M. R. An Auto-Encoder Strategy for Adaptive Image Segmentation. arXiv:2004.13903 [cs, eess] (2020). 
3.Adaloglou, N. An overview of Unet architectures for semantic segmentation and biomedical image segmentation. AI Summer https://theaisummer.com/unet-architectures/ (2021). 
4.Huber, J. Batch normalization in 3 levels of understanding. Medium https://towardsdatascience.com/batch-normalization-in-3-levels-of-understanding-14c2da90a338 (2022). 
5.Ioffe, S. & Szegedy, C. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. (2015). 
6.Zhou, X. et al. EAST: An Efficient and Accurate Scene Text Detector. arXiv:1704.03155 [cs] (2017). 
7.Santurkar, S., Tsipras, D., Ilyas, A. & Madry, A. How Does Batch Normalization Help Optimization? (2018). 
8.Alom, M. Z., Yakopcic, C., Taha, T. M. & Asari, V. K. Nuclei Segmentation with Recurrent Residual Convolutional Neural Networks based U-Net (R2U-Net). in NAECON 2018 - IEEE National Aerospace and Electronics Conference 228–233 (2018). doi:10.1109/NAECON.2018.8556686. 
9.Alom, M. Z., Yakopcic, C., Hasan, M., Taha, T. M. & Asari, V. K. Recurrent residual U-Net for medical image segmentation. JMI 6, 014006 (2019). 
10.Smith, A. G., Petersen, J., Selvan, R. & Rasmussen, C. R. Segmentation of roots in soil with U-Net. Plant Methods 16, 13 (2020). 
11.Ronneberger, O., Fischer, P. & Brox, T. U-Net: Convolutional Networks for Biomedical Image Segmentation. in Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015 (eds. Navab, N., Hornegger, J., Wells, W. M. & Frangi, A. F.) 234–241 (Springer International Publishing, 2015). 
12.Falk, T. et al. U-Net: deep learning for cell counting, detection, and morphometry. Nat Methods 16, 67–70 (2019).

- https://github.com/zhixuhao/unet
- https://github.com/Abe404/segmentation_of_roots_in_soil_with_unet
- https://keras.io/examples/vision/oxford_pets_image_segmentation/
- https://keras.io/examples/vision/autoencoder/


### Import the packages ###################
library(SuperpixelImageSegmentation)
library(magick)
library(imager)

### Functions ###################
get_array <- function(list_img, indice_img){
  
  # From an Imlist we need to get the image as an array of dim 3. 
  # input : list of images and the indice of the image we want in the list
  # output : An array of dim 3
  
  dtframe <- as.data.frame(list_img[[indice_img]][])# get the array of image nÃ‚Â°indice in the list of images
  
  w <- dim(dtframe)[1] # get the first dimension
  h <- (dim(dtframe)[2])/4 # get the second one
  img_array <- array(list_img[[indice_img]][], dim = c(w,h,4)) #change the dimension of the array so that they can be an input of the pixel_segmentation function
  
  return(img_array)
}

# Clustering with affinity propagation
# Load all the photo present in the folder and save the mask obtained with the function spixel_segmentation
seg <- function(path_img, sim_color_radius, superpixel) {
  
  list_photos <- load.dir(path_img, pattern = NULL, quiet = FALSE) # loading the images into a Imlist
  
  for (i in 1:length(list_photos)) {
    init = Image_Segmentation$new() 
    im <- get_array(list_photos, i) # using the previous function to get an dim 3 array for all the images
    spx = init$spixel_segmentation(input_image = im, # clustering with kmeans and affinity propagation
                                   superpixel = superpixel, # number of superpixel in the final mask
                                   AP_data = TRUE,
                                   use_median = TRUE, 
                                   sim_wL = 3, 
                                   sim_wA = 10, 
                                   sim_wB = 10,
                                   sim_color_radius = sim_color_radius, # parameter that controls the number of cluster
                                   verbose = FALSE) 
    str(spx)
    OpenImageR::imageShow(spx$AP_image_data) # displaying the image
  }
}

### Script ###################
path_rhizo17= '00.Datasets/modified/blackroots_sorted/N°17' # rhizobox number 17
seg(path_rhizo17,300,500)








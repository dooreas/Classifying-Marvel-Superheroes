#here I try to make a data augmentation program for image sets which is base on uniformly applying images in the data set some gaussian, noise and averaging filters, and rotations

import os
import glob
from scipy.ndimage.interpolation import rotate
from scipy.signal import convolve2d
from cv2 import GaussianBlur
import cv2
from cv2 import rectangle #used in occlusion
from utils import read_img, resize_img
from PIL import Image 
import skimage
import numpy as np
import random

#the document was used for multiple transformations in robustness, and hence it is a little bit messing 

DATA_PATH = 'test'#WE DO DATA PRE-AUGMENTATION IN THIS LOCATION TO BE SAVE
IMAGE_CATEGORIES = [
    'black widow', 'captain america', 'doctor strange', 'hulk', 'ironman',
    'loki', 'spider-man', "thanos"
]
#modification which only gives us the images from the training set 
def get_image_paths_for_each_transform(data_path, categories, num_train_per_cat):
    '''
    This function returns lists containing the file path for each train
    and test image, as well as lists with the label of each train and
    test image. By default both lists will be 1500x1, where each
    entry is a char array (or string).
    '''

    num_categories = len(categories) # number of scene categories.

    # This paths for each training and test image. By default it will have 1500
    # entries (15 categories * 100 training and test examples each)
    train_image_paths = [None] * (num_categories * num_train_per_cat)

    # The name of the category for each training and test image. With the
    # default setup, these arrays will actually be the same, but they are built
    # independently for clarity and ease of modification.
    train_labels = [None] * (num_categories * num_train_per_cat)

    for i,cat in enumerate(categories): #the enumerate give you a list with pairs: (index,category)
        #this takes images for training "train"
        images = glob.glob(os.path.join('train', cat, '*.jpg'))

        for j in range(num_train_per_cat):#we create a list of images
            #paths for the category cat with index i: train_image_paths
            #i*num_categories will locate us in the index of the first
            #image of category cat(i) then with j the num_train_per_cat
            #image's paths that correspong to this category.
            train_image_paths[i * num_train_per_cat + j] = images[j]
            
            #Here we do as in the above line but with categories
            train_labels[i * num_train_per_cat + j] = cat
        
        #the following does the same as the above loop but for images in the test set
        

    return (train_image_paths, train_labels)
 

def transformation_applications(input_,cat, name):
    #to perform transf.
    
    #rename the category so that saving changes works. By searching the presence of the categ name in cat
    for category in IMAGE_CATEGORIES:
        if category in cat: cat=category
    
    #start by coppying an image
    orig_im=resize_img(read_img(input_),size=(224,224))
    new_im = orig_im.copy()
    
    
    
    #salt and peper noise
    #for am in [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18]:
    #    for channel in [0,1,2]:
    #        new_im[channel]=skimage.util.random_noise(new_im[channel], mode="s&p", seed=None, clip=True, amount=am/3)*255  
        #notice that the first two dimensions give the image shape while
        #the 3 stands for the use of the RGB model
        #transforms our array into an image and saves it
    #    transf_im=Image.fromarray(new_im.clip(0, 255).astype(np.uint8))
    #    os.makedirs("C:/Users/RO-24/OneDrive/MSc AI/Image and Vision Computing/Project/valid_with_s&p/"+str(am)+"/"+cat, exist_ok=True)
    #    transf_im.save("C:/Users/RO-24/OneDrive/MSc AI/Image and Vision Computing/Project/valid_with_s&p"+"/"+str(am)+"/"+cat+"/"+name)
        
    
    



    
    #gaussian noise with variance 1
    #for am in [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]:
    #    zero_im=np.zeros(new_im.shape)
    #    new_im = new_im + skimage.util.random_noise(zero_im, mode="gaussian", seed=None, clip=False, var=am**2)
    #    new_im[new_im<0]=0
    #    new_im[new_im>255]=255
    #    transf_im=Image.fromarray(new_im.clip(0, 255).astype(np.uint8))
    #    os.makedirs("C:/Users/RO-24/OneDrive/MSc AI/Image and Vision Computing/Project/valid_with_gaussnois/"+str(am)+"/"+cat, exist_ok=True)
    #    transf_im.save("C:/Users/RO-24/OneDrive/MSc AI/Image and Vision Computing/Project/valid_with_gaussnois"+"/"+str(am)+"/"+cat+"/"+name)    
     
   
   
   
    #gaussian blurring 
    gauss_blurr_ker = np.array([1,2,1,2,4,2,1,2,1]).reshape(3,3)*(1/16)
    for times in [1,2,3,4,5,6,7,8,9]:
        c=1
        while c<=times:
            #print(c)
            #print(new_im.shape)
            #new_im[:,:,0] = convolve2d(new_im[:,:,0],gauss_blurr_ker, mode="same")
            #print(new_im[0])
            #new_im[:,:,1] = convolve2d(new_im[:,:,1],gauss_blurr_ker, mode="same")
            #new_im[:,:,2] = convolve2d(new_im[:,:,2],gauss_blurr_ker, mode="same")
            #new_im=GaussianBlur(new_im, (3,3),1,1)
            new_im = cv2.filter2D(src = new_im, ddepth=-1, kernel = gauss_blurr_ker)
            c=c+1
            
        transf_im=Image.fromarray(new_im.clip(0, 255).astype(np.uint8))
        os.makedirs("C:/Users/RO-24/OneDrive/MSc AI/Image and Vision Computing/Project/valid_with_gaussblurr/"+str(times)+"/"+cat, exist_ok=True)
        transf_im.save("C:/Users/RO-24/OneDrive/MSc AI/Image and Vision Computing/Project/valid_with_gaussblurr"+"/"+str(times)+"/"+cat+"/"+name)    
     
   
    
    
    #increases contrast 
    #for increase in  [1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.1, 1.15, 1.20, 1.25]:
    #    new_im= new_im*increase
         #transforms our array into an image and saves it
    #    transf_im=Image.fromarray(new_im.clip(0, 255).astype(np.uint8))
    #    os.makedirs("C:/Users/RO-24/OneDrive/MSc AI/Image and Vision Computing/Project/valid_with_increasing_contrast/"+str(increase)+"/"+cat, exist_ok=True)
    #    transf_im.save("C:/Users/RO-24/OneDrive/MSc AI/Image and Vision Computing/Project/valid_with_increasing_contrast"+"/"+str(increase)+"/"+cat+"/"+name) 
    
    
    
    
    
    #decreases contrast
    #for decrease in [1.0, 0.95, 0.90, 0.85, 0.80, 0.60, 0.40, 0.30, 0.20, 0.10]:
     #   new_im3 = new_im*decrease
      #  #transforms our array into an image and saves it
       # transf_im=Image.fromarray(new_im3.clip(0, 255).astype(np.uint8))
        #os.makedirs("C:/Users/RO-24/OneDrive/MSc AI/Image and Vision Computing/Project/valid_with_decreasing_contrast/"+str(decrease)+"/"+cat, exist_ok=True)
        #transf_im.save("C:/Users/RO-24/OneDrive/MSc AI/Image and Vision Computing/Project/valid_with_decreasing_contrast"+"/"+str(decrease)+"/"+cat+"/"+name) 
    
    
    
    #increasing brightness 
    #for incr in [ 0, 5, 10, 15, 20, 25, 30, 35, 40, 45]:       
     #   new_im=new_im+np.array([incr]*(new_im.shape[0]*new_im.shape[1]*3)).reshape(new_im.shape)#notice that the first two dimensions give the image shape while
        #the 3 stands for the use of the RGB model
        #transforms our array into an image and saves it
      #  transf_im=Image.fromarray(new_im.clip(0, 255).astype(np.uint8))
       # os.makedirs("C:/Users/RO-24/OneDrive/MSc AI/Image and Vision Computing/Project/valid_with_incr_brightness/"+str(incr)+"/"+cat, exist_ok=True)
        #transf_im.save("C:/Users/RO-24/OneDrive/MSc AI/Image and Vision Computing/Project/valid_with_incr_brightness"+"/"+str(incr)+"/"+cat+"/"+name)
    
      
    
    
    
    
    #decreasing brightness
    #for decr in [ 0, 5, 10, 15, 20, 25, 30, 35, 40, 45]:
     #   new_im=new_im-np.array([decr]*(new_im.shape[0]*new_im.shape[1]*3)).reshape(new_im.shape)
        #notice that the first two dimensions give the image shape while
        #the 3 stands for the use of the RGB model
        #transforms our array into an image and saves it
      #  transf_im=Image.fromarray(new_im.clip(0, 255).astype(np.uint8))
       # os.makedirs("C:/Users/RO-24/OneDrive/MSc AI/Image and Vision Computing/Project/valid_with_decr_brightness/"+str(decr)+"/"+cat, exist_ok=True)
        #transf_im.save("C:/Users/RO-24/OneDrive/MSc AI/Image and Vision Computing/Project/valid_with_decr_brightness"+"/"+str(decr)+"/"+cat+"/"+name)



    #occlusion:
    #for edg_length in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]:
    #    newer_im = new_im.copy()
    #    x, y = np.random.choice(list(range(newer_im.shape[0]))), np.random.choice(list(range(newer_im.shape[1])))
  
    #    if x+edg_length<=newer_im.shape[0] and y+edg_length<=newer_im.shape[1]:
    #        h=x+edg_length
    #        w=y+edg_length
    #    if x+edg_length>newer_im.shape[0]: h= newer_im.shape[0]
    #    if y+edg_length>newer_im.shape[1]: w=newer_im.shape[1]    
        
    #    newer_im = cv2.rectangle(newer_im, (x,y), (h, w), (0,0,0), thickness=-1)
    #    transf_im=Image.fromarray(newer_im.clip(0, 255).astype(np.uint8))
    #    os.makedirs("C:/Users/RO-24/OneDrive/MSc AI/Image and Vision Computing/Project/valid_with_occlusion/"+str(edg_length)+"/"+cat, exist_ok=True)
    #    transf_im.save("C:/Users/RO-24/OneDrive/MSc AI/Image and Vision Computing/Project/valid_with_occlusion/"+"/"+str(edg_length)+"/"+cat+"/"+name) 
    


paths_and_cat=get_image_paths_for_each_transform(DATA_PATH, IMAGE_CATEGORIES,50)
j=0

for u in range(len(paths_and_cat[0])):
    path_im=paths_and_cat[0][u]
    cat=paths_and_cat[1][u]
    transformation_applications(path_im, cat, str(j)+".jpg")
    j=j+1
    
#AFTER EXECUTING, IF YOU WANT TO REEXCUTE IT TRY RE_DOING THE DATA
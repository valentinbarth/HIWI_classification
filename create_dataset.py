import os, os.path
from PIL import Image, ImageDraw
import torch
from torchvision import transforms
from torch.utils.data import Dataset

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
#import shutil
import numpy as np
#import glob
import random

# set the dataset dir
datapath = '/home/vbarth/HIWI/classificationDataValentin/DataBase/'


def rgb2gray(rgb):

    rgb = np.array(rgb)
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def check4white(gray, tresh_value = 0.5):

    mask = gray > 220
    if sum(sum(mask)) > tresh_value * (len(mask) * len(mask)):
        all_white = True
    else:
        all_white = False

    return (all_white)

def crop(save_path, input, cls, n_img, height, width):
    im = Image.open(input)
    imgwidth = im.size[0]
    imgheight = im.size[1]
    n_tiles = 0
    k = 1
    for i in range(0,int(imgheight-height/2), int(height-2)):
        for j in range(0,int(imgwidth-width/2),int(width-2)):
            box = (j, i, j+width, i+height)
            a = im.crop(box)

            all_white = check4white(rgb2gray(a))

            if all_white == False:
                a.save(save_path + "/" + "_class" + str(cls) + "_img" + str(n_img) + "_tile" + str(k) + '.jpg')
                k = k + 1
                n_tiles += 1

    return(n_tiles)

# produce dataset of cropped images of the 4 classes (WS ecluded) in one folder
#the class is indicated in the images names, the images are cropped (height,width)
def create_dataset_mixed_cropped(path = datapath, target_dir = '/home/vbarth/HIWI/classificationDataValentin/mixed_cropped', height = 150, width = 150):

    # Create target Directory if doesn't exist
    mixed_cropped = target_dir
    if not os.path.exists(mixed_cropped):
        os.mkdir(mixed_cropped)
        print("Directory " , mixed_cropped ,  " Created ")
    else:    
        print("Directory " , mixed_cropped ,  " already exists")

    classes = {'ADC' : 1,
               'SqCC': 2,
               'Lung': 3,
               'OT'  : 4
               }
    
    #classes = {'ADC' : 1}    # delete later to also use the other classes
    
    n_img = 1
    
    for cls in classes:
        current_path = path + cls
        print(current_path)
        #find out number of instances per class
        #n_files = (len([name for name in os.listdir(current_path) if os.path.isfile(os.path.join(current_path, name))]))
        #for file in range(n_files):
        for root, dirs, files in os.walk(current_path, topdown=False):
            for name in files:
                   
                if os.path.isfile(os.path.join(current_path, name)) == False:
                    continue
                
                n_crop = crop(save_path = mixed_cropped, input = os.path.join(current_path, name), cls = classes[cls], n_img = n_img, height = height, width = width)
                print("cropped tiles: ",n_crop)
                n_img = n_img + 1
                

                
                
"""                 
                 
def load_dataset(datapath):          
    
    classes = {'1' : np.array([1,0,0,0]),
              '2' : np.array([0,1,0,0]),
              '3' : np.array([0,0,1,0]),
              '3' : np.array([0,0,0,1])}
    
    classes = {'1' : np.array([1,0,0,0])}    # delete later to also use the other classes
     
    
    arrs = []
                
    for root, dirs, files in os.walk(datapath, topdown=False):
        for name in files:   
            
            if os.path.isfile(os.path.join(datapath, name)) == False:
                    continue
                
            im = Image.open(os.path.join(datapath, name))    
            im = np.array(im)
            im = rgb2gray(im)/255
            
            #figure out the class
            print(os.path.join(datapath, name), os.path.join(datapath, name)[-5])
            
            img_label_pair = [im_crop, classes[os.path.join(datapath, name)[-5]]] #get class from the name
            arrs.append(img_label_pair)
                    
    print(len(img_label_pair), len(arrs))
    nparrs = np.array(arrs)
    
    all_imgs = nparrs[:,0]
    all_labels = nparrs[:,1]
    print(all_labels)
    
    return all_imgs, all_labels
    
    #all_imgs = torch.tensor(nparrs[:,0])
    #all_labels = torch.tensor(nparrs[:,1])
    #print(tnsr.shape)
            
            




class lung_cancer_dataset(Dataset):
    
    def __init__(self, img_dir="../HIWI/classificationDataValentin/DataBase"):
        self.images, self.labels = load_dataset(datapath = '/home/vbarth/HIWI/classificationDataValentin/mixed_cropped')
       
        
    """    
        
        
    
    
    
create_dataset_mixed_cropped()    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
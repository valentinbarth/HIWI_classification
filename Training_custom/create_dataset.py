import os, os.path
from PIL import Image, ImageDraw
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.transforms import Compose

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
#import shutil
import numpy as np
#import glob
import random

# original dataset dir, there will be created another dir which will actually be used by the load_dataset func
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
def create_dataset_mixed_cropped(path = datapath, target_dir = '/home/vbarth/HIWI/classificationDataValentin/mixed_cropped', height = 150, width = 150, testsplit = 0.2):

    # Create target Directory if doesn't exist
    mixed_cropped = target_dir
    #created root dir:
    if not os.path.exists(mixed_cropped):
        os.mkdir(mixed_cropped)
        print("Directory " , mixed_cropped ,  " Created ")
    else:    
        print("Directory " , mixed_cropped ,  " already exists")
    #training dir: 
    traindir = target_dir + "/train"   
    if not os.path.exists(traindir):
        os.mkdir(traindir)
        print("Directory " , traindir ,  " Created ")
    else:    
        print("Directory " , traindir ,  " already exists")
    #test dir:  
    testdir = target_dir + "/test"  
    if not os.path.exists(testdir):
        os.mkdir(testdir)
        print("Directory " , testdir ,  " Created ")
    else:    
        print("Directory " , testdir ,  " already exists")
    

    classes = {'ADC' : 1,
               'SqCC': 2,
               'Lung': 3,
               'OT'  : 4
               }
    
    #classes = {'ADC' : 1}    # delete later to also use the other classes
    
    
    
    for cls in classes:
        current_path = path + cls
        print(current_path)
        n_img = 1
        #find out number of instances per class
        n_files = (len([name for name in os.listdir(current_path) if os.path.isfile(os.path.join(current_path, name))]))
        for root, dirs, files in os.walk(current_path, topdown=False):
            for name in files:
                   
                if os.path.isfile(os.path.join(current_path, name)) == False:
                    continue
                #create trainset in traindir (from each class <testsplit> % is splitted for testing)
                if n_img <= n_files * (1-testsplit):
                    n_crop = crop(save_path = traindir, input = os.path.join(current_path, name), cls = classes[cls], n_img = n_img, height = height, width = width)
                    if n_img % 100 == 0:
                        print("cropped tiles train: ",n_crop)
                #create testset in testdir
                if n_img > n_files * (1-testsplit):
                    n_crop = crop(save_path = testdir, input = os.path.join(current_path, name), cls = classes[cls], n_img = n_img, height = height, width = width)
                    if n_img % 100 == 0:
                        print("cropped tiles test: ",n_crop)
                
                n_img = n_img + 1
       
                

                
                
                
                 
def load_dataset(datapath):          
    
    classes = np.array([[1,0,0,0],  #np.eye(number of classes)
                        [0,1,0,0],
                        [0,0,1,0],
                        [0,0,0,1]])     
    
    arrs = []
    #i = 0   #remove later   
    for root, dirs, files in os.walk(datapath, topdown=False):
        for name in files:   
            #i = i + 1
            #if i % 1000 != 1:  #just for overview, remove later
                #continue
            if os.path.isfile(os.path.join(datapath, name)) == False:
                continue
            
            im = Image.open(os.path.join(datapath, name))    
            im = np.array(im)
            #print (im.shape)
            #for better runtime: im = rgb2gray(im)  #/255 is done by the ToTensor operation
            #print("image: ", i, "shape: ", im.shape)
            #figure out the class (letter 65 in the path string refers to the class)
            #print(os.path.join(datapath, name), classes[int(name[6])-1])
            img_label_pair = [im, classes[int(name[6])-1]] #get class from the name
            arrs.append(img_label_pair)
                    
    #print(len(img_label_pair), len(arrs))
    nparrs = np.array(arrs)
    print(nparrs.shape)
    all_imgs = nparrs[:,0]
    all_labels = nparrs[:,1]
    #print(all_labels)
    #print (all_imgs.shape, all_imgs.dtype)
    #print (all_labels.shape, all_labels.dtype)
    all_imgs = np.array(all_imgs)
    all_labels = np.array(all_labels)
    print (all_imgs.shape, all_imgs.dtype)
    print (all_labels.shape, all_labels.dtype)
    return all_imgs, all_labels
    
    #all_imgs = torch.tensor(nparrs[:,0])
    #all_labels = torch.tensor(nparrs[:,1])
    #print(tnsr.shape)
            
#load_dataset('/home/vbarth/HIWI/classificationDataValentin/mixed_cropped/test')           




class imagewise_dataset(Dataset):
    
    def __init__(self, datadir, transforms = transforms.ToTensor()):
        self.datadir = datadir
        self.transforms = transforms
        self.composed_trsfm = Compose(transforms)
        self.images, self.labels = load_dataset(datapath = datadir)
        
    
    def __len__(self):
        return (len([name for name in os.listdir(self.datadir) if os.path.isfile(os.path.join(self.datadir, name))]))
    
    def __getitem__(self, idx):
            image = self.images[idx]
            label = self.labels[idx]
            #image, label = self.composed_trsfm((image, label)) #does not work
            image = self.transforms(image)
            label = torch.tensor(label)
            return image, label
        
        
    
    
    
    
       
        
if __name__ == "__main__":
    
    import time     
    

    #Frist create the dataset...already done: /home/vbarth/HIWI/classificationDataValentin/mixed_cropped
    #create_dataset_mixed_cropped()   # about 100000 cropped images depending on crop size
    dataset = imagewise_dataset(datadir = '/home/vbarth/HIWI/classificationDataValentin/mixed_cropped/test')
    
    print (dataset.images.shape, len(dataset))
    #print(dataset.images[:10], dataset.labels[:10])
    t1 = time.perf_counter()
    image, label = dataset[np.random.randint(len(dataset))]
    t2 = time.perf_counter() - t1
    
    print("get_item took {} s".format(t2))
    
    print(image.shape, label.shape, image.dtype, label.dtype)
    
    #plt.imshow(image.transpose((1,2,0)))
    #plt.show
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
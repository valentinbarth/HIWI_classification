import os, os.path
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.transforms import Compose

#import matplotlib.image as mpimg
#import matplotlib.pyplot as plt
#import shutil
import numpy as np
#import glob
#import random
                
                
                
                 
def load_dataset(datapath):          
    
    #only for one hot encoding, but Cross entropy wants just 1D tensor (batchsize,class)
    #classes = np.array([[1,0,0,0],  #np.eye(number of classes)
    #                    [0,1,0,0],
    #                    [0,0,1,0],
    #                    [0,0,0,1]])     
    
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
            img_label_pair = [im, int(name[6])-1] #get class from the name
            arrs.append(img_label_pair)
                    
    #print(len(img_label_pair), len(arrs))
    nparrs = np.array(arrs)
    print("number of samples: ", len(nparrs[:,0]))
    all_imgs = nparrs[:,0]
    all_labels = nparrs[:,1]
    #print(all_labels)
    #print (all_imgs.shape, all_imgs.dtype)
    #print (all_labels.shape, all_labels.dtype)
    all_imgs = np.array(all_imgs)
    all_labels = np.array(all_labels)
    #print (all_imgs.shape, all_imgs.dtype)
    #print (all_labels.shape, all_labels.dtype)
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
    
    #print (dataset.images.shape, len(dataset))
    #print(dataset.images[:10], dataset.labels[:10])
    t1 = time.perf_counter()
    image, label = dataset[np.random.randint(len(dataset))]
    t2 = time.perf_counter() - t1
    
    print("get_item took {} s".format(t2))
    
    print(image.shape, label.shape, image.dtype, label.dtype)
    
    #plt.imshow(image.transpose((1,2,0)))
    #plt.show
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
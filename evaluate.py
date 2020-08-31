import torch
from torch.utils.data import Dataset, DataLoader
#import matplotlib.pyplot as plt
#import seaborn as sns; sns.set()
import numpy as np
import os
from os.path import join
import argparse

import Training_custom.load_dataset

from senet.baseline import resnet20
from senet.se_resnet import se_resnet20

parser = argparse.ArgumentParser()
# e.g. in shell: python evaluate.py --dir Runs/se_net_trained/ --epochs 9
parser.add_argument('-d', '--dir', type=str, metavar='', required=True, help='Directory of the x_folds.')
parser.add_argument('-e', '--epochs', type=int, metavar='', required=True, help='from which epoch should the model be loaded?')

args = parser.parse_args()
working_dir = os.getcwd()

rootpath = join(working_dir, args.dir)

def evaluate(fold_i):
    # path zu einem checkpoint
    CHKPT = f"{args.dir}/fold_{fold_i}/checkpoints/train_chkpt_{args.epochs}.tar"
    
    
    # Das file train_chkpt_100.tar is ein dictionary das ein snapshot vom trainingszustand 
    # der 100. epoche ist.
    # es interessieren eig nur die keys "train_loss", "val_loss" und "model_state_dict".
    # Train und val loss sind 1D torch tensors die den mean loss von der jeweiligen epoche (idx)
    # halten. 
    train_status = torch.load(CHKPT, map_location='cpu')
    #print(train_status)
    
    # model wiederherstellen
    model = resnet20(num_classes=4)  #alternatively: se_resnet20(num_classes=4, reduction=16)
    model.load_state_dict(train_status['model_state_dict'])
    model.eval()
    
    
    
    
    test_data = Training_custom.load_dataset.imagewise_dataset(datadir = '/home/vbarth/HIWI/classificationDataValentin/mixed_cropped/test')
    #dataloader = DataLoader(test_data, batch_size=16,
    #                       shuffle=False, num_workers=0)
    
    
    acc=0   #initialize accuracy
    i = 0   #will count up
    for x, y in test_data:  #iterate over testset
       
        x = x.unsqueeze(0)  #add one dimension (batch missing) to get 4d tensor
        y_pred = model(x).squeeze()
        pred, ind = torch.max(y_pred, 0)
        
           
        if y.item() == ind.item():
            acc = acc + 1   #add one when the prediction was right else add nothing
        
        i = i +1 ##print every 3000th sampel
        if i % 3000 == 0:
            print("Sample: ", i, "\n y_pred: ",y_pred, "\n pred: ", pred, "\n ind: ", ind, "\n y: ", y.item())
    acc = acc/len(test_data)
    print("Accuracy: ", acc ) ##def of accuracy
    with open(rootpath, 'w') as f:
        f.write(f"folder: {fold_i}, accuracy: {acc} \n")
    
    
n_files = (len([name for name in os.listdir(rootpath)]))
#print(n_files)

for fold in range(n_files):
    print(f"Processing folder number {fold}")
    evaluate(fold+1)


#Acc was 0.8882




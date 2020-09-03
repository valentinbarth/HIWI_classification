'''
traincurve.py plotts losses of a training against epochs, call by:
 
python traincurve.py -d ?top_folder/? -i ?name des outputfiles? -e checkpoint_epoche
 
e.g.
python traincurve.py -d "Runs/resnet_trained/" -i "resnet" -e 49
 
train_chkpt_49.tar has to exist, the file is then loaded and used. The plot is safed at: 'evaluation/plots' 
(both dirs are created).
'''

import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
#import seaborn as sns; sns.set()
import numpy as np
import os
from os.path import join
import argparse

#import Training_custom.load_dataset

from senet.baseline import resnet20
from senet.se_resnet import se_resnet20



##############Traincurve##############

# Parsing
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dir', type=str, metavar='', required=True, help='Directory of the x_folds.')
parser.add_argument('-i', '--identifier', type=str, metavar='', required=True, help='Outputfile identifier (folder of model used).')
parser.add_argument('-e', '--epochs', type=int, metavar='', required=True, help='Number of epochs to plot')

args = parser.parse_args()
working_dir = os.getcwd()
root_path = join(working_dir, args.dir)

def plot_traincurve(fold_i):
    
    fold_path = join(args.dir, f"fold_{fold_i}")
    
    save_path = join(working_dir, fold_path,  "evaluation", "plots")
    try:
        os.makedirs(save_path)
    except FileExistsError:
        pass
    
    train_data_path = join(working_dir, fold_path, "checkpoints", "train_chkpt_" + str(args.epochs) + ".tar")
    
    # load in data
    train_data = (torch.load(train_data_path, map_location='cpu')['train_loss'].numpy()[:args.epochs])
    val_data = (torch.load(train_data_path, map_location='cpu')['val_loss'].numpy()[:args.epochs])
    n_epochs = args.epochs
    epoch_arr = np.arange(n_epochs)
    
    # losses
    plt.figure(figsize=(15, 8))
    plt.subplot(121)
    plt.plot(epoch_arr, np.log10(train_data), label="Train Loss")
    plt.plot(epoch_arr, np.log10(val_data), label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel(r"$\log_{10}\left(Cross Entropy\right)$")
    plt.legend()
    plt.subplot(122)
    plt.plot(epoch_arr, train_data, label="Train Loss")
    plt.plot(epoch_arr, val_data, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(join(save_path, "loss_curve_" + args.identifier + ".png"))
    plt.show() 

n_files = (len([name for name in os.listdir(root_path)]))
#print(n_files)

for fold in range(n_files):
    plot_traincurve(fold+1)                              
                                  
                                  
                                  
                                  
                                  

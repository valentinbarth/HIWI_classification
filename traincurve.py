import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
#import seaborn as sns; sns.set()
import numpy as np
import os
from os.path import join
import argparse

import Training_custom.create_dataset

from senet.baseline import resnet20
from senet.se_resnet import se_resnet20

'''
traincurve.py plottet losses von einem training. Aufruf  über:
 
python traincurve.py -d ?top_folder/fold_x/? -i ?name des outputfiles? -e checkpoint_epoche
 
also z.B.
python traincurve.py -d "Runs/se_net_trained/fold_1" -i "testfile" -e 9
 
train_chkpt_9.tar muss es dann geben, das file wird da geladen und benutzt. Das speichert den plot unter
                evaluation/plots
, wobei beide Verzeichnisse erstellt werden.
'''

##############Traincurve##############

# Parsing
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dir', type=str, metavar='', required=True, help='Directory of stored checkpoints.')
parser.add_argument('-i', '--identifier', type=str, metavar='', required=True, help='Outputfile identifier (folder of model used).')
parser.add_argument('-e', '--epochs', type=int, metavar='', required=True, help='Number of epochs to plot')

args = parser.parse_args()
working_dir = os.getcwd()
save_path = join(working_dir, args.dir, "evaluation", "plots")
try:
    os.makedirs(save_path)
except FileExistsError:
    pass

train_data_path = join(working_dir, args.dir, "checkpoints", "train_chkpt_" + str(args.epochs) + ".tar")

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



import torch
import matplotlib.pyplot as plt
#import seaborn as sns; sns.set()
import numpy as np
import os
from os.path import join
import argparse


# path zu einem checkpoint
CHKPT = "top_folder/fold_1/checkpoints/train_chkpt_100.tar"

# Das file train_chkpt_100.tar is ein dictionary das ein snapshot vom trainingszustand 
# der 100. epoche ist.
# Dich interessieren eig nur die keys "train_loss", "val_loss" und "model_state_dict".
# Train und val loss sind 1D torch tensors die den mean loss von der jeweiligen epoche (idx)
# halten. 
train_status = torch.load(CHKPT, map_location='cpu')
print(train_status)

# model wiederherstellen
model = ...
model.load_state_dict(train_status['model_state_dict'])
model.eval()

# jetzt kannst du das model benutzen wie du willst, z.b. übers test
# set loopen und predictions raushauen
test_data = ...

accs = []
for x, y in test_data:
    y_pred = model(x)
    
    acc = calc_accuracy(y, y_pred)

    accs.append(acc)

print(np.mean(acc))




##############Traincurve##############

# Parsing
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dir', type=str, metavar='', required=True, help='Directory of stored checkpoints.')
parser.add_argument('-i', '--identifier', type=str, metavar='', required=True, help='Outputfile identifier (folder of model used).')
parser.add_argument('-e', '--epochs', type=int, metavar='', required=True, help='Number of epochs to plot')

args = parser.parse_args()
working_dir = os.getcwd()
save_path = join(working_dir, "evaluation", "plots")
try:
    os.mkdir(save_path)
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
plt.ylabel(r"$\log_{10}\left(MSE\right)$")
plt.legend()
plt.subplot(122)
plt.plot(epoch_arr, train_data, label="Train Loss")
plt.plot(epoch_arr, val_data, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend()
plt.tight_layout()
plt.savefig(join(save_path, "loss_curve_" + args.identifier + ".png"))
plt.show()

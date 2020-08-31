import os
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
#from dataloading.dataset import MergedPatientDataset

from Training_custom.trainer import Trainer
from Training_custom.cvtrainer import CVTrainer
import Training_custom.load_dataset

from senet.baseline import resnet20
from senet.se_resnet import se_resnet20

import logging


# set device for training
cuda_enabled_gpu = torch.cuda.is_available()
device = "cuda" if cuda_enabled_gpu else "cpu"
# if using gpu, activate cudnn benchmark mode
torch.backends.cudnn.benchmark = cuda_enabled_gpu

# Global training parameter kwargs
train_config = {
    'device': device,
    'epochs': 50,   #change to ~200 
    'batches_per_epoch': 1400,
    'batch_size': 64,
    'num_workers': 1,
    'output_folder': 'Runs/resnet_trained',
    'validation_split': 0.25,
    'validation_indices': [],
    'prefetch_validation': False,
    'amp': False,
    'log_level': logging.INFO
}

if __name__ == "__main__":

    # create pytorch dataset
    dataset_imgwise = Training_custom.load_dataset.imagewise_dataset(datadir = '/home/vbarth/HIWI/classificationDataValentin/mixed_cropped/train')
    
    
    # set model, optimizer and loss criterion
    model = resnet20(num_classes=4)  # choose se_resnet20(num_classes=4, reduction=16) or resnet20(num_classes=4) for example

    # optimizer
    #optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    optimizer = optim.SGD(model.parameters(),lr=1e-1, momentum=0.9, weight_decay=1e-4)
    # lr scheduler
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=500, factor=0.2, min_lr=1e-6)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 80, 0.1)
    
    # loss
    criterion = F.cross_entropy
    
    # initialize trainer instance
    trainer = CVTrainer(
        folds=4, model=model, optimizer=optimizer, criterion=criterion,
        dataset=dataset_imgwise, scheduler=scheduler,
        train_config=train_config
    )

    # trainer = Trainer.from_checkpoint(
    #     model=model, optimizer=optimizer, criterion=criterion,
    #     dataset=dataset_tr, scheduler=scheduler,
    #    path='test_run'
    # )

    trainer.train()

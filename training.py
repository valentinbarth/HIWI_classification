import os
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn

#from dataloading.dataset import MergedPatientDataset
from training.cvtrainer import CVTrainer
from training.trainer import Trainer

import logging

#from models.unet import UNet3D, UNetCNN
from utils.losses import KLDivMSE

# set device for training
cuda_enabled_gpu = torch.cuda.is_available()
device = "cuda" if cuda_enabled_gpu else "cpu"
# if using gpu, activate cudnn benchmark mode
torch.backends.cudnn.benchmark = cuda_enabled_gpu

# Global training parameter kwargs
train_config = {
    'device': device,
    'epochs': 100,
    'batches_per_epoch': 3,
    'batch_size': 30,
    'num_workers': 2,
    'output_folder': 'test_run',
    'validation_split': 0.1,
    'validation_indices': [],
    'prefetch_validation': False,
    'amp': False,
    'log_level': logging.INFO
}

if __name__ == "__main__":

    # create pytorch dataset
    dataset_imgwise = ImagewiseDataset(
        "/home/vbarth/HIWI/classificationDataValentin/", to_RAM=True
    )

    dataset_tr.set_normalization()

    #dataset_tr.set_normalization(
    #    {
    #        'HU_lower_bound': -1000,
    #        'HU_upper_bound': 2100,
    #        'dose_norm': 8.25381e-5
    #    }
    #)

    # set model, optimizer and loss criterion
    #model = UNet3D(fmaps_per_level=[16, 32, 64, 128])
    model = se_resnet20(num_classes=5, reduction=args.reduction)

    # optimizer
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)

    # lr scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=500, factor=0.2, min_lr=1e-6
    )

    # loss
    criterion = KLDivMSE(alpha=1., reduction='mean')

    # initialize trainer instance
    trainer = CVTrainer(
        folds=10, model=model, optimizer=optimizer, criterion=criterion,
        dataset=dataset_tr, scheduler=scheduler,
        train_config=train_config
    )

    # trainer = Trainer.from_checkpoint(
    #     model=model, optimizer=optimizer, criterion=criterion,
    #     dataset=dataset_tr, scheduler=scheduler,
    #    path='test_run'
    # )

    trainer.train()

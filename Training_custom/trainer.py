import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torch.cuda import amp
import numpy as np
import logging
import time

import os
from os import mkdir, listdir, getcwd
from os.path import join, exists

from copy import deepcopy


class EarlyStopping(object):
    '''
    Performs early stopping if the validation loss has not
    improved for a specified amount of epochs.
    '''

    def __init__(self, patience, logger=None):
        self.patience = patience
        self.n_bad_epochs = 0
        self.running_best = np.inf
        self.logger = logger

    def __call__(self, loss):
        return self.step(loss)

    def step(self, loss):
        '''
        This function performs a step by checking if the new evaluated
        loss is lower than the running best stored in the instance.

        Args:
            loss (float): new criterion value
        Returns:
            (bool) if early stopping is triggered
        '''

        # update number of bad epochs
        if self.running_best > loss:
            self.running_best = loss
            self.n_bad_epochs = 0
        else:
            self.n_bad_epochs += 1

        # check if early stopping criterion is fulfilled
        if self.n_bad_epochs > self.patience:
            if self.logger:
                self.logger.info(
                    'Early Stopping: Criterion has'
                    f'not improved for {self.n_bad_epochs}.'
                )
            self._reset()
            return True
        else:
            self.logger.info(
                'Current Patience Level: '
                f'{self.n_bad_epochs}/{self.patience}'
            )
            return False

    def _reset(self):
        '''
        Reset the running best criterion and the number of bad epochs.
        '''
        self.n_bad_epochs = 0
        self.running_best = np.inf


class Trainer(object):

    __DEFAULT_CONFIG = {
        'device': 'cpu',  # 'cpu'/'cuda'/'cuda:0'/'cuda:1' ...
        'epochs': 100,
        'batches_per_epoch': 100,
        'batch_size': 1,
        'num_workers': 0,   # dataloading workers
        'output_folder': 'train_out',
        'validation_split': 0.2,
        'validation_indices': [],
        'prefetch_validation': False,   # push validation data to GPU
        'early_stopping_patience': 0,
        'amp': False,  # Automatic mixed precision
        'log_level': logging.INFO
    }

    def __init__(self, model, optimizer, criterion, dataset,
                 scheduler=None, train_config=None):

        # training config
        self._config = self._check_config_dict(train_config)

        # prepare output folder structure
        self._create_folder_structure()

        # set essentials
        self._model = model.to(device=self._config['device'])
        self._criterion = criterion #.to(device=self._config['device'])
        self._optimizer = optimizer
        self._dataset = dataset
        self._scheduler = scheduler
        self._early_stopper = None
        self._scaler = None

        # init automated mixed precision (amp) training
        if self._config['amp']:
            self._scaler = amp.GradScaler()

        # init logger
        self._logger = self._init_logger()

        # init current training status
        self._train_status = self._init_train_status()

        # validation and training subset init
        self._train_subset, self._val_subset = self._init_data_subsets()
        self._val_gpu_prefetch = None

        # initialize early stopping
        patience = self._config['early_stopping_patience']
        if (patience > 0) and self._val_subset:
            self._early_stopper = EarlyStopping(patience, logger=self._logger)

        # utilities
        self._save_step = self._config['epochs'] // 10

        # save training config
        self._save_config()

        self._log_init_info()

    def train(self):
        '''
        Implements the main training loop given the configuration of the
        training dictionary (_config). The training loop trains the model
        for the specified amount of epochs. In every iteration, the model
        is trained on the specified amount of batches and then validated
        on the validation set. The mean of both the training and validation
        loss is reported every iteration.
        '''

        # some variables
        device = self._config['device']
        non_blocking = True if device != 'cpu' else False
        batch_size = self._config['batch_size']
        batches_per_epoch = self._config['batches_per_epoch']

        # log device name
        if 'cuda' in device or 'gpu' in device:
            self._logger.info('Using GPU '
                              f'"{torch.cuda.get_device_name(device=device)}"'
                              ' for training.')
            # maybe prefetch validation data
            if self._config['prefetch_validation']:
                self._prefetch_val_data()
        else:
            self._logger.info('Training on CPU.')

        # get starting and ending epoch
        st_epoch = self._train_status['epoch']
        max_epoch = self._config['epochs']
        if st_epoch > 0:
            self._logger.info(f'Resuming training from epoch {st_epoch} on.')
            st_epoch += 1
        else:
            self._logger.info('Starting training.')
            # print AMP info
            if self._config['amp'] and self._scaler._enabled:
                self._logger.info('Automatic Mixed Precision is ENABLED')
            else:
                self._logger.info('Automatic Mixed Precision is DISABLED')

        # TRAINING LOOP
        for e in range(st_epoch, max_epoch):
            # current epoch
            self._train_status['epoch'] = e
            self._logger.info(f'Starting epoch {e}')

            # measure time
            t_start = time.perf_counter()

            # run training iteration
            # differentiate between amp and normal training
            if self._config['amp']:
                self._train_iter_amp(device, non_blocking, batch_size,
                                     batches_per_epoch)
            else:
                self._train_iter(device, non_blocking, batch_size,
                                 batches_per_epoch)

            # maybe validate
            if self._val_subset:
                self._validate(device, non_blocking)

            elapsed_time = time.perf_counter() - t_start

            # print info
            self._log_progress(elapsed_time)

            # maybe adjust LR
            if self._scheduler:
                self._adjust_lr()

            # check early stopping
            if self._config['early_stopping_patience'] > 0:
                val_loss = self._train_status['val_loss'][e].item()
                if self._early_stopper(val_loss):
                    self.save_checkpoint()
                    break

            # maybe save checkpoint
            if (e % self._save_step == 0) or (e == self._config["epochs"]-1):
                self.save_checkpoint()

    def set_validation_subset(self, subset):
        '''
        Sets the validation subset.

        Args:
            subset (sequence): `torch.utils.data.Subset` or indices
                               corresponding to the used dataset.
        '''

        if isinstance(subset, Subset) and subset.dataset == self._dataset:
            self._val_subset = subset
        elif hasattr(subset, '__iter__'):
            self._val_subset = Subset(self._dataset, subset)
        else:
            raise TypeError('Provide indices (sequence) or a'
                            'torch.utils.data.Subset instance!')

    def save_checkpoint(self):
        '''
        Saves the current training status including model and optimizer
        parameters to storage.
        '''

        self._logger.info('Saving checkpoint ...')

        if self._train_status['amp_state_dict']:
            self._train_status['amp_state_dict'] = self._scaler.state_dict()

        # save current training status in checkpoints folder
        path = join(self._config['output_folder'], 'checkpoints')
        filename = 'train_chkpt_' + str(self._train_status['epoch']) + '.tar'
        torch.save(self._train_status, join(path, filename))

        # save training config
        self._save_config()

    def _train_iter(self, device, non_blocking, b, bpe):
        '''
        Runs one iteration of the training loop.
        '''

        self._logger.debug('Starting training iteration.')

        iter_subset = self._sample_random_subset(self._train_subset, b * bpe)
        train_loader = self._get_dataloader(iter_subset)

        curr_epoch = self._train_status['epoch']
        losses = torch.zeros(size=(bpe,))

        # model to training mode
        self._model.train()

        # loop over batches
        for idx, (inputs, targets) in enumerate(train_loader):
            # zero gradients
            self._optimizer.zero_grad()

            # progress
            if (idx % np.ceil(bpe / 10) == 0):
                self._logger.debug(f"Processing training batch {idx}/{bpe}")

            # push data to device
            inputs = inputs.to(device=device, non_blocking=non_blocking)
            targets = targets.to(device=device, non_blocking=non_blocking)

            # forward pass
            predictions = self._model(inputs)
            del inputs

            # loss
            loss = self._criterion(predictions, targets)
            del targets

            # backprop
            loss.backward()
            self._optimizer.step()

            # keep track of loss
            losses[idx] = loss.item()

        self._train_status['train_loss'][curr_epoch] = losses.mean()

    def _train_iter_amp(self, device, non_blocking, b, bpe):
        '''
        Runs one iteration of the training loop using automatic mixed precision
        forward and backward passes.
        '''

        self._logger.debug('Starting training iteration (AMP).')

        iter_subset = self._sample_random_subset(self._train_subset, b * bpe)
        train_loader = self._get_dataloader(iter_subset)

        curr_epoch = self._train_status['epoch']
        losses = torch.zeros(size=(bpe,))

        # model to training mode
        self._model.train()

        # loop over batches
        for idx, (inputs, targets) in enumerate(train_loader):
            # zero gradients
            self._optimizer.zero_grad()

            # progress
            if (idx % np.ceil(bpe / 10) == 0):
                self._logger.debug(f"Processing training batch {idx}/{bpe}")

            # push data to device
            inputs = inputs.to(device=device, non_blocking=non_blocking)
            targets = targets.to(device=device, non_blocking=non_blocking)

            # autocast when forward passing
            with amp.autocast():
                predictions = self._model(inputs)
                del inputs

                # loss
                loss = self._criterion(predictions, targets)
                del targets

            # scale loss and backprop
            self._scaler.scale(loss).backward()
            self._scaler.step(self._optimizer)
            self._scaler.update()

            # keep track of loss
            losses[idx] = loss.item()

        self._train_status['train_loss'][curr_epoch] = losses.mean()

    def _validate(self, device, non_blocking):
        '''
        Perform validation on the validation set.
        '''

        self._logger.info('Validating ...')

        # epoch
        curr_epoch = self._train_status['epoch']

        # eval mode
        self._model.eval()

        # use prefetched data if available, else use dataloader
        if self._val_gpu_prefetch:
            bpe = len(self._val_gpu_prefetch)
            losses = torch.zeros(size=(bpe,))
            # loop over validation batches
            with torch.no_grad():
                for idx, (inputs, targets) in enumerate(self._val_gpu_prefetch):
                    # progress
                    if (idx % np.ceil(bpe / 10) == 0):
                        self._logger.debug(f"Processing validation batch {idx}/{bpe}")

                    if self._config['amp']:
                        with amp.autocast():
                            predictions = self._model(inputs)
                            loss = self._criterion(predictions, targets)
                    else:
                        predictions = self._model(inputs)
                        loss = self._criterion(predictions, targets)

                    # save val loss
                    losses[idx] = loss.item()
        else:
            loader = self._get_dataloader(self._val_subset)
            bpe = len(loader)
            losses = torch.zeros(size=(bpe,))
            with torch.no_grad():
                for idx, (inputs, targets) in enumerate(loader):
                    if (idx % np.ceil(bpe / 10) == 0):
                        self._logger.debug(f"Processing validation batch {idx}/{bpe}")
                    inputs = inputs.to(device=device, non_blocking=non_blocking)
                    targets = targets.to(device=device, non_blocking=non_blocking)

                    if self._config['amp']:
                        with amp.autocast():
                            predictions = self._model(inputs)
                            del inputs
                            loss = self._criterion(predictions, targets)
                            del targets
                    else:
                        predictions = self._model(inputs)
                        del inputs
                        loss = self._criterion(predictions, targets)
                        del targets

                    losses[idx] = loss.item()

        self._train_status['val_loss'][curr_epoch] = losses.mean()

    def _log_progress(self, time):
        e = self._train_status['epoch']
        self._logger.info(f"Epoch {e} finished --> Elapsed Time: {time}s")
        self._logger.info(f"Avg. train loss: {self._train_status['train_loss'][e].item()}")
        self._logger.info(f"Avg. validation loss: {self._train_status['val_loss'][e].item()}")

    def _init_train_status(self):
        '''
        Initializes training status.
        Training status is a dict storing info about the current training
        progress like current epoch, losses etc.
        '''
        train_status = {
            'epoch': 0,
            'model_state_dict': self._model.state_dict(),  # reference
            'optimizer_state_dict': self._optimizer.state_dict(),  # reference
            'scheduler_state_dict': self._scheduler.state_dict() if self._scheduler else None, # reference
            'amp_state_dict': self._scaler.state_dict() if self._config['amp'] else None,
            'train_loss': torch.zeros(size=(self._config['epochs'],)),
            'val_loss': torch.zeros(size=(self._config['epochs'],))
        }

        return train_status

    def _init_logger(self):
        # date
        from datetime import datetime
        now = datetime.now()
        date_str = now.strftime("%d_%m_%Y_%H.%M.%S")

        # init logger
        logger = logging.getLogger(f"{self.__class__.__name__}")
        logger.setLevel(self._config['log_level'])

        # log to console and file
        fh = logging.FileHandler(join(getcwd(), self._config['output_folder'], 'logs', f"training_{date_str}.log"))
        ch = logging.StreamHandler()

        # formatter
        formatter = logging.Formatter('%(asctime)s | %(name)s-%(levelname)s: %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # add handler to logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        return logger

    def _init_data_subsets(self):
        '''
        Initializes both the training and validation subset. Validation subset
        is based on training config parameters 'validation indices' and
        'validation_split'. Training set are all the data that is not the
        validation set.
        '''

        val_split = self._config['validation_split']
        val_indices = self._config['validation_indices']
        n_data = len(self._dataset)

        # check if proportion is specified
        if val_split == .0 and not val_indices:
            return self._dataset, None

        if val_indices:
            # given validation indices
            val_set = Subset(self._dataset, val_indices)

            # train set
            train_indices = [
                x for x in range(n_data) if x not in val_indices
            ]
            train_set = Subset(self._dataset, train_indices)
            return train_set, val_set
        else:
            # randomly split available data into validation and training set
            N = int(val_split * n_data)
            val_subset = self._sample_random_subset(self._dataset, N)
            sampled_val_indices = val_subset.indices
            self._config['validation_indices'] = sampled_val_indices

            # train set
            train_indices = [
                x for x in range(n_data) if x not in sampled_val_indices
            ]

            train_set = Subset(self._dataset, train_indices)

            return train_set, val_subset

    def _prefetch_val_data(self):
        '''
        Prefetch validation data: push validation tensors directly to the GPU or into pinned memory.
        '''
        self._logger.info('Prefetching validation data ...')
        self._logger.debug('Trying to push the validation set to the GPU ...')

        self._val_gpu_prefetch = []
        for idx, (inputs, targets) in enumerate(self._get_dataloader(self._val_subset)):
            inputs = inputs.to(device=self._config['device'])
            targets = targets.to(device=self._config['device'])

            self._val_gpu_prefetch.append((inputs, targets))

        self._logger.debug('Pushed validation data to GPU.')

    def _get_dataloader(self, data):
        return DataLoader(
            dataset = data,
            batch_size = self._config['batch_size'],
            shuffle = False,
            num_workers = self._config['num_workers'],
            pin_memory = False if self._config['device'] == 'cpu' else True,
            drop_last = False
        )

    def _create_folder_structure(self):
        '''
        Creates necessary folder structures.

        'logs' folder will hold training log files.
        'checkpoints' holds training checkpoints as dictionaries in .tar files.
        '''
        # dirs
        work_dir = getcwd()
        output_folder = self._config['output_folder']

        # top folder
        try:
            mkdir(join(work_dir, output_folder))
        except FileExistsError:
            print(f'Output folder {join(work_dir, output_folder)} already exists, checkpoints might get overwritten!')

        # subfolders
        try:
            os.mkdir(join(work_dir, output_folder, 'logs'))
            os.mkdir(join(work_dir, output_folder, 'checkpoints'))
        except FileExistsError:
            pass

    def _log_init_info(self):
        self._logger.info('Successfully initialized.')
        self._logger.info(f'Model: {repr(self._model)}')
        self._logger.info(f'Optimizer: {repr(self._optimizer)}')
        self._logger.info(f'Criterion: {repr(self._criterion)}')
        if (hasattr(self._dataset, '_normalize') and
            hasattr(self._dataset, '_norm')):
            if self._dataset._normalize:
                self._logger.info(
                    f"Normalization: {repr(self._dataset._norm)}"
                )

        self._logger.info(f'Number of total data: {len(self._dataset)}')

        self._logger.info('##### Training Configuration #####')
        for key, val in self._config.items():
            if key == 'validation_indices' and len(self._config['validation_indices']) > 25:
                self._logger.info(
                    f'{key} --->  [{val[0]}, {val[1]}, {val[2]} ... {val[-3]}, {val[-2]}, {val[-1]}]'
                )
            elif key == 'log_level':
                self._logger.info(f'{key} ---> {logging.getLevelName(val)}')
            else:
                self._logger.info(f'{key} ---> {val}')
        self._logger.info('##### Training Configuration #####')

    def _save_config(self):
        torch.save(self._config, join(self._config['output_folder'], 'train_config.pt'))

    def _adjust_lr(self):
        '''
        Performs learning rate adjustment according to the assigned
        scheduler '_scheduler', i.e. calls the scheduler's step
        function, which takes a loss value.

        If training is performed with validation, the validation loss will be
        the measure for adjustment, else training loss is used. The scheduler
        itself is configured by the user before initializing the Trainer
        class instance.
        '''
        epoch = self._train_status['epoch']

        # current lr
        before_lr = self._optimizer.param_groups[0]['lr']

        # use val loss if validation is performed
        if self._val_subset:
            val_loss = self._train_status['val_loss'][epoch].item()
            self._scheduler.step(val_loss)
        else:
            tr_loss = self._train_status['train_loss'][epoch].item()
            self._scheduler.step(tr_loss)

        # lr after step
        after_lr = self._optimizer.param_groups[0]['lr']

        # log lr update
        if after_lr < before_lr:
            self._logger.info(f'{self._scheduler.__class__.__name__} reduced the learning rate from {before_lr} to {after_lr}')

    @staticmethod
    def _sample_random_subset(data, N):
        '''
        Sample a random subset of the dataset for one iteration of the
        training loop.
        '''

        # sample random subset w/o replacement
        shuffled_indices = torch.randperm(len(data))
        rand_subset = shuffled_indices[:N]

        return Subset(data, rand_subset.tolist())

    @staticmethod
    def _load_config(path):
        '''
        Load the configuration dict from storage.

        Args:
            path (str): path to the .pt file.
        '''

        if exists(path):
            train_config = torch.load(path)
            if not isinstance(train_config, dict):
                raise TypeError('The config file is not a python dictionary!')
            return train_config
        else:
            raise FileNotFoundError('No training config was found!')

    @classmethod
    def _check_config_dict(cls, config_dict):
        '''
        Checks validity of the training configuration dictionary and fills
        missing entries with default values.

        Args:
            config_dict (dict): Configuration dictionary.

        Returns:
            (dict): Checked and maybe modified training config dictionary.
        '''

        if config_dict is None:
            return deepcopy(cls.__DEFAULT_CONFIG)

        if isinstance(config_dict, dict):

            dict_copy = deepcopy(config_dict)
            for default_key, default_val in cls.__DEFAULT_CONFIG.items():
                # fill with default value if empty and check for type
                dict_copy.setdefault(default_key, default_val)
                if type(dict_copy[default_key]) != type(default_val):
                    raise TypeError(f'{default_key} needs to be of {type(default_val)} but is {type(dict_copy[default_key])}!')
            return dict_copy
        else:
            raise TypeError(f"Config needs to be of {type(cls.__DEFAULT_CONFIG)} but is {type(config_dict)}!")

    @classmethod
    def from_checkpoint(cls, model, optimizer, criterion, dataset, path,
                        scheduler=None, epoch=None):
        '''
        Resume training from checkpoint.

        Args:

            model, optimizer, criterion, dataset: essentials used in previous
                                                  trainings.

            path (str): path to top level folder of the training output folder.
            epoch (int, optional): epoch to resume from, appropriate
                                   checkpoint has to exist. Defaults to the
                                   most recent checkpoint.
        '''

        if exists(path):
            # load config dict
            config_dict = cls._load_config(join(path, 'train_config.pt'))

            # load train status
            chkpts = listdir(join(path, 'checkpoints'))
            if not chkpts:
                raise FileNotFoundError("No checkpoints were found!")

            if epoch:
                path_to_chkpt = join(
                    path, 'checkpoints', 'train_chkpt_' + str(epoch) + '.tar'
                )
            else:
                # infer most recent chkpt
                max_ep = 0
                for chkpt in chkpts:
                    epoch = int(chkpt[12:-4])
                    if epoch > max_ep:
                        max_ep = epoch

                path_to_chkpt = join(
                    path, 'checkpoints', 'train_chkpt_' + str(max_ep) + '.tar'
                )

            # checkpoint
            train_status = torch.load(path_to_chkpt)

            # init trainer
            trainer = cls(model, optimizer, criterion, dataset,
                          scheduler, config_dict)
            trainer._train_status = train_status

            # restore model and optimizer checkpoints
            trainer._model.load_state_dict(
                trainer._train_status['model_state_dict']
            )
            trainer._optimizer.load_state_dict(
                trainer._train_status['optimizer_state_dict']
            )

            # restore lr scheduler state if one exists
            if trainer._scheduler:
                trainer._scheduler.load_state_dict(
                    trainer._train_status['scheduler_state_dict']
                )

            # amp scaler state dict
            if trainer._config['amp']:
                trainer._scaler.load_state_dict(
                    trainer._train_status['amp_state_dict']
                )

            return trainer

        else:
            raise FileNotFoundError("The specified path does not exist!")

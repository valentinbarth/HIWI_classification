import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torch.cuda import amp

import numpy as np

from training.trainer import Trainer

from os import getcwd, mkdir
from os.path import join

from copy import deepcopy


class CVTrainer(Trainer):
    def __init__(self, folds, *args, **kwargs):

        # additional member
        self._nfolds = folds

        # holds all training subsets
        self._cvsubsets = dict()

        # call base class constructor
        super(CVTrainer, self).__init__(*args, **kwargs)

        # add finish flag for current fold
        self._config["finished_fold"] = False

        # keep a deepcopy of initial training status
        # for fold reinitialization
        self._start_status = deepcopy(self._train_status)

    def _create_folder_structure(self):
        '''
        Overwrites base class method.

        Additionally creates subfolders for all cross validation folds.
        Each fold-folder will hold logs and checkpoints.

        '''
        # dirs
        work_dir = getcwd()
        output_folder = self._config['output_folder']

        # top folder
        try:
            mkdir(join(work_dir, output_folder))
        except FileExistsError:
            print(f'Output folder {join(work_dir, output_folder)} already exists, checkpoints might get overwritten!')

        # create subfolders for each cv fold
        for i in range(1, self._nfolds+1):
            # fold
            try:
                mkdir(join(work_dir, output_folder, f'fold_{i}'))
            except FileExistsError:
                pass

            # subfolders
            try:
                mkdir(join(work_dir, output_folder, f'fold_{i}', 'logs'))
                mkdir(join(work_dir, output_folder, f'fold_{i}', 'checkpoints'))
            except FileExistsError:
                pass

        # set output folder to first fold
        self._config['output_folder'] = join(self._config['output_folder'], 'fold_1')

    def _init_data_subsets(self):
        '''
        Overwrites base class method.

        Initializes both the training and validation subset for each cross-
        validation fold. Amount of training and validation data is
        automatically determined by the number of folds and total data.
        All fold combinations get stored in the member _cvsubsets.

        Access via:
            _cvsubsets["fold_i"]:
                (Subset(data, train_indices), Subset(data, val_indices))
        '''

        # total available indices
        indices = np.random.permutation(len(self._dataset))

        # split into _nfolds partitions
        partitions = np.array_split(indices, self._nfolds)

        for ifold in range(self._nfolds):
            tr_idx = np.concatenate([partition for i, partition
                                    in enumerate(partitions)
                                    if i != ifold])

            val_idx = partitions[ifold]

            # get dataset subsets for this fold
            tr_sub = Subset(self._dataset, tr_idx)
            val_sub = Subset(self._dataset, val_idx)

            # add them to the fold dict
            self._cvsubsets["fold_" + str(ifold + 1)] = (tr_sub, val_sub)

        # save val indices of first fold
        first_val_set = self._cvsubsets["fold_1"][1]
        self._config['validation_indices'] = first_val_set.indices

        return self._cvsubsets["fold_1"]

    def train(self):
        '''
        Overwrites base class method.

        Perform training on ALL folds. Starts with the
        first fold and applies base class training method.
        If the fold has finished training, the current fold
        is changed and training for this fold is initialized.
        '''

        for ifold in range(1, self._nfolds + 1):
            # takes time! Full training on one fold
            super(CVTrainer, self).train()

            # set finished flag
            self._config["finished_fold"] = True
            self._logger.info(f"Finished training on Fold {ifold}.")

            # switch to next fold
            if ifold != self._nfolds:
                self._prepare_next_fold(ifold + 1)

    def _prepare_next_fold(self, fold):
        '''
        Prepares the next fold for training.
        I.e. initialize the _config dict for the next fold
        together with the logger.
        '''

        # config infos
        top_folder = self._config['output_folder'].split('/')[0]

        self._config['output_folder'] = join(top_folder, f'fold_{fold}')
        self._config['finished_fold'] = False

        # train and validation data
        self._train_subset, self._val_subset = self._cvsubsets[f'fold_{fold}']

        # save val indices for info/chkpt resuming
        self._config['validation_indices'] = self._val_subset.indices

        # reinit model and optimizer
        self._model.load_state_dict(
            self._start_status['model_state_dict']
        )
        self._optimizer.load_state_dict(
            self._start_status['optimizer_state_dict']
        )

        # reinitialize lr scheduler and amp scaler
        if self._scheduler:
            self._scheduler.load_state_dict(
                self._start_status['scheduler_state_dict']
            )
        if self._config['amp']:
            self._scaler.load_state_dict(
                self._start_status['amp_state_dict']
            )

        self._train_status = self._init_train_status()

        # save config dict
        self._save_config()

        # update logger
        self._logger.handlers.clear()
        self._logger = self._init_logger()
        self._log_init_info()

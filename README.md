# SENet in pytorch for medical image analysis

An implementation of SENet, proposed in **Squeeze-and-Excitation Networks** by Jie Hu, Li Shen and Gang Sun, who are the winners of ILSVRC 2017 classification competition. The baseline (model implementation in pytorch) is taken as it is (for the most part) from [moskomule](https://github.com/moskomule/senet.pytorch/tree/58844943617b5215f2d3eab149735ac4a66ed393). The aim of this poject is to assess the SE-module on medical images by comparing it with the plain resnet model.


## Pre-requirements

The codebase is tested on the following setting.

* Python>=3.6.9
* PyTorch>=1.6.0
* torchvision>=0.7

You can use the files provided in the folder `Docker` to accomplish these settings.
1. open terminal and navigate to the `Docker` dir
2. to configure the rootless docker type `. config_rootless_docker.sh` in the terminal (change user and UID in the file before)
3. build the docker image by executing `bash build_pytorch_image.sh`
4. start a container with `bash run_pytorch.sh` (change the mounted volume and the workdir in the file before)
5. you can check the torch verion, the torchvision verion and the availability of the GPU by running `python check_pytorch.py` and the python verion by just        typing `python --version`

## Dataset and Preprocessing

The dataset is supplied by Marlen (Dr. Weiss?). The preprocessing of the data is done in the module `rearrange_dataset`. The images are cropped (further data augmentation is not performed but might be added) checked for white space (and in case of more than 50% white discarded - too much background). There is no mean normalization/zero centering, this might be added  if needed. The dataset is split into a test set (20%) and a train set(80%). The class will be denoted in the file's name (and not in the folder anymore).

Before running `python rearrange_dataset.py` you should give the right datapaths in the beginning of the file:
* `datapath` should be the path to the original data supplied by Marlen
* `new_datapath` should be the path to the new created dataset

Some hyperparameters might be adjusted if wished (but defaults are given):
- the number of images split for testing (when calling the function in the end)
- the size of the cropped images: default is 150x150, I would not go smaller because then there is not enough structure in one sample (also when calling)
- the threshold for discarding white images might be reduced to create more data (in the `crop` function)

## Training

Different Resnet and SE-ResNet architectures (18, 34, 50, 101, 152/20, 32) are implemented and can be choosen(see `senet/baseline.py` and `senet/se_resnet.py`).
The code was tested on the smallest model resnet 18 (in the code called se_resnet20 or resnet20) for runtime reasons. 

Call in terminal `python training.py` to train the model. 
Basically this script initializes the trainer with the **specified model**(in line 55), optimizer, scheduler and criterion.
These default hyperparameters are oriented on the paper (or respectively the pytorch implementation of it). 

In the 'train_config' dictionary you might set the number of epochs, batch size, the **outputfolder** (relative path)
and the split for cross validation (here you should also choose the appropriate number of folds when inizializen the trainer in line 68).
Also take care of **giving the right path to your data**(in line 51) when inizializing the dataset.

The final version of the training was performed with the following settings:
* number of epochs: 150
* batches per epoch: 700
* batchsize: 64
* optimizer: SGD with initial learning rate of 0.0001 and a reduction of 0.1 every 40 eepochs
* loss: cross entropy

The evaluation is performed on a 4-fold cross validation (each time 25% of the trainset is splitted for validation)
One trainig on the medphys410 took about 24h with those settings.

The training produces folders with checkpoints to load the trained model from and a log file, both within the folder of the experiment. 

## Evaluation

There are two scripts for evaluation: `traincurve.py` and `evaluate.py`:

**Traincurve.py** plotts losses of a training against epochs, call by:
 
`python traincurve.py -d <top_folder/> -i <name des outputfiles> -e checkpoint_epoche` 
for example: `python traincurve.py -d "Runs/se_resnet_trained_final/" -i "se_resnet_final" -e 149`
 
train_chkpt_149.tar has to exist, the file is then loaded and used. The plot is safed at: 'evaluation/plots' 
(both dirs are created).

**evaluatie.py** calculates the accuracy, call in shell: 

`python evaluate.py --dir <rootdir/experiment/> --epoch <epoch to> `
e.g. in shell: `python evaluate.py --dir Runs/se_resnet_trained/ --epoch 149`

loops over all folds and calculates + stores the accuracies in a file in the root folder of the experiment
you might change the model in line 45 from resnet to se_resnet (see comment)


## Result

All trained models/results/plots can be found in the folder `Runs`

with the configuration from above the following results are obtained. There is no significant difference between the two networks.
appart from that: the training curve shows that the learning rate was probably reduced too early (The curve flattens significantly at epoch 40)

**Case 1**

without SE-module:

folder: 1, accuracy: 0.8867491079720417 

folder: 2, accuracy: 0.8888019942323672 

folder: 3, accuracy: 0.8866024732391613 

folder: 4, accuracy: 0.8881176988122587 

with SE-module:

folder: 1, accuracy: 0.8850383694217704 

folder: 2, accuracy: 0.8843540740016619 

folder: 3, accuracy: 0.8831321178943252 

folder: 4, accuracy: 0.8847450999560096


better results have been obtained in an earlier run without using a learning rate scheduler (no learning rate reduction).
In this case though, the use of the SE-module even worsens the results (not dramatic though). 


**Case 2:** without reducing the learning rate

without SE-module:

folder: 1, accuracy: 0.9005327728627988 

folder: 2, accuracy: 0.8948628965247568 

folder: 3, accuracy: 0.9013148247714942 

folder: 4, accuracy: 0.9006305293513857 

with SE-module:

folder: 1, accuracy: 0.8971112957622562 

folder: 2, accuracy: 0.8964758785864412 

folder: 3, accuracy: 0.8962803656092673 

folder: 4, accuracy: 0.8983821301138863 

hughe differences have shown up in a previous training with a much higher initial learning rate (0.1). The model without SE-module struggles tremendously and lacks robustness (high variance of accuracies), while the other model performs well:

**Case 3:** (too) high learning rate

with SE-module:

folder: 1, accuracy: 0.9166625934796422 

folder: 2, accuracy: 0.9266337553155091 

folder: 3, accuracy: 0.8540495625397135 

folder: 4, accuracy: 0.9008260423285596 

without SE-module:

folder: 1, accuracy: 0.6791143262134024 

folder: 2, accuracy: 0.8304902487902635 

folder: 3, accuracy: 0.8525832152109096 

folder: 4, accuracy: 0.7123515323329586 

### Further notes/assumptions:

* I assume that the optimal configuration in terms of initial learning rate and learning rate step has just not been found yet. 
* There are also other measures beside accuracy (e.g. precision, recall, ...) that should to be considered in order to assess the performance of the model.
* This task (classification of 4 classes) is not very difficult for a neural network and it might be possible that the gain of the SE-module is stronger in harder tasks.
* SE-module seems to be increase the robustness and overall performance in unfavorable settings of hyperparameters.
* There might be investigated different settings of the reduction rate



## References

[paper](https://arxiv.org/pdf/1709.01507.pdf)

[authors' Caffe implementation](https://github.com/hujie-frank/SENet)

[pytorch implemenation (base of this repo)](https://github.com/moskomule/senet.pytorch/tree/58844943617b5215f2d3eab149735ac4a66ed393)

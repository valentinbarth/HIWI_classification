# SENet in pytorch for medical image analysis

An implementation of SENet, proposed in **Squeeze-and-Excitation Networks** by Jie Hu, Li Shen and Gang Sun, who are the winners of ILSVRC 2017 classification competition. The baseline (model implementation in pytorch) is taken as it is (for the most part) from [moskomule](https://github.com/moskomule/senet.pytorch/tree/58844943617b5215f2d3eab149735ac4a66ed393). The aim of this poject is to assess the SE-module on medical images by comparing it with the plain resnet model.


## Pre-requirements

The codebase is tested on the following setting.

* Python>=3.6.9
* PyTorch>=1.6.0
* torchvision>=0.7

You can use the files in the folder `Docker` to accomplish these settings.
1. open terminal and navigate to the `Docker` dir
2. to configure the rootless docker type `. config_rootless_docker.sh` in the terminal (change user and UID in the file before)
3. build the docker image by executing `bash build_pytorch_image.sh`
4. start a container with `bash run_pytorch.sh` (change the mounted volume and the workdir in the file before)
5. you can check the torch verion, the torchvision verion and the availability of the GPU by running `python check_pytorch.py` and the python verion by just        typing `python --version`

## Dataset and Preprocessing

The dataset is supplied by Marlen (Dr. Weiss?). The preprocessing of the data is done in the module `rearrange_dataset`. The images are cropped (further data augmentation is not performed but might be added) checked for white space (and in case of more than 50% white discarded - too much background)  The dataset is split into a test (20%) and a train (80%) set. The class will be denoted in the name (and not in the folder anymore).

Before running `python rearrange_dataset.py` you should give the right datapaths in the beginning of the file:
* `datapath` should be the path to the original data supplied by Marlen
* `new_datapath` should be the path to the new created dataset

Some hyperparameters might be adjusted if wished (but defaults are given):
- the number of images split for testing (when calling the function in the end)
- the size of the cropped images: default is 150x150, I would not go smaller because then there is not enough structure in one sample (also when calling)
- the threshold for discarding white images might be reduced to create more data (in the `crop` function)

## Training

Different Resnet and SE-ResNet architectures (18, 34, 50, 101, 152/20, 32) are implemented and can be choosen.

* `python cifar.py` runs SE-ResNet20 with Cifar10 dataset.

* `python imagenet.py` and `python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} imagenet.py` run SE-ResNet50 with ImageNet(2012) dataset,
    + You need to prepare dataset by yourself in `~/.torch/data` or set an enviroment variable `IMAGENET_ROOT=${PATH_TO_YOUR_IMAGENET}`
    + First download files and then follow the [instruction](https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset).
    + The number of workers and some hyper parameters are fixed so check and change them if you need.
    + This script uses all GPUs available. To specify GPUs, use `CUDA_VISIBLE_DEVICES` variable. (e.g. `CUDA_VISIBLE_DEVICES=1,2` to use GPU 1 and 2)

For SE-Inception-v3, the input size is required to be 299x299 [as the original Inception](https://github.com/tensorflow/models/tree/master/inception).


### For training

To run `cifar.py` or `imagenet.py`, you need

* `pip install git+https://github.com/moskomule/homura@2020.05`

## hub

You can use some SE-ResNet (`se_resnet{20, 56, 50, 101}`) via `torch.hub`.

```python
import torch.hub
hub_model = torch.hub.load(
    'moskomule/senet.pytorch',
    'se_resnet20',
    num_classes=10)
```

Also, a pretrained SE-ResNet50 model is available.

```python
import torch.hub
hub_model = torch.hub.load(
    'moskomule/senet.pytorch',
    'se_resnet50',
    pretrained=True,)
 ```

## Result

### SE-ResNet20/Cifar10

```
python cifar.py [--baseline]
```

Note that the CIFAR-10 dataset expected to be under `~/.torch/data`.

|                  | ResNet20       | SE-ResNet20 (reduction 4 or 8)    |
|:-------------    | :------------- | :------------- |
|max. test accuracy|  92%           | 93%            |

### SE-ResNet50/ImageNet

```
python [-m torch.distributed.launch --nproc_per_node=${NUM_GPUS}] imagenet.py
```

The option [-m ...] is for distributed training. Note that the Imagenet dataset is expected to be under `~/.torch/data` or specified as `IMAGENET_ROOT=${PATH_TO_IMAGENET}`.

*The initial learning rate and mini-batch size are different from the original version because of my computational resource* .

|                  | ResNet         | SE-ResNet      |
|:-------------    | :------------- | :------------- |
|max. test accuracy(top1)|  76.15 %(*)             | 77.06% (**)          |


+ (*): [ResNet-50 in torchvision](https://pytorch.org/docs/stable/torchvision/models.html)

+ (**): When using `imagenet.py` with the `--distributed` setting on 8 GPUs. The weight is [available](https://github.com/moskomule/senet.pytorch/releases/download/archive/seresnet50-60a8950a85b2b.pkl).

```python
# !wget https://github.com/moskomule/senet.pytorch/releases/download/archive/seresnet50-60a8950a85b2b.pkl

senet = se_resnet50(num_classes=1000)
senet.load_state_dict(torch.load("seresnet50-60a8950a85b2b.pkl"))
```

## Contribution

I cannot maintain this repository actively, but any contributions are welcome. Feel free to send PRs and issues.

## References

[paper](https://arxiv.org/pdf/1709.01507.pdf)

[authors' Caffe implementation](https://github.com/hujie-frank/SENet)

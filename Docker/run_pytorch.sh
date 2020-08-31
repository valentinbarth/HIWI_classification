#!/bin/bash
docker run --rm -it --gpus all -v /home/vbarth:/home/vbarth -w /home/vbarth nvcr.io/nvidia/pytorch:20.03-py3_torch1.6.0



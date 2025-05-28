# Overview
- This directory has code to train a toy neural network model on toy dataset using Pytorch

# Environment
    - Create conda env
        - `mlenv2c`: `conda env create -f environment_mlenv2c.yml` 

# Commands:
- To train using single GPU with no support for multi-GPU training
    - `python train.py`
- To train using multiple GPUs for distributed data parallel
    - To manage processes on our own using `multiprocessing.spawn`
        - `CUDA_VISIBLE_DEVICES=<> python train_ddp.py`
    - To manage processes automatically uisng `torchrun`
        - Change `USE_TORCHRUN` to  `True` in `main` section of `train_ddp.py`
        - `CUDA_VISIBLE_DEVICES=<> python --nproc_per_node=2 train_ddp.py`
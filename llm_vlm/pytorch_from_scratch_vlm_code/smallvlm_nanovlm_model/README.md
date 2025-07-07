# Overview
- This directory has my experiments after cloning nanoVLM from https://github.com/huggingface/nanoVLM
- For detailed instructions, follow README of nanoVLM directory directly, but here are my instructions in short

# Environment
- Use environment.yml file to setup the conda env

# Example training command
- python train.py &> logs/train_log

# Example inference command
- CUDA_VISIBLE_DEVICES=6 python generate.py &> logs/generate_log_verbose
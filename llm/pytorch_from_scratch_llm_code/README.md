# Overview:
- This directory has following reference code from scratch using Pytorch (to develop deep understanding)
    - Code to define LLM models GPT2 and LLama (2 to 3.2) and all its components
    - Code to train LLM to predict next token 
    - Code to finetune this LLM to be a spam classifier
    - Code to finetune this LLM to follow instructions
- The code is inspired from `https://github.com/rasbt/LLMs-from-scratch`, but it has been heavily modified & restructured in many parts to make it modular & easy to maintain so that it can be easily modified for newer datasets or model architectures. 
    - For example, it can be used an initial reference code structure to build a production code base (Of course in such a scenario, it would need more refactoring, cleanup, etc.).

# Dataset
    - For LLM next token prediction, 
        - Download data from `https://github.com/pgcorpus/gutenberg.git` using its download_data.py
        - Follow instructions to process the dataset as described in the `https://github.com/rasbt/LLMs-from-scratch` README.md.
        - Create a soft link to this processed dataset from ../data/gutenberg_preprocessed
    - For LLM Classifer
        - Use classifer section of `download_data.py` to download dataset
        - The data will be located at ../data/sms_spam_collection/
    - For Instruct LLM
        - Use instruct LLM section of `download_data.py` to download dataset
        - The data will be located at ../data/instruction-data.json

# Environment
    - Create conda env
        - `mlenv2`:  `conda env create -f environment_mlenv2.yml`
        - `mlenv2c`: `conda env create -f environment_mlenv2c.yml` 

# Pretrained models
    - Use conda env `mlenv2c`
    - Use `download_models.py` code to download pretrained weights for GPT-2, Llama

# High level code calls
    - Train next token prediction LLM
        - Use conda env `mlenv2`
        - `CUDA_VISIBLE_DEVICES=<> python train_next_token.py`  
    - Finetune LLM to classify spam
        - Use conda env `mlenv2` (Use conda env `mlenv2c` if using OpenAI weights instead of your own weights for GPT2 model )
        - `CUDA_VISIBLE_DEVICES=<> python finetune_classifier.py`
    - Finetune LLM to follow instructions
        - Use conda env `mlenv2` (Use conda env `mlenv2c` if using OpenAI weights instead of your own weights for GPT2 model)
        - `CUDA_VISIBLE_DEVICES=<> python finetune_instruct_llm.py`

# Code structure



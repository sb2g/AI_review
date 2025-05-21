# Overview
- This part of the has the reference codes to train / run inference on LLMs (also applicable to VLMs). It shows how to:
    - Use Huggingface library to train, finetune or run inference on LLMs
    - Use LLM APIs like ChatGPT
    - Use Pytorch to define key components of LLMs/VLMs from scratch. Define models like GPT2 & Llama3.*.  Train them to predict next token & later finetune them for downstream tasks like classification or to follow instructions.

# Key directoies here
    - `hf_api_notebooks`: All the notebooks on how to use Huggingface library or APIs
    - `pytorch_from_scratch_notebooks`: All the notebooks on how to use Pytorch from scratch to define & train LLMs
    - `pytorch_from_scratch_llm_code`: All the code organized in modular form
    - `misc_notebooks`: Miscellaneous notebooks
    - `data`: Datasets are to be stored here. (Omitted large files to avoid git repo bloat)
    - 'output_dir`: Models & training outputs are saved here. (Omitted from checkin to avoid git repo bloat)
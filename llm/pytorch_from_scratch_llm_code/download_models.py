"""
Helper script to download GPT/Llama models & load weights into custom LLMs with their architectures
Reference: https://github.com/rasbt/LLMs-from-scratch
"""

import os
import urllib.request
# import requests
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import torch

from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from llm_configs_1 import *
from llm_models_gpt_llama import GPT2Model, Llama3Model

def download_file(url, destination, backup_url=None):
    def _attempt_download(download_url):
        with urllib.request.urlopen(download_url) as response:
            # Get the total file size from headers, defaulting to 0 if not present
            file_size = int(response.headers.get("Content-Length", 0))

            # Check if file exists and has the same size
            if os.path.exists(destination):
                file_size_local = os.path.getsize(destination)
                if file_size == file_size_local:
                    print(f"File already exists and is up-to-date: {destination}")
                    return True  # Indicate success without re-downloading

            block_size = 1024  # 1 Kilobyte

            # Initialize the progress bar with total file size
            progress_bar_description = os.path.basename(download_url)
            with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
                with open(destination, "wb") as file:
                    while True:
                        chunk = response.read(block_size)
                        if not chunk:
                            break
                        file.write(chunk)
                        progress_bar.update(len(chunk))
            return True

    try:
        if _attempt_download(url):
            return
    except (urllib.error.HTTPError, urllib.error.URLError):
        if backup_url is not None:
            print(f"Primary URL ({url}) failed. Attempting backup URL: {backup_url}")
            try:
                if _attempt_download(backup_url):
                    return
            except urllib.error.HTTPError:
                pass

        # If we reach here, both attempts have failed
        error_message = (
            f"Failed to download from both primary URL ({url})"
            f"{' and backup URL (' + backup_url + ')' if backup_url else ''}."
            "\nCheck your internet connection or the file availability.\n"
            "For help, visit: https://github.com/rasbt/LLMs-from-scratch/discussions/273"
        )
        print(error_message)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# Alternative way using `requests`
"""
def download_file(url, destination):
    # Send a GET request to download the file in streaming mode
    response = requests.get(url, stream=True)

    # Get the total file size from headers, defaulting to 0 if not present
    file_size = int(response.headers.get("content-length", 0))

    # Check if file exists and has the same size
    if os.path.exists(destination):
        file_size_local = os.path.getsize(destination)
        if file_size == file_size_local:
            print(f"File already exists and is up-to-date: {destination}")
            return

    # Define the block size for reading the file
    block_size = 1024  # 1 Kilobyte

    # Initialize the progress bar with total file size
    progress_bar_description = url.split("/")[-1]  # Extract filename from URL
    with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
        # Open the destination file in binary write mode
        with open(destination, "wb") as file:
            # Iterate over the file data in chunks
            for chunk in response.iter_content(block_size):
                progress_bar.update(len(chunk))  # Update progress bar
                file.write(chunk)  # Write the chunk to the file
"""


def download_gpt2(model_size, models_dir):

    # Validate model size
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    # Define paths
    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    backup_base_url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/gpt2"
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]
    print(f"Downloading GPT model size {model_size} to {model_dir}")

    # Download files
    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        backup_url = os.path.join(backup_base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path, backup_url)
    
    print("Finished downloading")

    # Load settings and params
    #return load_gpt2(model_size, models_dir)
    

def load_gpt2(model_size, models_dir):

    print(f"Retrieving parameters GPT2 model size {model_size}")

    # Validate model size
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    # Define paths
    model_dir = os.path.join(models_dir, model_size)

    # Load settings and params
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json"), "r", encoding="utf-8"))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    print("Finished retrieving")

    return settings, params

def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    # Initialize parameters dictionary with empty blocks for each layer
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # Iterate over each variable in the checkpoint
    for name, _ in tf.train.list_variables(ckpt_path):
        # Load the variable and remove singleton dimensions
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # Process the variable name to extract relevant parts
        variable_name_parts = name.split("/")[1:]  # Skip the 'model/' prefix

        # Identify the target dictionary for the variable
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # Recursively access or create nested dictionaries
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # Assign the variable array to the last key
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params

# def assign(left, right):
#     if left.shape != right.shape:
#         raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
#     return torch.nn.Parameter(torch.tensor(right))

def assign(left, right, tensor_name="unknown"):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}")

    if isinstance(right, torch.Tensor):
        return torch.nn.Parameter(right.clone().detach())
    else:
        return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt, params):

    print(f"Loading weights into GPT model")

    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.layers[b].att.wq.weight = assign(
            gpt.layers[b].att.wq.weight, q_w.T)
        gpt.layers[b].att.wk.weight = assign(
            gpt.layers[b].att.wk.weight, k_w.T)
        gpt.layers[b].att.wv.weight = assign(
            gpt.layers[b].att.wv.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.layers[b].att.wq.bias = assign(
            gpt.layers[b].att.wq.bias, q_b)
        gpt.layers[b].att.wk.bias = assign(
            gpt.layers[b].att.wk.bias, k_b)
        gpt.layers[b].att.wv.bias = assign(
            gpt.layers[b].att.wv.bias, v_b)

        gpt.layers[b].att.proj.weight = assign(
            gpt.layers[b].att.proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.layers[b].att.proj.bias = assign(
            gpt.layers[b].att.proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.layers[b].ff.layers[0].weight = assign(
            gpt.layers[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.layers[b].ff.layers[0].bias = assign(
            gpt.layers[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.layers[b].ff.layers[2].weight = assign(
            gpt.layers[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.layers[b].ff.layers[2].bias = assign(
            gpt.layers[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.layers[b].norm1.scale = assign(
            gpt.layers[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.layers[b].norm1.shift = assign(
            gpt.layers[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        gpt.layers[b].norm2.scale = assign(
            gpt.layers[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.layers[b].norm2.shift = assign(
            gpt.layers[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


def download_llama(model_size="1B", model_dir="../output_dir/Llama"):

    print(f"Downloading Llama model size {model_size} to {model_dir}")

    os.makedirs(model_dir, exist_ok=True)

    if model_size == "1B":
        weights_file = hf_hub_download(
            repo_id=f"meta-llama/Llama-3.2-{model_size}-Instruct",
            filename="model.safetensors",
            local_dir=os.path.join(model_dir, f"Llama-3.2-{model_size}-Instruct")
        )
        print(f"weights_file: {weights_file}")
        combined_weights = load_file(weights_file)

    else:
        combined_weights = {}
        for i in range(1, 3):
            weights_file = hf_hub_download(
                repo_id=f"meta-llama/Llama-3.2-{model_size}-Instruct",
                filename=f"model-0000{i}-of-00002.safetensors",
                local_dir=os.path.join(model_dir, f"Llama-3.2-{model_size}-Instruct")
            )
            print(f"weights_file: {weights_file}")
            current_weights = load_file(weights_file)
            combined_weights.update(current_weights)

    print("Finished loading")

    return combined_weights


def load_llama(model_size="1B", model_dir="../output_dir/Llama"):

    print(f"Loading Llama model size {model_size}")

    if model_size == "1B":
        filename="model.safetensors"
        weights_file = os.path.join(model_dir, f"Llama-3.2-{model_size}-Instruct", filename)
        print(f"weights_file: {weights_file}")
        combined_weights = load_file(weights_file)

    else:
        combined_weights = {}
        for i in range(1, 3):
            filename="model-0000{i}-of-00002.safetensors"
            weights_file = os.path.join(model_dir, f"Llama-3.2-{model_size}-Instruct", filename)
            print(f"weights_file: {weights_file}")
            current_weights = load_file(weights_file)
            combined_weights.update(current_weights)

    print("Finished loading")

    return combined_weights


def load_weights_into_llama(model, param_config, params):

    print(f"Loading weights into Llama model")

    model.tok_emb.weight = assign(model.tok_emb.weight, params["model.embed_tokens.weight"], "model.embed_tokens.weight")

    for l in range(param_config["n_layers"]):

        # Load attention weights
        model.layers[l].att.wq.weight = assign(
            model.layers[l].att.wq.weight,
            params[f"model.layers.{l}.self_attn.q_proj.weight"],
            f"model.layers.{l}.self_attn.q_proj.weight"
        )
        model.layers[l].att.wk.weight = assign(
            model.layers[l].att.wk.weight,
            params[f"model.layers.{l}.self_attn.k_proj.weight"],
            f"model.layers.{l}.self_attn.k_proj.weight"
        )
        model.layers[l].att.wv.weight = assign(
            model.layers[l].att.wv.weight,
            params[f"model.layers.{l}.self_attn.v_proj.weight"],
            f"model.layers.{l}.self_attn.v_proj.weight"
        )
        model.layers[l].att.proj.weight = assign(
            model.layers[l].att.proj.weight,
            params[f"model.layers.{l}.self_attn.o_proj.weight"],
            f"model.layers.{l}.self_attn.o_proj.weight"
        )
        model.layers[l].norm1.scale = assign(
            model.layers[l].norm1.scale,
            params[f"model.layers.{l}.input_layernorm.weight"],
            f"model.layers.{l}.input_layernorm.weight"
        )

        # Load FeedForward weights
        model.layers[l].ff.fc1.weight = assign(
            model.layers[l].ff.fc1.weight,
            params[f"model.layers.{l}.mlp.gate_proj.weight"],
            f"model.layers.{l}.mlp.gate_proj.weight"
        )
        model.layers[l].ff.fc2.weight = assign(
            model.layers[l].ff.fc2.weight,
            params[f"model.layers.{l}.mlp.up_proj.weight"],
            f"model.layers.{l}.mlp.up_proj.weight"
        )
        model.layers[l].ff.fc3.weight = assign(
            model.layers[l].ff.fc3.weight,
            params[f"model.layers.{l}.mlp.down_proj.weight"],
            f"model.layers.{l}.mlp.down_proj.weight"
        )
        model.layers[l].norm2.scale = assign(
            model.layers[l].norm2.scale,
            params[f"model.layers.{l}.post_attention_layernorm.weight"],
            f"model.layers.{l}.post_attention_layernorm.weight"
        )

    # Load output layer weights
    model.final_norm.scale = assign(model.final_norm.scale, params["model.norm.weight"], "model.norm.weight")

    if "lm_head.weight" in params.keys():
        model.out_head.weight = assign(model.out_head.weight, params["lm_head.weight"], "lm_head.weight")
    else:
        model.out_head.weight = assign(model.out_head.weight, params["model.embed_tokens.weight"], "model.embed_tokens.weight")
        print("Model uses weight tying.")


if __name__ == "__main__":

    # GPT model
    if 0:
        
        model_size = "124M"
        
        # Download model
        if 0:
            download_gpt2(model_size=model_size, models_dir="../output_dir/gpt2")

        # Load weigths into model
        if 1:
            settings, params = load_gpt2(model_size=model_size, models_dir="../output_dir/gpt2")
            LLM_CONFIG = GPT_CONFIG_124M
            LLM_CONFIG['qkv_bias'] = True # Since GPT2 OpenAI weights have this set to True
            model = GPT2Model(LLM_CONFIG)
            load_weights_into_gpt(model, params)
            print(f"Done loading weights")

    # Llama models
    if 1:

        model_size="1B"

        # Download model
        if 0:
            download_llama(model_size=model_size, models_dir="../output_dir/Llama")

        # Load weigths into model
        if 1:
            combined_weights = load_llama(model_size="1B")
            LLM_CONFIG = LLAMA32_CONFIG_1B
            model = Llama3Model(LLM_CONFIG)
            load_weights_into_llama(model, LLM_CONFIG, combined_weights)
            print(f"Done loading weights")
            #model.to(device)
            del combined_weights  # free up memory

            # Sanity check
            print("Weight tying:", torch.equal(model.tok_emb.weight, model.out_head.weight))
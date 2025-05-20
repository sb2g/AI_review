"""
Script to finetune an LLM to follow instructions.
Reference : https://github.com/rasbt/LLMs-from-scratch
"""

import argparse
import os
from pathlib import Path
import time
import tiktoken
import torch

from llm_configs_1 import *
from llm_models_gpt_llama import GPT2Model
from data import create_instruct_llm_dataloaders, format_input
from train_helpers import calc_loss_batch, calc_loss_loader, plot_values
from infer_helpers import generate_and_print_sample, generate, text_to_token_ids, token_ids_to_text
import tqdm
import json
from download_models import load_weights_into_gpt, load_gpt2


def train_model_simple(model, 
                       train_loader, 
                       val_loader, 
                       optimizer, 
                       device, 
                       num_epochs,
                       eval_freq, 
                       eval_num_batches, 
                       start_context, 
                       tokenizer):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1
            print(f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Batch train loss {loss.item():.3f}")
            
            # Optional evaluation step
            if global_step % eval_freq == 0:
                model.eval()
                with torch.no_grad():
                    train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_num_batches)
                    val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_num_batches)
                model.train()
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Print a sample text after each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Finetune a LLM model for instruct LLM")
    parser.add_argument('--output_dir', type=str, default='../output_dir/model_next_token/',
                        help='Directory where the model checkpoints will be saved')
    parser.add_argument("--debug", default=False, action="store_true",
                        help=("Uses a very small model for debugging purposes"))
    args = parser.parse_args()

    # Config
    lr = 5e-5
    num_epochs = 2
    eval_freq = 5
    eval_num_batches = 5
    batch_size = 8
    train_ratio = 0.85
    test_ratio = 0.1
    num_workers = 0

    # Seed and device
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # LLM Config
    LLM_CONFIG = GPT_CONFIG_124M

    # Initialize Tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create data loaders
    train_loader, val_loader, val_data, test_data = create_instruct_llm_dataloaders(device,
                                                        train_ratio=train_ratio,
                                                        test_ratio=test_ratio,
                                                        batch_size=batch_size,
                                                        num_workers=num_workers,
                                                        tokenizer=tokenizer
                                                    )

    # ----
    # Load pretrained model
    # ---

    if args.debug:
        LLM_CONFIG = {
            "vocab_size": 50257,     # Vocabulary size
            "context_length": 120,    # Context length
            "emb_dim": 12,           # Embedding dimension
            "n_heads": 2,            # Number of attention heads
            "n_layers": 2,           # Number of layers
            "drop_rate": 0.0,        # Dropout rate, deactivated via 0.0 as dropout in LLMs is not recommended anymore
            "qkv_bias": False        # Query-key-value bias
        }
        model = GPT2Model(LLM_CONFIG)
    else:
        model_size = "124M"
        #settings, params = download_and_load_gpt2(model_size=model_size, models_dir="./output_dir/gpt2")
        settings, params = load_gpt2(model_size=model_size, models_dir="./output_dir/gpt2")
        model = GPT2Model(LLM_CONFIG)
        load_weights_into_gpt(model, params)
    model.eval()
    model.to(device)

    # ---
    # Finetuning the model
    # ---

    print("Initial losses")
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
    print("Training loss:", train_loss)
    print("Validation loss:", val_loss)

    start_time = time.time()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    train_losses, val_losses, tokens_seen = train_model_simple(
                                                model, 
                                                train_loader, 
                                                val_loader, 
                                                optimizer, 
                                                device,
                                                num_epochs=num_epochs, 
                                                eval_freq=eval_freq, 
                                                eval_num_batches=eval_num_batches,
                                                start_context=format_input(val_data[0]), 
                                                tokenizer=tokenizer
                                            )
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    # ---
    # Plot results & save model
    # ---

    # loss plot
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_values(epochs_tensor, tokens_seen, train_losses, val_losses)
    print(50*"-")

    # Save model
    model_path =  os.path.join(output_dir,"model_instruct_llm.pth")
    print(f"Model saved as {model_path}")
    torch.save(model.state_dict(), model_path)
    print(f"Maximum GPU memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # ---
    # Inference on test set
    # ---

    print("Generating responses")
    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
        input_text = format_input(entry)
        token_ids = generate(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=256,
            context_size=LLM_CONFIG["context_length"],
            eos_id=50256
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)
        response_text = generated_text[len(input_text):].replace("### Response:", "").strip()
        test_data[i]["model_response"] = response_text

    test_data_path = os.path.join(output_dir, "test_set_results.json")
    with open(test_data_path, "w") as file:
        json.dump(test_data, file, indent=4)  # "indent" for pretty-printing
    print(f"Responses saved as {test_data_path}")


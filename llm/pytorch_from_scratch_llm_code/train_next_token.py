"""
Script to train an LLM to predict next tokens on books from Project Gutenberg.
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
from data import create_gpt_dataloaders, create_gpt_dataloaders_full_dataset
from train_helpers import calc_loss_batch, calc_loss_loader, print_eta, plot_values
from infer_helpers import generate_and_print_sample

def train_model_simple_multiple_data_files(model, 
                                        optimizer, 
                                        device, 
                                        num_epochs=1,
                                        eval_freq=100, 
                                        eval_num_batches=1, 
                                        print_sample_freq=100, 
                                        start_context="",
                                        save_ckpt_freq=100, 
                                        tokenizer=None,
                                        batch_size=1024, 
                                        train_ratio=0.90, 
                                        num_workers=0,
                                        output_dir="../output_dir/"):

    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1
    start_time = time.time()

    # Check data
    data_dir = '../data/gutenberg_preprocessed'
    all_files = [os.path.join(path, name) for path, subdirs, files
                 in os.walk(data_dir) for name in files if name.endswith((".txt"))]
    total_files = len(all_files)
    if total_files == 0:
        print("No training text files found. Make sure you "
              "selected the correct input directory")
        quit()
    print("Total files:", total_files)

    try:
        for epoch in range(num_epochs):

            # Iterate over the books in the training corpus
            for index, file_path in enumerate(all_files, 1):
                book_start_time = time.time()
                print(f"Tokenizing file {index} of {total_files}: {file_path}")

                # Initialize new data loaders for each book
                train_loader, val_loader = create_gpt_dataloaders(
                    file_path,
                    train_ratio=train_ratio,
                    batch_size=batch_size,
                    max_length=LLM_CONFIG["context_length"],
                    stride=LLM_CONFIG["context_length"],
                    num_workers=num_workers,
                    tokenizer=tokenizer
                )

                print("Training ...")
                model.train()
                for input_batch, target_batch in train_loader:
                    optimizer.zero_grad()
                    loss = calc_loss_batch(input_batch, target_batch, model, device)
                    loss.backward()
                    optimizer.step()
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
                        print(f"Ep {epoch+1} (Step {global_step}): "
                              f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

                    # Generate text passage
                    if global_step % print_sample_freq == 0:
                        generate_and_print_sample(
                            model, tokenizer, device, start_context
                        )

                if global_step % save_ckpt_freq:
                    file_name = os.path.join(f"model_pg_{global_step}.pth")
                    torch.save(model.state_dict(), file_name)
                    print(f"Saved {file_name}")

                print_eta(start_time, book_start_time, index, total_files)

            # Print a sample text after each epoch
            generate_and_print_sample(
                model, tokenizer, device, start_context
            )

    except KeyboardInterrupt:
        file_name = os.path.join(output_dir, f"model_pg_{global_step}_interrupted.pth")
        torch.save(model.state_dict(), file_name)
        print(f"Saved {file_name}")

    return train_losses, val_losses, track_tokens_seen


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
    tokens_seen = 0
    global_step = -1

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

    parser = argparse.ArgumentParser(description='GPT Model Training Configuration')

    parser.add_argument('--data_dir', type=str, default='../data/gutenberg_preprocessed',
                        help='Directory containing the training data files. Hardcoded here')
    parser.add_argument('--output_dir', type=str, default='../output_dir/model_next_token/',
                        help='Directory where the model checkpoints will be saved')
    parser.add_argument('--num_epochs', type=int, default=1,
                        help='Number of epochs to train the model')
    parser.add_argument('--print_sample_freq', type=int, default=1000,
                        help='Iterations between printing sample outputs')
    parser.add_argument('--eval_freq', type=int, default=100,
                        help='Frequency of evaluations during training')
    parser.add_argument('--save_ckpt_freq', type=int, default=100,
                        help='Frequency of saving model checkpoints during training')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate for the optimizer')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    #parser.add_argument('--debug', type=bool, default=False,
    #                    help='Uses a very small model for debugging purposes')
    parser.add_argument("--debug", default=False, action="store_true",
                        help=("Uses a very small model for debugging purposes"))

    args = parser.parse_args()


    # Config
    lr = 5e-4
    num_epochs = 1
    eval_freq = 100
    save_ckpt_freq = 100
    print_sample_freq = 100
    eval_num_batches = 1
    batch_size = 8
    train_ratio = 0.9
    num_workers = 0
    
    # LLM Config
    LLM_CONFIG = GPT_CONFIG_124M
    if args.debug:
        # Small version of GPT_CONFIG_124M
        LLM_CONFIG = {
            "vocab_size": 50257,     # Vocabulary size
            "context_length": 10,    # Context length
            "emb_dim": 12,           # Embedding dimension
            "n_heads": 2,            # Number of attention heads
            "n_layers": 2,           # Number of layers
            "drop_rate": 0.0,        # Dropout rate, deactivated via 0.0 as dropout in LLMs is not recommended anymore
            "qkv_bias": False        # Query-key-value bias
        }
        print(f"Debug mode: Using smaller model config")

    # Seed and device
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize model
    model = GPT2Model(LLM_CONFIG)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)

    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Train model
    if 0:
        # Create dataloader with all files together
        # Tokenize all text files together and train on it
        # Needs CPU resources to tokenize entire dataset at same time
        train_loader, val_loader = create_gpt_dataloaders_full_dataset(
                                        train_ratio=train_ratio,
                                        batch_size=batch_size,
                                        max_length=LLM_CONFIG["context_length"],
                                        stride=LLM_CONFIG["context_length"],
                                        num_workers=num_workers,
                                        tokenizer=tokenizer
                                    )
        train_losses, val_losses, tokens_seen = train_model_simple(
                                                                    model, 
                                                                    train_loader, 
                                                                    val_loader, 
                                                                    optimizer, 
                                                                    device,
                                                                    num_epochs=num_epochs, 
                                                                    eval_freq=eval_freq, 
                                                                    eval_num_batches=eval_num_batches,
                                                                    start_context='',
                                                                    tokenizer=tokenizer
                                                                )
    else:
        # Create dataloder per file
        # Tokenize one file at a time and train on it 
        # Wastes runtime on re-tokenizing & re-creating dataloaders for the files on each epoch
        train_losses, val_losses, tokens_seen = train_model_simple_multiple_data_files(
                                                    model, 
                                                    optimizer, 
                                                    device,
                                                    batch_size=batch_size,
                                                    num_epochs=num_epochs,
                                                    eval_freq=eval_freq,
                                                    eval_num_batches=eval_num_batches,
                                                    print_sample_freq=print_sample_freq,
                                                    save_ckpt_freq=save_ckpt_freq,
                                                    start_context="Every effort moves you",
                                                    train_ratio=train_ratio,
                                                    tokenizer=tokenizer,
                                                    num_workers=num_workers,
                                                    output_dir=output_dir
                                                )



    # Plot results
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_values(epochs_tensor, tokens_seen, train_losses, val_losses, output_dir)
    print(50*"-")

    # Save model
    model_path =  os.path.join(output_dir, "model_gutenberg.pth")
    print(f"Model saved as {model_path}")
    torch.save(model.state_dict(), model_path)
    print(f"Maximum GPU memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # # Load model
    # model = GPT2Model(LLM_CONFIG)
    # model.load_state_dict(torch.load(model_path, weights_only=True))


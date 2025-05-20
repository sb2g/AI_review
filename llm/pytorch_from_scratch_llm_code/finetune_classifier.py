"""
Script to finetune an LLM to classify text.
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
from data import create_spam_cls_dataloaders
from train_helpers import calc_loss_batch, calc_loss_loader, calc_accuracy_loader, plot_values
from download_models import load_weights_into_gpt, load_gpt2

def train_classifier_simple(model, 
                            train_loader, 
                            val_loader, 
                            optimizer, 
                            device, 
                            num_epochs,
                            eval_freq, 
                            eval_num_batches, 
                            tokenizer):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients
            examples_seen += input_batch.shape[0]  # New: track examples instead of tokens
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
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Calculate accuracy after each epoch
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_num_batches)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_num_batches)
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen



if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Finetune a LLM model for classification"
    )
    parser.add_argument('--output_dir', type=str, default='../output_dir/model_classifier/',
                        help='Directory where the model checkpoints will be saved')
    parser.add_argument("--debug", default=False, action="store_true",
                        help=("Uses a very small model for debugging purposes"))
    args = parser.parse_args()

    # Config
    lr = 5e-5
    num_epochs = 5
    eval_freq = 50
    eval_num_batches = 5
    batch_size = 8
    train_ratio = 0.7
    val_ratio = 0.1
    num_workers = 0
    num_classes = 2

    # Seed and device
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # LLM Config
    LLM_CONFIG = GPT_CONFIG_124M

    # # Initialize model & tokenizer
    # model = GPT2Model(LLM_CONFIG)
    # model.to(device)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    
    # Initialize Tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create data loaders
    train_loader, val_loader, test_loader = create_spam_cls_dataloaders(
                                                        train_ratio=train_ratio,
                                                        val_ratio=val_ratio,
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
        #settings, params = download_and_load_gpt2(model_size=model_size, models_dir="../output_dir/gpt2")
        settings, params = load_gpt2(model_size=model_size, models_dir="../output_dir/gpt2")
        model = GPT2Model(LLM_CONFIG)
        load_weights_into_gpt(model, params)
    model.eval()
    model.to(device)

    # ---
    # Modify pretrained model
    # ---

    for param in model.parameters():
        param.requires_grad = False
    model.out_head = torch.nn.Linear(in_features=LLM_CONFIG["emb_dim"], out_features=num_classes)
    model.to(device)
    for param in model.layers[-1].parameters():
        param.requires_grad = True
    for param in model.final_norm.parameters():
        param.requires_grad = True

    # ---
    # Finetune modified model
    # ---

    start_time = time.time()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
                                                                        model, 
                                                                        train_loader, 
                                                                        val_loader, 
                                                                        optimizer, 
                                                                        device,
                                                                        num_epochs=num_epochs, 
                                                                        eval_freq=eval_freq, 
                                                                        eval_num_batches=eval_num_batches,
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
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))
    plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)

    # accuracy plot
    epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))
    plot_values(epochs_tensor, examples_seen_tensor, train_accs, val_accs, label="accuracy")

    # Save model
    model_path =  os.path.join(output_dir, "model_spam_cls.pth")
    print(f"Model saved as {model_path}")
    torch.save(model.state_dict(), model_path)
    print(f"Maximum GPU memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # # Load model
    # model = GPT2Model(LLM_CONFIG)
    # model.load_state_dict(torch.load(model_path, weights_only=True))

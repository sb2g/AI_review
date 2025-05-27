"""
Script to train an toy model on toy dataset
Reference : https://github.com/rasbt/LLMs-from-scratch
"""

import torch
import torch.nn.functional as F

import os
import platform
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from data import prepare_dataset
from model import NeuralNetwork
from train_helpers import compute_accuracy

# function to initialize a distributed process group (1 process / GPU)
# this allows communication among processes
def ddp_setup(rank, world_size):
    """
    Arguments:
        rank: a unique process ID
        world_size: total number of processes in the group
    """
    # rank of machine running rank:0 process
    # here, we assume all GPUs are on the same machine
    os.environ["MASTER_ADDR"] = "localhost"
    # any free port on the machine
    os.environ["MASTER_PORT"] = "12345"

    print(f"Platform system: {platform.system()}, rank: {rank}, world_size: {world_size}")

    # initialize process group
    if platform.system() == "Windows":
        # Disable libuv because PyTorch for Windows isn't built with support
        os.environ["USE_LIBUV"] = "0"
        # Windows users may have to use "gloo" instead of "nccl" as backend
        # gloo: Facebook Collective Communication Library
        init_process_group(backend="gloo", rank=rank, world_size=world_size)
    else:
        # nccl: NVIDIA Collective Communication Library
        init_process_group(backend="nccl", rank=rank, world_size=world_size)

    torch.cuda.set_device(rank)

def train(rank, world_size, num_epochs):

    ddp_setup(rank, world_size)  # initialize process groups

    train_loader, test_loader = prepare_dataset()
    model = NeuralNetwork(num_inputs=2, num_outputs=2)
    model.to(rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    model = DDP(model, device_ids=[rank])  # wrap model with DDP
    # the core model is now accessible as model.module

    for epoch in range(num_epochs):

        # Set sampler to ensure each epoch has a different shuffle order
        train_loader.sampler.set_epoch(epoch)

        model.train()
        for features, labels in train_loader:

            features, labels = features.to(rank), labels.to(rank)  # use rank
            logits = model(features)
            loss = F.cross_entropy(logits, labels)  # Loss function

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"[GPU{rank}] Epoch: {epoch+1:03d}/{num_epochs:03d}"
                  f" | Batchsize {labels.shape[0]:03d}"
                  f" | Train/Val Loss: {loss:.2f}")

    model.eval()

    train_acc = compute_accuracy(model, train_loader, device=rank)
    print(f"[GPU{rank}] Training accuracy", train_acc)
    test_acc = compute_accuracy(model, test_loader, device=rank)
    print(f"[GPU{rank}] Test accuracy", test_acc)

    destroy_process_group()  # cleanly exit distributed mode

if __name__ == "__main__":

    torch.manual_seed(123)
    num_epochs = 3

    #USE_TORCHRUN = False
    USE_TORCHRUN = True

    if USE_TORCHRUN:

        # --- Use torchrun ---

        if "WORLD_SIZE" in os.environ:
            world_size = int(os.environ["WORLD_SIZE"])
        else:
            world_size = 1

        if "LOCAL_RANK" in os.environ:
            rank = int(os.environ["LOCAL_RANK"])
        elif "RANK" in os.environ:
            rank = int(os.environ["RANK"])
        else:
            rank = 0

        print(f"Using torchrun: world_size: {world_size}, rank: {rank}")

        if rank == 0:
            # Only print on rank 0 to avoid duplicate prints from each GPU process
            print("Number of GPUs available:", torch.cuda.device_count())

        train(rank, world_size, num_epochs)

    else:
        
        # --- Use mp spawn ---
        # spawn new processes: spawn will automatically pass the rank
        
        world_size = torch.cuda.device_count()
        
        print(f"Using multiprocessing span: world_size: {world_size}")
        print("Number of GPUs available:", torch.cuda.device_count())

        mp.spawn(train, args=(world_size, num_epochs), nprocs=world_size)
        # nprocs=world_size spawns one process per GPU

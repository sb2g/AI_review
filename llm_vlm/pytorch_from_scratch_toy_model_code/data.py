"""
Dataloaders with toy dataset to train toy model
Reference : https://github.com/rasbt/LLMs-from-scratch
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y

    def __len__(self):
        return self.labels.shape[0]



def prepare_dataset(multi_gpu=True):

    print(f"Preparing toy dataset ... ")
    X_train = torch.tensor([
        [-1.2, 3.1],
        [-0.9, 2.9],
        [-0.5, 2.6],
        [2.3, -1.1],
        [2.7, -1.5]
    ])
    y_train = torch.tensor([0, 0, 0, 1, 1])

    X_test = torch.tensor([
        [-0.8, 2.8],
        [2.6, -1.6],
    ])
    y_test = torch.tensor([0, 1])

    print(f"X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}")
    print(f"X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}")

    factor = 4
    X_train = torch.cat([X_train + torch.randn_like(X_train) * 0.1 for _ in range(factor)])
    y_train = y_train.repeat(factor)
    X_test = torch.cat([X_test + torch.randn_like(X_test) * 0.1 for _ in range(factor)])
    y_test = y_test.repeat(factor)

    print(f"After adding more tensors")
    print(f"X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}")
    print(f"X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}")

    train_ds = ToyDataset(X_train, y_train)
    test_ds = ToyDataset(X_test, y_test)

    if multi_gpu:
        train_loader = DataLoader(
            dataset=train_ds,
            batch_size=2,
            shuffle=False,  # False because of DistributedSampler below
            pin_memory=True,
            drop_last=True,
            # chunk batches across GPUs without overlapping samples
            sampler=DistributedSampler(train_ds)
        )
    else:
        train_loader = DataLoader(
            dataset=train_ds,
            batch_size=2,
            shuffle=True, 
            pin_memory=True,
            drop_last=True,
        )        
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=2,
        shuffle=False,
    )

    return train_loader, test_loader

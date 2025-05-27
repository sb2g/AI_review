"""
Script to train an toy model on toy dataset
Reference : https://github.com/rasbt/LLMs-from-scratch
"""

import torch
import torch.nn.functional as F

from data import prepare_dataset
from model import NeuralNetwork
from train_helpers import compute_accuracy


def train(num_epochs):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # NEW

    train_loader, test_loader = prepare_dataset(multi_gpu=False)

    model = NeuralNetwork(num_inputs=2, num_outputs=2)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    for epoch in range(num_epochs):

        model.train()
        for features, labels in train_loader:

            features, labels = features.to(device), labels.to(device)  # use rank
            logits = model(features)
            loss = F.cross_entropy(logits, labels)  # Loss function

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
                  f" | Batchsize {labels.shape[0]:03d}"
                  f" | Train/Val Loss: {loss:.2f}")

    model.eval()

    train_acc = compute_accuracy(model, train_loader, device=device)
    print(f"Training accuracy", train_acc)
    test_acc = compute_accuracy(model, test_loader, device=device)
    print(f"Test accuracy", test_acc)

if __name__ == "__main__":

    torch.manual_seed(123)

    num_epochs = 3
    train(num_epochs)
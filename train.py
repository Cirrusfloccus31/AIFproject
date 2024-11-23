import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from statistics import mean
import argparse
import numpy as np
from model import load_model
from dataset import get_dataloaders
from settings import PLOT_PATH
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(net, optimizer, train_loader, validation_loader, epochs=10):
    criterion = nn.CrossEntropyLoss(reduction='sum')
    train_loss = []
    validation_loss =[]
    train_accuracy =[]
    validation_accuracy = []
    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        goods = 0.0
        net.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            outputs = net(x)
            loss = criterion(outputs, y)
            running_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            goods += (torch.argmax(outputs, dim=1) == torch.argmax(y, dim=1)).sum().item()
        train_loss.append(running_loss / len(train_loader.dataset))
        train_accuracy.append(goods / len(train_loader.dataset))

        net.eval()
        running_loss = 0.0
        goods = 0.0
        for x, y in validation_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                outputs = net(x)
            loss = criterion(outputs, y)
            running_loss += loss.item()
            goods += (torch.argmax(outputs, dim=1) == torch.argmax(y, dim=1)).sum().item()
        validation_loss.append(running_loss / len(validation_loader.dataset))
        validation_accuracy.append(goods / len(validation_loader.dataset))
    return train_loss, validation_loss, train_accuracy, validation_accuracy

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exp_name',
        type=str,
        default = 'movie',
        help='experiment name',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default = 1,
        help='batch size',
    )
    parser.add_argument(
        '--lr',
        type=float,
        default = 0.0001,
        help='learning rate',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default = 10,
        help='Number of training epochs',
    )
    args = parser.parse_args()
    exp_name = args.exp_name
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr

    # dataloaders
    loaders = get_dataloaders(batch_size=batch_size)
    train_loader = loaders["train"]
    validation_loader = loaders["validation"]
    test_loader = loaders["test"]

    # Create the model
    net = load_model()
    # setting net on device(GPU if available, else CPU)
    net = net.to(device)
    optimizer = Adam(net.parameters(), lr=lr)

    # launch training
    train_loss, validation_loss, train_accuracy, validation_accuracy = train(net, optimizer, train_loader, validation_loader, epochs=epochs)
    
    # Saving the model
    torch.save(net.state_dict(), 'weights/movie_net.pth')

    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(10,5))
    ax1.plot(train_loss, label="train")
    ax1.plot(validation_loss, label="validation")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.set_title("Evolution of Cross Entropy loss during training")
    ax1.legend()
    ax2.plot(train_accuracy, label="train")
    ax2.plot(validation_accuracy, label="validation")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Evolution of accuracy during training")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(PLOT_PATH + "loss.png")

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from statistics import mean
import argparse
from model import load_model
from dataset import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(net, optimizer, train_loader, validation_loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    train_loss = []
    validation_loss =[]
    for epoch in range(epochs):
        running_loss = []
        t = tqdm(train_loader)
        net.train()
        for x, y in t:
            x, y = x.to(device), y.to(device)
            outputs = net(x)
            loss = criterion(outputs, y)
            running_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            mean_loss = mean(running_loss)
            t.set_description(f'training loss: {mean_loss:.7f}')
        train_loss.append(mean_loss)

        t = tqdm(validation_loader)
        net.eval()
        running_loss = []
        for x, y in t:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                outputs = net(x)
            loss = criterion(outputs, y)
            running_loss.append(loss.item())
            mean_loss = mean(running_loss)
            t.set_description(f'validation loss: {mean(running_loss):.7f}')
        validation_loss.append(mean_loss)
    return train_loss, validation_loss

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
        default = 0.001,
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
    train_loader = get_dataloaders(["train"], batch_size, 1)["train"]
    validation_loader = get_dataloaders(["test"], batch_size, 1)["test"]

    
    net = load_model()
    # setting net on device(GPU if available, else CPU)
    net = net.to(device)
    optimizer = Adam(net.parameters(), lr=lr)

    train(net, optimizer, train_loader, validation_loader, epochs=epochs)
    
    # Saving the model
    torch.save(net.state_dict(), 'weights/movie_net.pth')

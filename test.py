import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from dataset import get_dataloaders
from model import load_model
from settings import PLOT_PATH


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test(net, test_loader):
    net.eval()
    goods = 0.0
    num_classes = test_loader.dataset.num_classes
    confusion_matrix = np.zeros((num_classes, num_classes))
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            outputs = net(x)
        confusion_matrix[torch.argmax(outputs, dim=1).item(), torch.argmax(y, dim=1).item()] += 1
    goods = sum([confusion_matrix[i,i] for i in range(num_classes)])
    all = np.sum(confusion_matrix)
    accuracy = goods / all
    genres = test_loader.dataset.genres
    fig = plt.figure()
    plt.imshow(confusion_matrix, origin='upper', interpolation='nearest', aspect='equal')
    ax=plt.gca()
    plt.xticks(range(num_classes), labels=genres)
    plt.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")
    ax.xaxis.set_label_position("top")
    plt.xlabel("True genres")
    plt.yticks(range(num_classes), labels=genres)
    plt.ylabel("Predicted genre")
    plt.title(f"Confusion Matrix on the test dataset Global accuracy = {round(accuracy,3)}")
    plt.tight_layout()
    fig.savefig(PLOT_PATH + "confusion_matrix.png", bbox_inches="tight")
    return accuracy, confusion_matrix

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        default="weights/movie_net.pth",
        type=str,
        help="Path to the weights of the pretrained model"
    )
    args = parser.parse_args()
    weights = args.weights
    
    net = load_model()
    net.load_state_dict(torch.load(weights))
    net = net.to(device)

    loaders = get_dataloaders()
    test_loader = loaders["test"]

    accuracy, confusion_matrix =test(net, test_loader)
    print(accuracy)
    print(confusion_matrix)
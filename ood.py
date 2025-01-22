import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve
import numpy as np
import matplotlib.pyplot as plt
from model import trained_model_logits
from part1.dataset import get_dataloaders
from part1.settings import PLOT_PATH
from dataset import Movie_Dataset
from detect_anomalies import mls, compute_threshold


# Les mêmes prétraitements que dans dataset
preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

def compute_logits(dataset, model, device):
    all_logits = []
    with torch.no_grad():
        for i in range(len(dataset)):
            image, _ = dataset[i]  
            input_tensor = preprocess(image).unsqueeze(0)
            input_tensor = input_tensor.to(device) 
            logits = model(image)
            all_logits.append(logits)
    return torch.cat(all_logits, dim=0)  


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    batch_size = 128
  
    movie_test = Movie_Dataset("test", "MovieGenre", 0.7)
    svhn_test = datasets.SVHN(root='./data', split='test', transform=None, download=True)
    # Extract 10_000 random images from the svhn_test set
    svhn_test, _ = torch.utils.data.random_split(svhn_test, [10_000, len(svhn_test) - 10_000])
    
    # dataloaders
    loaders = get_dataloaders(batch_size=batch_size) #MovieGenre
    test_loader = loaders["test"]
    
    svhn_test_loader = DataLoader(svhn_test, batch_size=batch_size, shuffle=False)
    
    # Classifier entraîné de la partie 1
    weights_model = "movie_net.pth"
    model_without_softmax = trained_model_logits(weights_model)
    model_without_softmax.eval()

    test_logits_negatives = compute_logits(movie_test, model_without_softmax, device)
    test_logits_positives = compute_logits(svhn_test, model_without_softmax, device)
    
    target_tpr = 0.9
    scores_negatives = mls(test_logits_negatives)
    scores_positives = mls(test_logits_positives)
    threshold = compute_threshold(scores_positives, target_tpr)
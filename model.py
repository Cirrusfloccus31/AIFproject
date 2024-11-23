import torch
import torch.nn as nn
from torchvision.models import vgg16
import torch.hub

# Set the cache directory for model weights
torch.hub.set_dir("./cache")


def load_model():
    model = vgg16(weights="IMAGENET1K_V1")

    for param in model.features.parameters():
        param.requires_grad = False

    model.classifier[6] = nn.Sequential(
        nn.Linear(in_features=4096, out_features=10), nn.Softmax(dim=-1)
    )
    return model

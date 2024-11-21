import torch
import torch.nn as nn
from torchvision.models import vgg16

def load_model():
    model = vgg16(pretrained=True)

    for param in model.features.parameters():
        param.requires_grad = False

    model.classifier[6] = nn.Sequential(nn.Linear(in_features=4096, out_features=10),nn.Sigmoid())
    return model



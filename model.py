import torch
import torch.nn as nn
from torchvision.models import vgg16, mobilenet_v3_small
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

def load_model_reco():
    mobilenet = mobilenet_v3_small(pretrained=True)
    #On récupère les features de mobilenet
    model_reco = torch.nn.Sequential(mobilenet.features, mobilenet.avgpool, torch.nn.Flatten())
    model_reco = model_reco.eval() #permet d'enlever le dropout
    return model_reco

import torch
import torch.nn as nn
import torch.hub
from torchvision.models import vgg16, mobilenet_v3_small


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
    mobilenet = mobilenet_v3_small(weights="MobileNet_V3_Small_Weights.IMAGENET1K_V1")
    # On récupère les features de mobilenet
    model_reco = torch.nn.Sequential(
        mobilenet.features, mobilenet.avgpool, torch.nn.Flatten()
    )
    model_reco = model_reco.eval()  # permet d'enlever le dropout
    print("load mobilenet")
    return model_reco


def trained_model_logits(weights_model):
    model = vgg16(weights="IMAGENET1K_V1")
    
    model.classifier[6] = nn.Linear(in_features=4096, out_features=10)
    
    # Charger les poids du modèle entraîné
    model.load_state_dict(torch.load(weights_model, weights_only=True, map_location=torch.device("cpu")))
    
    return model



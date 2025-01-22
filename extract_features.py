import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import ImageAndPathsDataset, transform_MLP_dataset
from model import load_model_reco

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_dataset_features(dataloader, model):
    # Création d'un dataframe contenant les features extraites des images du dataset
    features_list = []
    paths_list = []  # paths vers les images
    for x, paths in tqdm(dataloader):
        with torch.no_grad():
            embeddings = model(x.to(device))
            features_list.extend(embeddings.cpu().numpy())
            paths_list.extend(paths)

    df = pd.DataFrame({"features": features_list, "paths": paths_list})
    image_paths = df["paths"]

    # Enregistrement du dataframe qui associe paths et features
    df.to_csv("features_paths.csv", index=False)
    image_paths.to_csv("image_paths.csv", index=False)


def extract_embedding(input_img, model):
    # extraction des features de l'input
    embedding = model(input_img.to(device))
    return embedding


if __name__ == "__main__":
    transform, normalize, inv_normalize = transform_MLP_dataset()

    dataset = ImageAndPathsDataset("MLP-20M", transform)
    dataloader = DataLoader(dataset, batch_size=128, num_workers=2, shuffle=False)

    model_reco = load_model_reco()
    extract_dataset_features(dataloader, model_reco)

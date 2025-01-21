from flask import Flask, request, jsonify
from model import load_model, load_model_reco
from inference import predict_genre
from dataset import Movie_Dataset
from recommendation import search
from extract_features import extract_embedding
from annoy import AnnoyIndex
import pandas as pd
from PIL import Image
from io import BytesIO
from torchvision import transforms
import torch

from model import load_model
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from dataset import ImageAndPathsDataset, transform_MLP_dataset

model_reco = load_model_reco()

def plot_image(path):
  img = mpimg.imread(path)
  plt.imshow(img)
  plt.axis('off')

def plot_images(paths_list):
  plt.figure(figsize=(20,20))
  n = len(paths_list)
  for i, path in enumerate(paths_list):
    plt.subplot(1, n+1, i+1)
    plot_image(path)


def predict_reco(image):
    # Appeler la prédiction
    query_vector = extract_embedding(image, model_reco).detach().cpu().numpy()
    print(query_vector.shape)
    dim = 576 #taille des features 
    annoy_index = AnnoyIndex(dim, 'angular')
    annoy_index.load('rec_movies.ann')
    df = pd.read_csv('features_paths.csv')
    paths_list = df['paths'].tolist()
    recos = search(query_vector[0], annoy_index, paths_list, k=5)
    return recos

transform, normalize, inv_normalize = transform_MLP_dataset()
image = Image.open('MLP-20M/MLP-20M/1.jpg')
tensor = transform(image).unsqueeze(0)
result = predict_reco(tensor)
plot_images(result)
plt.show()


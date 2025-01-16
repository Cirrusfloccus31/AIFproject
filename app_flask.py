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


import torch
import gdown
from torchvision import transforms
from flask import Flask, request, jsonify
from model import load_model
from PIL import Image
from io import BytesIO
from dataset import transform_MLP_dataset

# Les mêmes prétraitements que dans dataset
preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

transform, normalize, inv_normalize = transform_MLP_dataset()

app = Flask("Movie_genre")

# Download model weights
share_url = (
    "https://drive.google.com/file/d/1otYecurXLj7WqWjHjkE4wjOemNhKVDiU/view?usp=sharing"
)
gdown.download(share_url, "movie_net.pth", fuzzy=True, quiet=True)

# Charger le modèle
model = load_model()
model_reco = load_model_reco()
model.load_state_dict(
    torch.load("movie_net.pth", weights_only=True, map_location=torch.device("cpu"))
)

# Charger les genres à partir du dataset
genres = [
    "action",
    "animation",
    "comedy",
    "documentary",
    "drama",
    "fantasy",
    "horror",
    "romance",
    "science Fiction",
    "thriller",
]  # Liste des genres disponibles


@app.route("/predict", methods=["POST"])
def predict():
    image = request.data

    try:
        img_pil = Image.open(BytesIO(image))
        # Prétraiter l'image
        input_tensor = preprocess(img_pil).unsqueeze(0)  # Ajouter une dimension batch

        # Passer l'image dans le modèle
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)

        predicted_index = output.argmax(
            dim=1
        ).item()  # obtient l'index du genre avec la proba la plus haute
        predicted_genre = genres[predicted_index]
        print("ok")
        return jsonify({"predicted_genre": predicted_genre}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/predict_reco', methods=['POST'])
def predict_reco():
    image = request.data

    try:
        img_pil = Image.open(BytesIO(image))
        tensor = transform(img_pil).unsqueeze(0)
        # Appeler la prédiction
        query_vector = extract_embedding(tensor, model_reco).detach().cpu().numpy()
        print(query_vector.shape)
        dim = 576 #taille des features 
        annoy_index = AnnoyIndex(dim, 'angular')
        annoy_index.load('rec_movies.ann')
        df = pd.read_csv('features_paths.csv')
        paths_list = df['paths'].tolist()
        recos = search(query_vector[0], annoy_index, paths_list, k=5)
        recos_imgs=[]
        for i, path in enumerate(recos):
            recos_imgs.append(Image.open(path))
        return jsonify({'recommended_movies': recos_imgs}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
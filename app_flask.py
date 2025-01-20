import torch
import pickle
import pandas as pd
from torchvision import transforms
from flask import Flask, request, jsonify
from model import load_model
from PIL import Image
from io import BytesIO
from annoy import AnnoyIndex
from glove import load_glove_model, find_similar_movies_glove
from bow import find_similar_movies_bow
from settings import (
    MOVIE_NET_PATH,
    TFIDF_VECTORIZER_PATH,
    MOVIES_METADATA_BOW_PATH,
    ANNOY_BOW_PATH,
    GLOVE_FILE_PATH,
    MOVIES_METADATA_GLOVE_PATH,
    ANNOY_GLOVE_PATH,
)

# Les mêmes prétraitements que dans dataset
preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

app = Flask("Movie_genre")

# Charger le modèle
model = load_model()
model.load_state_dict(
    torch.load(MOVIE_NET_PATH, weights_only=True, map_location=torch.device("cpu"))
)

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

# Load model, data and annoy index for glove
glove_model = load_glove_model(GLOVE_FILE_PATH)
embedding_dim_glove = 300
annoy_index_glove = AnnoyIndex(embedding_dim_glove, "angular")
annoy_index_glove.load(ANNOY_GLOVE_PATH)
movies_metadata_glove = pd.read_csv(MOVIES_METADATA_GLOVE_PATH)

# Load model, data and annoy index for bow
embedding_dim_bow = 1000
with open(TFIDF_VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)
annoy_index_bow = AnnoyIndex(embedding_dim_bow, "angular")
annoy_index_bow.load(ANNOY_BOW_PATH)
movies_metadata_bow = pd.read_csv(MOVIES_METADATA_BOW_PATH)


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


@app.route("/reco_overview", methods=["POST"])
def reco_overview():

    data = request.json
    new_overview = data.get("plot")
    method = data.get("method")

    try:
        if method == "glove":
            similar_movies = find_similar_movies_glove(
                new_overview,
                movies_metadata_glove,
                glove_model,
                annoy_index_glove,
                embedding_dim_glove,
            )
        elif method == "bow":
            similar_movies = find_similar_movies_bow(
                new_overview, movies_metadata_bow, vectorizer, annoy_index_bow
            )
        else:
            raise NotImplementedError(f"method must be 'glove' or 'bow' not {method}")

        return jsonify({"Most similar movies": similar_movies}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

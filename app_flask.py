import torch
import pickle
import pandas as pd
import base64
from torchvision import transforms
from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
from annoy import AnnoyIndex
from model import load_model, load_model_reco
from glove import load_glove_model, find_similar_movies_glove
from bow import find_similar_movies_bow
from extract_features import extract_embedding
from recommendation import search
from dataset import transform_MLP_dataset
from settings import (
    MOVIE_NET_PATH,
    TFIDF_VECTORIZER_PATH,
    MOVIES_METADATA_BOW_PATH,
    ANNOY_BOW_PATH,
    GLOVE_FILE_PATH,
    MOVIES_METADATA_GLOVE_PATH,
    ANNOY_GLOVE_PATH,
    EMBEDDING_DIM_GLOVE,
    EMBEDDING_DIM_BOW,
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
transform, normalize, inv_normalize = transform_MLP_dataset()
app = Flask("Movie_genre")

# Charger le modèle
model = load_model()
model.load_state_dict(
    torch.load(MOVIE_NET_PATH, weights_only=True, map_location=torch.device("cpu"))
)
model_reco = load_model_reco()
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
annoy_index_glove = AnnoyIndex(EMBEDDING_DIM_GLOVE, "angular")
annoy_index_glove.load(ANNOY_GLOVE_PATH)
movies_metadata_glove = pd.read_csv(MOVIES_METADATA_GLOVE_PATH)

# Load model, data and annoy index for bow
with open(TFIDF_VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)
annoy_index_bow = AnnoyIndex(EMBEDDING_DIM_BOW, "angular")
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
                EMBEDDING_DIM_GLOVE,
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
        print(recos)
        for i, path in enumerate(recos):
            image=Image.open(path)
            image_binary = BytesIO()
            image.save(image_binary, format="JPEG")
            recos_imgs.append(base64.b64encode(image_binary.getvalue()).decode('utf-8'))
        return jsonify({"images": recos_imgs}), 200
    

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

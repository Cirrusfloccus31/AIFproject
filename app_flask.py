from flask import Flask, request, jsonify
from model import load_model, load_model_reco
from inference import predict_genre
from dataset import Movie_Dataset
from recommendation import search
from extract_features import extract_embedding
from annoy import AnnoyIndex
import pandas as pd



app = Flask('Movie_genre')

# Charger le modèle pour la prédiction de genres
model_genre = load_model()
model_reco = load_model_reco()
# Charger les genres à partir du dataset
dataset_genres = Movie_Dataset(split="all")
genres = dataset_genres.genres  # Liste des genres disponibles



@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Sauvegarder temporairement le fichier pour prédiction
        file_path = f"/tmp/{file.filename}"
        file.save(file_path)
        
        # Appeler la prédiction
        genre = predict_genre(model_genre, file_path, genres)
        return jsonify({'predicted_genre': genre}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route('/predict_reco', methods=['POST'])
def predict_reco():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Sauvegarder temporairement le fichier pour prédiction
        file_path = f"/tmp/{file.filename}"
        file.save(file_path)
        
        # Appeler la prédiction
        query_vector = extract_embedding(file, model_reco)
        dim = 576 #taille des features 
        annoy_index = AnnoyIndex(dim, 'angular')
        annoy_index.load('rec_movies.ann')
        df = pd.read_csv('features_paths.csv')
        paths_list = df['paths'].tolist()
        recos = search(query_vector, annoy_index, paths_list, k=5)
        return jsonify({'recommended_movies': recos}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
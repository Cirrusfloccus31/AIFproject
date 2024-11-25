from flask import Flask, request, jsonify
from model import load_model
from inference import predict_genre
from dataset import Movie_Dataset

app = Flask('Movie_genre')

# Charger le modèle
model = load_model()

# Charger les genres à partir du dataset
dataset = Movie_Dataset(split="all")
genres = dataset.genres  # Liste des genres disponibles

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
        genre = predict_genre(model, file_path, genres)
        return jsonify({'predicted_genre': genre}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
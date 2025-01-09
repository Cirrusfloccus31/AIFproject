import torch
import gdown
from torchvision import transforms
from flask import Flask, request, jsonify
from model import load_model
from dataset import Movie_Dataset
from PIL import Image
from io import BytesIO

# Les mêmes prétraitements que dans dataset
preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

app = Flask('Movie_genre')

#Download model weights
share_url = "https://drive.google.com/file/d/1otYecurXLj7WqWjHjkE4wjOemNhKVDiU/view?usp=sharing"
gdown.download(share_url, "movie_net.pth", fuzzy=True, quiet=True)

# Charger le modèle
model = load_model()
model.load_state_dict(torch.load("movie_net.pth", weights_only=True, map_location=torch.device('cpu')))

# Charger les genres à partir du dataset
dataset = Movie_Dataset(split="all")
genres = dataset.genres  # Liste des genres disponibles

@app.route('/predict', methods=['POST'])
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
        
        predicted_index = output.argmax(dim=1).item() #obtient l'index du genre avec la proba la plus haute 
        predicted_genre = genres[predicted_index]
        print("ok")
        return jsonify({'predicted_genre': predicted_genre}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
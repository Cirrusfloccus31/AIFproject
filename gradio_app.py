import gradio as gr 
import requests
from PIL import Image
from io import BytesIO

def predict_genre_via_api(image):
    url = "http://127.0.0.1:5000/predict"  # Modifié pour docker-compose
    image_binary = BytesIO()
    image.save(image_binary, format="JPEG")
    response = requests.post(url, data=image_binary.getvalue())
    return response.json().get("predicted_genre", response)
    


# Création de l'interface Gradio 
    
Interface = gr.Interface(
    fn = predict_genre_via_api,
    inputs = gr.Image(type = 'pil'),
    outputs = "text",
    title = 'Movie genre predictor',
    description = 'Predict a movie genre based on its poster' 
)

# Lancer l'interface 

if __name__ == "__main__":
    Interface.launch()

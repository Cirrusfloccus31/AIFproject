import gradio as gr 
import requests

API_url = "http://localhost:5000/predict"

def predict_genre_via_api(image):
    url = "http://model_api:5000/predict"  # Modifié pour docker-compose
    files = {'file': image}
    response = requests.post(url, files=files)
    return response.json().get("predicted_genre", "Error in prediction")
    


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

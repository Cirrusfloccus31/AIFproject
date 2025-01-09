import gradio as gr 
import requests

API_url = "http://localhost:5000/predict"

def predict_genre_via_api(image):
    url = "http://model_api:5000/predict"  # Modifié pour docker-compose
    files = {'file': image}
    response = requests.post(url, files=files)
    return response.json().get("predicted_genre", "Error in prediction")
    
def predict_recos_via_api(image):
    url = "http://model_api:5000/predict_reco"  # Modifié pour docker-compose
    files = {'file': image}
    response = requests.post(url, files=files)
    return response.json().get("recommended_movies", "Error in prediction")

# Création de l'interface Gradio 

with gr.Blocks() as interface: 
    with gr.Tab("Interface_genres"): 
        gr.Interface(
            fn = predict_genre_via_api,
            inputs = gr.Image(type = 'pil'),
            outputs = "text",
            title = 'Movie genre predictor',
            description = 'Predict a movie genre based on its poster' 
        )
    with gr.Tab("Interface_recos"): 
        gr.Interface(
            fn = predict_recos_via_api,
            inputs = gr.Image(type = 'pil'),
            outputs = ["image" for i in range(5)],
            title = 'Movie recommendations predictor',
            description = 'Recommend the 5 movies most similar to ours' 
        )

# Lancer l'interface 

if __name__ == "__main__":
    interface.launch()

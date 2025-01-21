import gradio as gr
import requests
from io import BytesIO
import base64
from PIL import Image

api_url = "http://api:5000"

def predict_genre_via_api(image):
    url = f"{api_url}/predict"
    image_binary = BytesIO()
    image.save(image_binary, format="JPEG")
    response = requests.post(url, data=image_binary.getvalue())
    return response.json().get("predicted_genre", response)

def predict_recos_via_api(image):
    url = f"{api_url}/predict_reco" 
    image_binary = BytesIO()
    image.save(image_binary, format="JPEG")
    response = requests.post(url, data=image_binary.getvalue())
    response_json = response.json()
    images = [Image.open(BytesIO(base64.b64decode(img))) for img in response_json]
    return images[0], images[1], images[2], images[3], images[4]

def predict_reco_plot_via_api(plot, method):
    url = f"{api_url}/reco_overview"
    method = "bow" if method == "Bag of words" else "glove"
    print(method)
    data = {"plot": plot, "method": method}
    response = requests.post(url, json=data)
    return response.json().get("Most similar movies", response)


# Création de l'interface Gradio

with gr.Blocks(title="AIF Project") as interface:
    with gr.Row():
        gr.Markdown("<h1 style='text-align: center; font-size: 36px;'>AIF Project</h1>")
    with gr.Tab("Movie genre predictor"):
        gr.Interface(
            fn=predict_genre_via_api,
            inputs=gr.Image(type="pil"),
            outputs=gr.Textbox(label="Predicted genre:"),
            title="Movie genre predictor",
            description="Predict a movie genre based on its poster",
        )
        
    with gr.Tab("Interface_recos"): 
        gr.Interface(
            fn = predict_recos_via_api,
            inputs = gr.Image(type = 'pil'),
            outputs = [gr.Image(type = 'pil') for i in range(5)],
            title = 'Movie recommendations predictor',
            description = 'Recommend the 5 movies most similar to ours' 
        )

    with gr.Tab("Movie recommender based on plot"):
        gr.Interface(
            fn=predict_reco_plot_via_api,
            inputs=[
                gr.Textbox(label="Write the plot here:"),
                gr.Dropdown(
                    choices=["Bag of words", "GloVe"],
                    label="Choose the word vectorization method:",
                ),
            ],
            outputs=gr.Textbox(label="Recommended movies:"),
            title="Movie recommender based on plot",
            description="Recommend 5 movies based on the given plot",
        )


# Lancer l'interface

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)

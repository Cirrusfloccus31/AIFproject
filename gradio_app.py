import gradio as gr
import requests
from io import BytesIO


def predict_genre_via_api(image):
    url = "http://model_api:5000/predict"  # Modifié pour docker-compose
    image_binary = BytesIO()
    image.save(image_binary, format="JPEG")
    response = requests.post(url, data=image_binary.getvalue())
    return response.json().get("predicted_genre", response)


def predict_reco_plot_via_api(plot, method):
    url = "http://model_api:5000/reco_overview"  # Modifié pour docker-compose
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

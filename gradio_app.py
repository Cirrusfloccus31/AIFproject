import gradio as gr
import requests
import base64
from io import BytesIO
from PIL import Image
from settings import NUMBER_RECO_POSTER

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
    response_json = response.json().get("images", [])
    images = [Image.open(BytesIO(base64.b64decode(img))) for img in response_json]
    return (*images,)


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
            inputs=gr.Image(
                type="pil", label="Poster of the movie whose genre you want to predict"
            ),
            outputs=gr.Textbox(label="Predicted genre:"),
            title="Movie genre predictor",
            description="Predict a movie genre based on its poster.",
        )

    with gr.Tab("Movie recommender based on posters"):
        gr.Interface(
            fn=predict_recos_via_api,
            inputs=gr.Image(
                type="pil",
                label="Poster of the movie from which we want recommendations",
            ),
            outputs=[
                gr.Image(label=f"Recommendation {i+1}")
                for i in range(NUMBER_RECO_POSTER)
            ],
            title="Movie recommender based on posters",
            description="Recommend the 5 movies based on the posters.",
        )

    with gr.Tab("Movie recommender based on plots"):
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
            title="Movie recommender based on plots",
            description="Recommend 5 movies based on the plots.",
        )


# Lancer l'interface

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)

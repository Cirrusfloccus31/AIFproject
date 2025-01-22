from annoy import AnnoyIndex
import pandas as pd
import json


def create_database(features_list, dim=576):
    dim = 576  # taille des features
    annoy_index = AnnoyIndex(
        dim, "angular"
    )  # on choisit une distance cosinus comme métrique

    # On crée les vecteurs correspondant à nos features
    for i, embedding in enumerate(features_list):
        annoy_index.add_item(i, embedding)

    annoy_index.build(10)
    annoy_index.save("rec_movies.ann")


if __name__ == "__main__":
    df = pd.read_csv("features_paths.csv")
    df["features"] = df["features"].apply(json.loads)
    features_list = df["features"].tolist()
    create_database(features_list, dim=576)

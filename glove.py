import pandas as pd
import numpy as np
import re
import os
import nltk
from nltk.corpus import stopwords
from annoy import AnnoyIndex
from gensim.models import KeyedVectors
from settings import (
    MOVIES_METADATA_PATH,
    GLOVE_FILE_PATH,
    MOVIES_METADATA_GLOVE_PATH,
    ANNOY_GLOVE_PATH,
    EMBEDDING_DIM_GLOVE,
    NUMBER_RECO_PLOT,
)

# Télécharger les stop words
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))


def load_glove_model(glove_file_path):
    return KeyedVectors.load_word2vec_format(
        glove_file_path, binary=False, no_header=True
    )


def delete_other_glove_files(glove_file_path, embedding_dim_glove):
    dim_list = [50, 100, 200, 300]
    glove_files_directory = os.path.dirname(glove_file_path)
    for dim in dim_list:
        if dim != embedding_dim_glove:
            os.remove(f"{glove_files_directory}/glove.6B.{dim}d.txt")


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)  # Supprimer la ponctuation
    words = text.split()  # Tokenisation
    words = [
        word for word in words if word not in stop_words
    ]  # Supprimer les stop words
    return words


def load_and_process_df():
    df = pd.read_csv(MOVIES_METADATA_PATH)
    df.dropna(subset=["title"], inplace=True)
    df["overview"] = df["overview"].fillna("")
    df["embedding"] = df["overview"].apply(
        lambda x: get_average_embedding(x, glove_model, EMBEDDING_DIM_GLOVE)
    )
    df = df[["title", "embedding"]]
    df.to_csv(MOVIES_METADATA_GLOVE_PATH)
    return df


def get_average_embedding(text, model, embedding_dim):
    words = preprocess_text(text)
    valid_embeddings = [model[word] for word in words if word in model]
    if not valid_embeddings:
        return np.zeros(
            embedding_dim
        )  # Si aucun mot n'a d'embedding, retourner un vecteur nul
    return np.mean(valid_embeddings, axis=0)


def build_annoy_index(df, embedding_dim, num_trees=10):
    index = AnnoyIndex(embedding_dim, "angular")
    for i, embedding in enumerate(df["embedding"]):
        index.add_item(i, embedding)
    index.build(num_trees)
    index.save(ANNOY_GLOVE_PATH)


def find_similar_movies_glove(
    new_overview, df, model, annoy_index, embedding_dim, top_n=NUMBER_RECO_PLOT
):
    new_embedding = get_average_embedding(new_overview, model, embedding_dim)
    similar_indices = annoy_index.get_nns_by_vector(new_embedding, top_n)
    similar_movies = ", ".join(df.iloc[i]["title"] for i in similar_indices)
    return similar_movies


if __name__ == "__main__":
    glove_model = load_glove_model(GLOVE_FILE_PATH)
    # Deletion of unused glove files
    delete_other_glove_files(GLOVE_FILE_PATH, EMBEDDING_DIM_GLOVE)
    df = load_and_process_df()
    annoy_index = build_annoy_index(df, EMBEDDING_DIM_GLOVE)

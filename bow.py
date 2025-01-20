import pickle
import pandas as pd
import re
from annoy import AnnoyIndex
from sklearn.feature_extraction.text import TfidfVectorizer
from settings import (
    MOVIES_METADATA_PATH,
    TFIDF_VECTORIZER_PATH,
    MOVIES_METADATA_BOW_PATH,
    ANNOY_BOW_PATH,
)


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)  # Supprimer la ponctuation
    return text


def load_and_preprocess_df():
    df = pd.read_csv(MOVIES_METADATA_PATH)
    df.dropna(subset=["title"], inplace=True)
    df["id"] = pd.to_numeric(df["id"])
    df["overview"] = df["overview"].fillna("")
    df["overview"] = df["overview"].apply(preprocess_text)
    df = df[["title", "overview"]]
    return df


def build_vectorizer(df, embedding_dim):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=embedding_dim)
    vect_overview = vectorizer.fit_transform(df["overview"])
    with open(TFIDF_VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    vect_overview_list = vect_overview.toarray().tolist()
    df["vect_overview"] = vect_overview_list
    df.to_csv(MOVIES_METADATA_BOW_PATH)
    return vect_overview_list


def build_annoy_index(vect_overview_list, embedding_dim):
    annoy_index = AnnoyIndex(embedding_dim, "angular")
    for i, embedding in enumerate(vect_overview_list):
        annoy_index.add_item(i, embedding)

    annoy_index.build(10)
    annoy_index.save(ANNOY_BOW_PATH)


def find_similar_movies_bow(new_overview, df, vectorizer, annoy_index, top_n=5):

    new_embedding = vectorizer.transform([new_overview]).toarray().tolist()[0]
    similar_indices = annoy_index.get_nns_by_vector(new_embedding, top_n)

    similar_movies = ", ".join(df.iloc[i]["title"] for i in similar_indices)
    return similar_movies


if __name__ == "__main__":
    embedding_dim = 1000
    df = load_and_preprocess_df()
    vect_overview_list = build_vectorizer(df, embedding_dim)
    build_annoy_index(vect_overview_list, embedding_dim)

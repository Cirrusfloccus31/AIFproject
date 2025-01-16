import pandas as pd
import numpy as np
from annoy import AnnoyIndex
from sklearn.feature_extraction.text import TfidfVectorizer

def reco_overview(input_overview):
    input_metadata = pd.DataFrame(
        {
            "id": [0],
            "title": ["__input_title__"],
            "overview": [input_overview]
        }
    )

    size = 400

    vectorizer = TfidfVectorizer(stop_words='english', max_features=size)

    metadata = pd.read_csv('data/movies_metadata.csv')
    metadata.dropna(subset=['title'], inplace=True)
    metadata['id'] = pd.to_numeric(metadata['id'])
    metadata['overview'] = metadata['overview'].fillna('')
    metadata = metadata[['id', 'title', 'overview']]
    metadata = pd.concat([metadata, input_metadata], ignore_index=True)

    vect_overview = vectorizer.fit_transform(metadata['overview'])

    vect_overview_list = vect_overview.toarray().tolist()

    metadata['vect_overview'] = vect_overview_list

    metadata = metadata[['id', 'title', 'overview', 'vect_overview']]

    query_vector = metadata[metadata["title"]=="__input_title__"]["vect_overview"].tolist()[0]

    vect_overview_list = metadata[metadata["title"]!="__input_title__"]['vect_overview'].tolist() #Removing the input from the embeddings

    annoy_index = AnnoyIndex(size, 'angular')
    for i, embedding in enumerate(vect_overview_list):
        annoy_index.add_item(metadata.at[i, 'title'], embedding)

    annoy_index.build(10)
    annoy_index.save('data/rec_overview.ann')

    nearest_neighbors, distances = annoy_index.get_nns_by_vector(query_vector, n=5, include_distances=True)

    # Afficher les résultats
    for title, distance in zip(nearest_neighbors, distances):
        print(f"Title: {title}, Distance: {distance}")

if __name__ == "__main__":
    reco_overview("Led by Woody, Andy's toys live happily in his room until Andy's birthday brings Buzz Lightyear onto the scene. Afraid of losing his place in Andy's heart, Woody plots against Buzz. But when circumstances separate Buzz and Woody from their owner, the duo eventually learns to put aside their differences.")
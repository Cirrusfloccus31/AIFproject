import pandas as pd
import numpy as np
import re
from annoy import AnnoyIndex
from gensim.models import KeyedVectors

# Prétraiter le texte
def preprocess_text(text):
    text = text.lower()  # Convertir en minuscule
    text = re.sub(r'[^\w\s]', '', text)  # Supprimer la ponctuation
    return text.split()  # Tokeniser

# Calculer l'embedding moyen pour un texte
def get_average_embedding(text, model, embedding_dim=300):
    words = preprocess_text(text)
    valid_embeddings = [model[word] for word in words if word in model]
    if not valid_embeddings:
        return np.zeros(embedding_dim)  # Si aucun mot n'a d'embedding, retourner un vecteur nul
    return np.mean(valid_embeddings, axis=0)

def build_annoy_index(df, embedding_dim, num_trees=10):
    index = AnnoyIndex(embedding_dim, 'angular')  # Utilisation de la distance angulaire (proche de la cosinus)
    for i, embedding in enumerate(df['embedding']):
        index.add_item(i, embedding)  # Ajouter chaque vecteur avec son index
    index.build(num_trees)  # Construire l'index avec le nombre d'arbres spécifié
    return index

def find_similar_movies(new_overview, df, model, annoy_index, embedding_dim, top_n=5):
    # Calculer l'embedding pour le nouvel overview
    new_embedding = get_average_embedding(new_overview, model, embedding_dim)
    
    # Trouver les indices des films les plus proches
    similar_indices = annoy_index.get_nns_by_vector(new_embedding, top_n, include_distances=True)
    
    # Récupérer les titres et scores des films correspondants
    similar_movies = [(df.iloc[i]['title'], 1 - dist) for i, dist in zip(similar_indices[0], similar_indices[1])]
    return pd.DataFrame(similar_movies, columns=['title', 'similarity'])

# Chemin vers le fichier GloVe
glove_file_path = 'data/glove.6B.100d.txt'
embedding_dim = 100

# Charger les embeddings avec Gensim
glove_model = KeyedVectors.load_word2vec_format(glove_file_path, binary=False, no_header=True)

df = pd.read_csv('data/movies_metadata.csv')

df['overview'] = df['overview'].fillna('')

# Calculer les embeddings pour chaque overview
df['embedding'] = df['overview'].apply(lambda x: get_average_embedding(x, glove_model, embedding_dim))

# Construire l'index Annoy
annoy_index = build_annoy_index(df, embedding_dim)

# Recherche des films les plus similaires
new_overview = df.at[0, 'overview']
most_similar_movies = find_similar_movies(new_overview, df, glove_model, annoy_index, embedding_dim, top_n=5)

print(most_similar_movies)

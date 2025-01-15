import pandas as pd
import numpy as np
from annoy import AnnoyIndex
from sklearn.feature_extraction.text import TfidfVectorizer

size = 400

vectorizer = TfidfVectorizer(stop_words='english', max_features=size)

metadata = pd.read_csv('data/movies_metadata.csv')
metadata.dropna(subset=['title'], inplace=True)
metadata['id'] = pd.to_numeric(metadata['id'])
metadata['overview'] = metadata['overview'].fillna('')
metadata = metadata[['id', 'title', 'overview']]

vect_overview = vectorizer.fit_transform(metadata['overview'])

vect_overview_list = vect_overview.toarray().tolist()

metadata['vect_overview'] = vect_overview_list

metadata = metadata[['id', 'title', 'vect_overview']]

metadata.to_csv('data/movies_metadata_bow.csv', index=False)

annoy_index = AnnoyIndex(size, 'angular')
for i, embedding in enumerate(vect_overview_list):
    annoy_index.add_item(i, embedding)

annoy_index.build(10)
annoy_index.save('data/rec_overview.ann')
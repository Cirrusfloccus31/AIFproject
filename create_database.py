from annoy import AnnoyIndex
import numpy as np

dim = 576
annoy_index = AnnoyIndex(dim, 'angular')
features_list = list(np.load("features.npy"))
for i, embedding in enumerate(features_list):
    annoy_index.add_item(i, embedding)

annoy_index.build(10)
annoy_index.save('rec_movies.ann')
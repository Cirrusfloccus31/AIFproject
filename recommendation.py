from settings import NUMBER_RECO_POSTER


# On cherche les vecteurs les plus proches de notre vecteur de features
def search(query_vector, annoy_index, paths_list, k=NUMBER_RECO_POSTER):
    indices = annoy_index.get_nns_by_vector(query_vector, k)
    paths = [paths_list[idx] for idx in indices]
    return paths

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataset import Movie_Dataset
from settings import PLOT_PATH

if __name__ == "__main__":
    data = Movie_Dataset("all")
    genres = data.genres
    count = np.array(
        [np.array(data[idx][1]) for idx in range(len(data))], dtype=int
    ).sum(axis=0)
    genre_count = pd.DataFrame({"count": count}, index=genres).sort_values(
        by="count", ascending=False
    )
    fig = plt.figure()
    bar = plt.bar(range(len(genres)), genre_count["count"].tolist())
    plt.bar_label(bar, labels=genre_count["count"].tolist())
    plt.xticks(range(len(genres)), labels=genre_count.index.to_list(), rotation=90)
    plt.xlabel("Genres")
    plt.ylabel("Number of movies")
    plt.title("Number of movies per genre")
    plt.tight_layout()
    fig.savefig(PLOT_PATH + "hist.png")
    plt.show()

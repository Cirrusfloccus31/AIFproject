mkdir -p data

# Poids de movies_net
wget "https://drive.usercontent.google.com/download?id=1otYecurXLj7WqWjHjkE4wjOemNhKVDiU&export=download&confirm=t&uuid=94cee497-75f7-47af-a562-5d7f41dd2c9c" -O "data/movie_net.pth" > /dev/null 2>&1

# Movies metadata overview
wget --no-check-certificate "https://drive.google.com/uc?export=download&id=1GTm75uv6CeBOBfbGsq6-EQp0gyg8Fzfn" -O "data/movies_metadata.csv" > /dev/null 2>&1

# Glove file
wget "https://nlp.stanford.edu/data/glove.6B.zip" -O "data/glove.6B.zip" > /dev/null 2>&1
cd data
unzip glove.6B.zip > /dev/null 2>&1
rm glove.6B.zip
cd ..

# Download the VGG16 weights into the cache dir
CACHE_DIR="./cache/checkpoints"
mkdir -p "$CACHE_DIR"

# Download the weights of vgg16
WEIGHTS_URL="https://download.pytorch.org/models/vgg16-397923af.pth"
wget "$WEIGHTS_URL" -O "$CACHE_DIR/vgg16-397923af.pth" > /dev/null 2>&1

# Download the weights of mobilenet
WEIGHTS_URL="https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth"
wget "$WEIGHTS_URL" -O "$CACHE_DIR/mobilenet_v3_small-047dcff4.pth" > /dev/null 2>&1

# Download annoy file part 2
wget --no-check-certificate "https://drive.google.com/uc?export=download&id=1xEbpdGCzWlJjv1SGtiTjz0puOF-8J34c" -O "data/rec_movies.ann" > /dev/null 2>&1

# Download csv file part 2
wget "https://drive.usercontent.google.com/download?id=1wMcYiJhrKGAAcKlwrWNbaCTtSoDttXjQ&export=download&confirm=t&uuid=c77b8f52-1f41-4e99-bb80-e8bd6ff44fc9" -O "data/features_paths.csv" > /dev/null 2>&1

# Download dataset MLP-20M
kaggle datasets download "ghrzarea/movielens-20m-posters-for-machine-learning"
unzip movielens-20m-posters-for-machine-learning.zip

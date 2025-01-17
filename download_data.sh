mkdir -p data

# Poids de movies_net
wget "https://drive.usercontent.google.com/download?id=1otYecurXLj7WqWjHjkE4wjOemNhKVDiU&export=download&confirm=t&uuid=94cee497-75f7-47af-a562-5d7f41dd2c9c" -O "data/movie_net.pth" > /dev/null 2>&1

# Movies metadata overview
wget --no-check-certificate "https://drive.google.com/uc?export=download&id=1GTm75uv6CeBOBfbGsq6-EQp0gyg8Fzfn" -O "data/movies_metadata.csv" > /dev/null 2>&1

# Glove file
wget https://nlp.stanford.edu/data/glove.6B.zip -O "data/glove.6B.zip" > /dev/null 2>&1
cd data
unzip glove.6B.zip > /dev/null 2>&1

cd ..

# Download the VGG16 weights into the cache dir
CACHE_DIR="./cache/checkpoints"
mkdir -p "$CACHE_DIR"

# Download the weights
WEIGHTS_URL="https://download.pytorch.org/models/vgg16-397923af.pth"
wget "$WEIGHTS_URL" -O "$CACHE_DIR/vgg16-397923af.pth" > /dev/null 2>&1
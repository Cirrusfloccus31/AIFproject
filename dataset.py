import torch
import random
import PIL
import os
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from settings import DATA_PATH

NUM_CLASSES = 10

seed = random.seed(42)


class Movie_Dataset(Dataset):
    """The Dataset class of the movie posters and their associated genre"""

    def __init__(self, split, data_path=DATA_PATH, training_fraction=0.7):
        self.split = split
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),  # Resize the image to 256x256 pixels
                transforms.CenterCrop(224),  # Crop the center 224x224 pixels
                transforms.ToTensor(),  # Convert the image to a tensor
                # Normalize with ImageNet mean and std
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        # Create list to associate genres and index
        self.genres = [
            genres
            for genres in os.listdir(data_path)
            if os.path.isdir(os.path.join(data_path, genres))
        ]
        self.num_classes = len(self.genres)
        self.paths = list(
            Path(data_path).glob("**/*.jpg")
        )  # List all the paths of the images
        random.shuffle(self.paths)  # shuffle the list
        index_max_train = int(len(self.paths) * training_fraction)
        index_max_validation = (index_max_train + len(self.paths)) // 2
        if self.split == "train":
            self.data = self.paths[:index_max_train]
        elif self.split == "validation":
            self.data = self.paths[index_max_train:index_max_validation]
        elif self.split == "test":
            self.data = self.paths[index_max_validation:]
        elif self.split == "all":
            self.data = self.paths
        else:
            raise Exception("Invalid split choose 'train', 'validation' or 'test'")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = PIL.Image.open(self.data[idx])
        x = self.transform(img)
        genre_str = self.data[idx].parts[-2]  # extract the genre of the movie
        # Convert genre to one_hot encoding tensors
        y = torch.nn.functional.one_hot(
            torch.tensor(self.genres.index(genre_str)), num_classes=self.num_classes
        ).to(torch.float)
        return x, y


def get_dataloaders(batch_size=1, num_workers=1):
    loaders = {}
    splits = ["train", "validation", "test"]
    for split in splits:
        dataset = Movie_Dataset(split, "MovieGenre", 0.7)
        batch_size_data = 1 if split in ["validation", "test"] else batch_size
        shuffle = split == "train"
        loader = DataLoader(
            dataset, batch_size_data, num_workers=num_workers, shuffle=shuffle
        )
        loaders[split] = loader
    return loaders

import torch
import torchvision
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import random
import PIL 
import numpy as np
import os 

NUM_CLASSES = 10

seed = random.seed(42)



class Movie_Dataset(Dataset):
    def __init__(self,split,path,training_fraction):
        self.split=split
        self.transform = transform = transforms.Compose([
            transforms.Resize(256),  # Resize the image to 256x256 pixels
            transforms.CenterCrop(224),  # Crop the center 224x224 pixels
            transforms.ToTensor(),  # Convert the image to a tensor
            # Normalize with ImageNet mean and std
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        ])
        self.training_fraction = training_fraction
        self.genres = [genres for genres in os.listdir(path) if os.path.isdir(os.path.join(path, genres))]
        self.paths = list(Path(path).glob("**/*.jpg"))
        random.shuffle(self.paths)
        self.num_samples=int(len(self.paths)*self.training_fraction)
        if self.split == "train":
            self.data = self.paths[:self.num_samples]
        if self.split == "test":
            self.data = self.paths[self.num_samples:]

    def __len__(self):
        return len(self.data)
    
        

    def __getitem__(self, idx):
        img = PIL.Image.open(self.data[idx])
        x = self.transform(img)
        genre_str = self.data[idx].parts[-2]
        y = torch.nn.functional.one_hot(torch.tensor(self.genres.index(genre_str)), num_classes=NUM_CLASSES).to(torch.float)
        return x, y

def get_dataloaders(splits, batch_size, num_workers):
    loaders = {}
    for split in splits:
        dataset = Movie_Dataset(split,"MovieGenre",0.7)
        batch_size_data = 1 if split == "test" else batch_size
        shuffle = split == "train"
        loader = DataLoader(
            dataset, batch_size_data, num_workers=num_workers, shuffle=shuffle
        )
        loaders[split] = loader
    return loaders

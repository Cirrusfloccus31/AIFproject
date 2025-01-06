import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
import torchvision.models as models
import pandas as pd
from tqdm.notebook import tqdm
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets



class ImageAndPathsDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        img, _= super(ImageAndPathsDataset, self).__getitem__(index)
        path = self.imgs[index][0]
        return img, path


mean = [ 0.485, 0.456, 0.406 ]
std = [ 0.229, 0.224, 0.225 ]
normalize = transforms.Normalize(mean, std)
inv_normalize = transforms.Normalize(
   mean= [-m/s for m, s in zip(mean, std)],
   std= [1/s for s in std]
)

transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                normalize])

dataset = ImageAndPathsDataset('MLP-20M', transform)
dataloader = DataLoader(dataset, batch_size=128, num_workers=2, shuffle=False)

    

mobilenet = models.mobilenet_v3_small(pretrained=True)
model = torch.nn.Sequential(mobilenet.features, mobilenet.avgpool, torch.nn.Flatten()).cuda()
model = model.eval()


features_list = []

for x, _ in tqdm(dataloader):
    with torch.no_grad():
        embeddings = model(x.cuda())
        features_list.extend(embeddings.cpu().numpy())
        
features = np.vstack(features_list)
np.save("features.npy", features)

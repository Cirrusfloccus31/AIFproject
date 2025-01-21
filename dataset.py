from torchvision import transforms
from torchvision import datasets


class ImageAndPathsDataset(datasets.ImageFolder):
    
    def __getitem__(self, index):
        img, _= super(ImageAndPathsDataset, self).__getitem__(index)
        path = self.imgs[index][0]
        return img, path
    
def transform_MLP_dataset():
    '''Normalisation des posters pour les faire passer dans mobilenet v3'''
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
    return transform, normalize, inv_normalize

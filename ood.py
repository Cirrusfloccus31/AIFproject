import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import trained_model_logits, load_model
from part1.dataset import get_dataloaders, Movie_Dataset
from detect_anomalies import mls, compute_threshold
from settings import MOVIE_NET_PATH
from tqdm import tqdm
genres = [
    "action",
    "animation",
    "comedy",
    "documentary",
    "drama",
    "fantasy",
    "horror",
    "romance",
    "science Fiction",
    "thriller",
]  # Liste des genres disponibles


# Les mêmes prétraitements que dans dataset
preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

def compute_logits(dataset, model, device):
    all_logits = []
    model.to(device)
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            input, _ = dataset[i]  
            if type(input)!=torch.Tensor: 
                input = preprocess(input)
            input = input.unsqueeze(0)
            input = input.to(device) 
            logits = model(input)
            all_logits.append(logits)
    return torch.cat(all_logits, dim=0) 

def predict(input, score, threshold, model_logits, model_genre):
    # Prétraiter l'input
    if type(input)!=torch.Tensor: 
        input = preprocess(input)
    input = input.unsqueeze(0)# Ajouter une dimension batch
    input = input.to(device) 
    
    model_logits.eval()
    with torch.no_grad():
        logits = model_logits(input)
        s = score(logits)
        
    if s > threshold: 
        return "This input is not valid. Please enter a movie poster."
    else: 
        model_genre.eval()
        with torch.no_grad():
            output = model_genre(input)
        predicted_index = output.argmax(
            dim=1
        ).item()  # obtient l'index du genre avec la proba la plus haute
        predicted_genre = genres[predicted_index]

    return "predicted_genre", predicted_genre 


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    batch_size = 128
  
    movie_test = Movie_Dataset("test", "MovieGenre", 0.7) #normal
    svhn_test = datasets.SVHN(root='./data', split='test', transform=None, download=True) #anomalies
    # Extract 10_000 random images from the svhn_test set
    svhn_test, _ = torch.utils.data.random_split(svhn_test, [10_000, len(svhn_test) - 10_000])
    
    # dataloaders
    loaders = get_dataloaders(batch_size=batch_size) #MovieGenre
    test_loader = loaders["test"]
    svhn_test_loader = DataLoader(svhn_test, batch_size=batch_size, shuffle=False)
    
    # Classifier entraîné de la partie 1
    weights_model = "movie_net.pth"
    model_without_softmax = trained_model_logits(weights_model)
    model_without_softmax.eval()

    #test_logits_positives = compute_logits(svhn_test, model_without_softmax, device)
    test_logits_positives = torch.load("test_logits_positives.pt", map_location=torch.device("cpu"), weights_only=True)
    
    target_tpr = 0.95
    scores_positives = mls(test_logits_positives)
    threshold = compute_threshold(scores_positives, target_tpr)
    
    model = load_model()
    model.load_state_dict(
    torch.load(weights_model, weights_only=True, map_location=torch.device("cpu")))  
    
    normal_ex, _ = movie_test[0]
    anomaly_ex, _ = svhn_test[0]
    print(predict(normal_ex, mls, threshold, model_without_softmax, model))
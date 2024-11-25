import torch
from torchvision import transforms
from PIL import Image

# Les mêmes prétraitements que dans dataset
preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def predict_genre(model, image_path, genres):
    """
    Prédire le genre d'un film à partir de son poster.
    
    Args:
    - model: Le modèle PyTorch chargé.
    - image_path: Chemin vers l'image.
    - genres: Liste des genres associée au dataset.
    
    Returns:
    - Genre prédit (chaîne de caractères).
    """
    image = Image.open(image_path).convert("RGB")  # Assurer que l'image est en RGB
    # Prétraiter l'image
    input_tensor = preprocess(image).unsqueeze(0)  # Ajouter une dimension batch

    # Passer l'image dans le modèle
    model.eval()  
    with torch.no_grad():
        output = model(input_tensor)
    
    predicted_index = output.argmax(dim=1).item() #obtient l'index du genre avec la proba la plus haute 
    predicted_genre = genres[predicted_index]
    
    return predicted_genre
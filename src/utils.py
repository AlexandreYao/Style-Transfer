import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
# Chargement et transformation des images
def image_loader(image_name, device):
    image = Image.open(image_name).convert("RGB")
    # Mise à l'échelle de l'image et conversion en tenseur
    loader = transforms.Compose([
        transforms.Resize((128, 128)),  # Redimensionner pour la compatibilité
        transforms.ToTensor()
    ])
    image = loader(image).unsqueeze(0)  # Ajouter une dimension batch
    return image.to(device, torch.float)

def imshow(tensor, title):
    image = tensor.cpu().clone()  # Déplacer l'image vers le CPU
    image = image.squeeze(0)  # Supprimer la dimension batch
    image = torch.clamp(image, 0, 1)  # S'assurer que les valeurs sont dans l'intervalle [0, 1]
    image = transforms.ToPILImage()(image)
    plt.figure()
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.show()
import torch
from torchvision import transforms
from PIL import Image
import os

# Load trained model
def load_model(model_path="breast_cancer_cnn.pth"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found")

    model = torch.load(model_path, map_location=torch.device("cpu"))
    model.eval()
    return model


# Preprocess uploaded image
def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

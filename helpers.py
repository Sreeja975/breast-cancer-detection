import torch
from torchvision import transforms
from PIL import Image
import os
from model import BreastCancerCNN


def load_model(model_path):
    """
    Loads model and threshold from checkpoint
    """
    if not os.path.exists(model_path):
        return None, None

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False  # Required for PyTorch >= 2.6
    )

    model = BreastCancerCNN()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    threshold = checkpoint.get("threshold", 0.5)

    return model, threshold


def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return transform(image).unsqueeze(0)

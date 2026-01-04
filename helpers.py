import os
import torch
from torchvision import transforms
from PIL import Image
from model import BreastCancerCNN  # Make sure this matches the training model

def load_model(model_path):
    """
    Loads the trained BreastCancerCNN model and automatically learned threshold.
    Returns:
        model (torch.nn.Module)
        threshold (float)
    """
    if not os.path.exists(model_path):
        return None, None

    # Load checkpoint (trusted source)
    checkpoint = torch.load(
        model_path,
        map_location=torch.device("cpu"),
        weights_only=False  # Required for PyTorch >= 2.6
    )

    # Initialize model architecture EXACTLY as it was trained
    model = BreastCancerCNN()  # This MUST use `self.model` inside class
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Load automatic ROC-based threshold
    threshold = float(checkpoint["threshold"])

    return model, threshold


def preprocess_image(image: Image.Image):
    """
    Preprocesses a PIL image for ResNet inference
    Returns a tensor of shape (1, 3, 224, 224)
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return transform(image).unsqueeze(0)

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from PIL import Image
import os


# --------------------------------------------------
# Model Definition
# --------------------------------------------------
class BreastCancerCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x):
        return self.model(x)


# --------------------------------------------------
# Load trained model
# --------------------------------------------------
def load_model(model_path="breast_cancer_cnn.pth", device="cpu"):
    """
    Loads the trained breast cancer detection model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = BreastCancerCNN()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------
def get_transform():
    """
    Returns the image preprocessing pipeline
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


# --------------------------------------------------
# Predict cancer probability
# --------------------------------------------------
def predict(image: Image.Image, model, threshold=0.65, device="cpu"):
    """
    Predicts cancer probability for a single image

    Returns:
        probability (float)
        prediction_label (str)
    """
    transform = get_transform()
    image = image.convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probability = torch.sigmoid(logits).item()

    label = "Cancer" if probability >= threshold else "Normal"
    return probability, label

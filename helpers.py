import torch
import numpy as np
import cv2
from torchvision import models, transforms
from torch import nn
from PIL import Image

# =====================================================
# DEVICE
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# IMAGE PREPROCESSING (MATCHES RESNET-18 TRAINING)
# =====================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def preprocess_image(image: Image.Image):
    """
    Convert PIL image to model-ready tensor
    """
    image = image.convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)  # batch dimension
    return image.to(device)

# =====================================================
# MODEL LOADING
# =====================================================
def load_model(model_path="binary_breast_cancer_model.pth"):
    """
    Load trained ResNet-18 binary classification model
    """
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model

# =====================================================
# PREDICTION
# =====================================================
def predict(image: Image.Image, model):
    """
    Run inference and return label + confidence
    """
    image_tensor = preprocess_image(image)

    with torch.no_grad():
        output = model(image_tensor)
        prob = torch.sigmoid(output).item()

    label = "Cancer" if prob >= 0.5 else "No Cancer"
    confidence = prob if prob >= 0.5 else 1 - prob

    return label, confidence

# =====================================================
# GRAD-CAM
# =====================================================
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer = model.layer4[-1]
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor):
        """
        Generate Grad-CAM heatmap
        """
        self.model.zero_grad()
        output = self.model(input_tensor)
        output.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)

        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam

# =====================================================
# GRAD-CAM OVERLAY
# =====================================================
def overlay_gradcam(image: Image.Image, cam):
    """
    Overlay Grad-CAM heatmap on original image
    """
    image = image.resize((224, 224))
    image_np = np.array(image)

    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam),
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)
    return overlay

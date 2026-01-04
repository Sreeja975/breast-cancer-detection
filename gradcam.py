import torch
import numpy as np
import cv2
from PIL import Image

class GradCAM:
    """
    Grad-CAM implementation for PyTorch CNN models
    Works with ResNet-based architectures
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        # Forward hook to get activations
        def forward_hook(module, input, output):
            self.activations = output.detach()

        # Backward hook to get gradients
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor):
        """
        Generate Grad-CAM heatmap for a single input tensor
        input_tensor: torch.Tensor (1, 3, H, W)
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)
        score = output.squeeze()  # For single output

        # Backward pass
        self.model.zero_grad()
        score.backward(retain_graph=True)

        # Global average pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        # Weighted sum of activations
        cam = torch.sum(weights * self.activations, dim=1).squeeze()

        # ReLU
        cam = torch.clamp(cam, min=0)

        # Normalize to [0,1]
        cam = cam.cpu().numpy()
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        return cam


def overlay_gradcam(image, heatmap, alpha=0.4):
    """
    Overlay Grad-CAM heatmap on original image

    image: PIL.Image or numpy array (H,W,3)
    heatmap: Grad-CAM output (H,W)
    """
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Ensure uint8
    image = image.astype(np.uint8)

    # Resize heatmap to match image
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    return overlay

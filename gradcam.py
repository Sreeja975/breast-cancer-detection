import torch
import numpy as np
import cv2


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
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor):
        """
        Generate Grad-CAM heatmap
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)
        score = output.squeeze()

        # Backward pass
        self.model.zero_grad()
        score.backward(retain_graph=True)

        # Global average pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        # Weighted sum of activations
        cam = torch.sum(weights * self.activations, dim=1).squeeze()

        # ReLU
        cam = torch.clamp(cam, min=0)

        # Normalize
        cam = cam.cpu().numpy()
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        return cam


def overlay_gradcam(image, heatmap, alpha=0.4):
    """
    Overlay heatmap on original image

    image: PIL or numpy image (H,W,3)
    heatmap: Grad-CAM output
    """
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    return overlay

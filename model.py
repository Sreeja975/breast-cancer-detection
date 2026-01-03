import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class BreastCancerCNN(nn.Module):
    """
    ResNet18-based binary classifier for breast cancer detection
    """
    def __init__(self):
        super().__init__()

        self.backbone = resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        # Replace final fully connected layer
        self.backbone.fc = nn.Linear(
            self.backbone.fc.in_features,
            1
        )

    def forward(self, x):
        return self.backbone(x)

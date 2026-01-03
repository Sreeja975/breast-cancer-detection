import streamlit as st
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from PIL import Image
import numpy as np
import os

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Breast Cancer Detection",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Breast Cancer Detection System")
st.write("Upload a mammogram image to predict cancer risk using a deep learning model.")

# --------------------------------------------------
# Model definition
# --------------------------------------------------
class BreastCancerCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x):
        return self.model(x)

# --------------------------------------------------
# Load model (cached)
# --------------------------------------------------
@st.cache_resource
def load_model():
    model = BreastCancerCNN()

    model_path = "breast_cancer_cnn.pth"
    if not os.path.exists(model_path):
        st.error("❌ Model file not found. Please add 'breast_cancer_cnn.pth' to the repository.")
        st.stop()

    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()

# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------------------------------------
# Threshold selection
# --------------------------------------------------
st.subheader("🔧 Decision Threshold")
threshold = st.slider(
    "Adjust cancer detection sensitivity",
    min_value=0.30,
    max_value=0.90,
    value=0.65,
    step=0.01
)

# --------------------------------------------------
# Image upload
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload a mammogram image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)
        probability = torch.sigmoid(logits).item()

    st.markdown("### 🔍 Prediction Result")
    st.write(f"**Cancer Probability:** `{probability:.3f}`")

    if probability >= threshold:
        st.error("⚠️ **Cancer Detected**")
    else:
        st.success("✅ **Normal**")

    st.caption(
        "⚠️ This tool is for educational and research purposes only. "
        "It should not be used as a substitute for professional medical diagnosis."
    )

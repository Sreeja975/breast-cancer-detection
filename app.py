import streamlit as st
import torch
import numpy as np
from PIL import Image
import cv2

from model import BreastCancerCNN
from helpers import preprocess_image
from gradcam import GradCAM, overlay_gradcam

# -------------------------
# Config
# -------------------------
MODEL_PATH = "breast_cancer_model_with_threshold.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Load Model
# -------------------------
@st.cache_resource
def load_model():
    model = BreastCancerCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# -------------------------
# UI
# -------------------------
st.title("🩺 Breast Cancer Detection")
st.write("Upload a mammogram image to predict cancer")

uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    input_tensor = preprocess_image(image).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        probability = torch.sigmoid(output).item()

    threshold = st.slider("Decision Threshold", 0.1, 0.9, 0.5)

    prediction = "Cancer" if probability > threshold else "Normal"

    st.subheader(f"Prediction: {prediction}")
    st.write(f"Confidence: {probability:.2f}")

    # -------------------------
    # Grad-CAM
    # -------------------------
    if st.checkbox("Show Grad-CAM"):
       target_layer = model.model.layer4[1].conv2
       cam = GradCAM(model, target_layer)

       heatmap = cam.generate(input_tensor)
       overlay = overlay_gradcam(image, heatmap)

       st.image(overlay, caption="Grad-CAM Visualization", use_container_width=True)

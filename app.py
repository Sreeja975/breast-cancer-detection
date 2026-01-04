import streamlit as st
from PIL import Image
import torch
import numpy as np
import cv2

from helpers import load_model, preprocess_image
from gradcam import GradCAM, overlay_gradcam

st.set_page_config(page_title="Breast Cancer Detection")

st.title("Breast Cancer Detection System")

model = load_model("breast_cancer_model_with_threshold.pth")

if model is None:
    st.error("❌ Model file not found")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a mammogram image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = preprocess_image(image)

    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()

    if prob > 0.5:
        st.error(f"Cancer Detected (Confidence: {prob:.2f})")
    else:
        st.success(f"No Cancer Detected (Confidence: {1 - prob:.2f})")

    if st.checkbox("Show Grad-CAM"):
        target_layer = model.backbone.layer4[-1]
        cam = GradCAM(model, target_layer)

        heatmap = cam.generate(input_tensor)

        img_np = np.array(image)
        cam_image = overlay_gradcam(img_np, heatmap)

        st.image(cam_image, caption="Grad-CAM Visualization", use_column_width=True)

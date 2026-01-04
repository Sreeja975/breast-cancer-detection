import streamlit as st
from PIL import Image
import torch
from helpers import load_model, preprocess_image

# Load your model
model = load_model("breast_cancer_model_with_threshold.pth")
model.eval()

st.title("Breast Cancer Detection System")

# Upload image
uploaded_file = st.file_uploader("Upload a mammogram image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    input_tensor = preprocess_image(image)  # Implement in helpers.py

    # Prediction
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()  # If binary classification

    # Display result
    if prob > 0.5:
        st.error(f"Cancer Detected! Probability: {prob:.2f}")
    else:
        st.success(f"No Cancer Detected. Probability: {prob:.2f}")

    # Optional: Grad-CAM visualization
    if st.checkbox("Show Grad-CAM"):
        cam_image = get_gradcam(model, input_tensor)  # Implement in gradcam.py
        st.image(cam_image, caption="Grad-CAM", use_column_width=True)

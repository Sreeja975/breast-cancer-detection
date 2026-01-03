import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

from model import BreastCancerCNN
from gradcam import GradCAM

# ----------------------
# Page config
# ----------------------
st.set_page_config(page_title="Breast Cancer Detection", layout="centered")
st.title("🩺 Breast Cancer Detection App")
st.write("Deep Learning with ResNet18 + Grad-CAM")

# ----------------------
# Load model
# ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BreastCancerCNN().to(device)
model.load_state_dict(torch.load("breast_cancer_cnn.pth", map_location=device))
model.eval()

gradcam = GradCAM(model, model.model.layer4)

# ----------------------
# Image preprocessing
# ----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ----------------------
# Upload image
# ----------------------
uploaded_file = st.file_uploader("Upload a breast image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()

    prediction = "Cancer" if prob >= 0.7 else "Normal"

    st.subheader(f"Prediction: **{prediction}**")
    st.write(f"Confidence: **{prob:.2f}**")

    # Grad-CAM
    cam = gradcam.generate(input_tensor)

    img_np = np.array(image.resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

    st.subheader("Grad-CAM Visualization")
    st.image(overlay, use_container_width=True)
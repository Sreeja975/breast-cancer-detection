import streamlit as st
from PIL import Image
import torch
import numpy as np
import cv2
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os

from model import BreastCancerCNN
from gradcam import GradCAM, overlay_gradcam

# =========================================================
# 1️⃣ Streamlit config
# =========================================================
st.set_page_config(page_title="Breast Cancer Detection")
st.title("Breast Cancer Detection System")

# =========================================================
# 2️⃣ Paths & device
# =========================================================
MODEL_PATH = "model.pth"
VAL_DIR = r"C:\Users\biswa\OneDrive\Documents\cancer-detection-app\archive (19)\val"

# Check folder structure
for folder in ["normal", "cancer"]:
    folder_path = os.path.join(VAL_DIR, folder)
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# 3️⃣ Load model
# =========================================================
@st.cache_resource
def load_model(model_path):
    if not os.path.exists(model_path):
        return None
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = BreastCancerCNN().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model

model = load_model(MODEL_PATH)
if model is None:
    st.error("❌ Model not found.")
    st.stop()

# =========================================================
# 4️⃣ Image preprocessing
# =========================================================
def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])
    return transform(image).unsqueeze(0).to(device)

# =========================================================
# 5️⃣ Validation dataset
# =========================================================
class ValDataset(Dataset):
    def __init__(self, root_dir):
        self.image_paths = []
        self.labels = []

        for label in ["normal", "cancer"]:
            folder = os.path.join(root_dir, label)
            for file in os.listdir(folder):
                self.image_paths.append(os.path.join(folder, file))
                self.labels.append(0 if label=="no_cancer" else 1)

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image)
        label = self.labels[idx]
        return image, label

# =========================================================
# 6️⃣ Compute ROC & dynamic thresholds
# =========================================================
from sklearn.metrics import roc_curve

@st.cache_data
def compute_roc(_model, val_dir):
    val_dataset = ValDataset(val_dir)
    val_loader = DataLoader(val_dataset, batch_size=16)
    y_true, y_scores = [], []

    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            y_scores.extend(probs.flatten())
            y_true.extend(labels.numpy())

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    return fpr, tpr, thresholds

fpr, tpr, thresholds = compute_roc(model, VAL_DIR)

# =========================================================
# 7️⃣ Sliders for sensitivity & specificity
# =========================================================
sensitivity = st.slider(
    "Select desired Sensitivity (True Positive Rate)",
    min_value=0.5, max_value=1.0, value=0.95, step=0.01
)

specificity = st.slider(
    "Select desired Specificity (True Negative Rate)",
    min_value=0.5, max_value=1.0, value=0.95, step=0.01
)

# Thresholds closest to desired sensitivity and specificity
threshold_sens_idx = np.argmin(np.abs(tpr - sensitivity))
threshold_spec_idx = np.argmin(np.abs((1 - fpr) - specificity))

# Conservative threshold to reduce false positives & negatives
threshold = float(min(thresholds[threshold_sens_idx], thresholds[threshold_spec_idx]))

st.info(f"Selected threshold: {threshold:.3f} "
        f"(Sensitivity ~{sensitivity*100:.0f}%, Specificity ~{specificity*100:.0f}%)")

# =========================================================
# 8️⃣ Image upload & prediction
# =========================================================
uploaded_file = st.file_uploader("Upload a mammogram image", type=["png","jpg","jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = preprocess_image(image)

    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()

    if prob > threshold:
        st.error(f"Cancer Detected (Confidence: {prob:.2f})")
    else:
        st.success(f"No Cancer Detected (Confidence: {1-prob:.2f})")

    if st.checkbox("Show Grad-CAM"):
        # Adjust according to your model architecture
        target_layer = model.model.layer4[-1]  # for ResNet-based backbone
        cam = GradCAM(model, target_layer)
        heatmap = cam.generate(input_tensor)
        img_np = np.array(image)
        cam_image = overlay_gradcam(img_np, heatmap)
        st.image(cam_image, caption="Grad-CAM Visualization", use_column_width=True)

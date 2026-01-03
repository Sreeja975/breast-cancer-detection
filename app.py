import streamlit as st
from PIL import Image
from helpers import (
    load_model,
    predict,
    preprocess_image,
    GradCAM,
    overlay_gradcam
)

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Breast Cancer Detection",
    page_icon="🩺",
    layout="centered"
)

# =====================================================
# HEADER
# =====================================================
st.title("🩺 Breast Cancer Detection System")
st.markdown(
    """
    This application uses a **ResNet-18 deep learning model** to detect  
    **breast cancer** from medical images.

    ⚠️ **For educational purposes only.**
    """
)

st.divider()

# =====================================================
# LOAD MODEL (CACHED)
# =====================================================
@st.cache_resource
def get_model():
    return load_model()

model = get_model()

# =====================================================
# FILE UPLOAD
# =====================================================
uploaded_file = st.file_uploader(
    "📤 Upload a medical image (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

# =====================================================
# MAIN LOGIC
# =====================================================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_column_width=True
    )

    st.divider()

    if st.button("🔍 Analyze Image"):
        with st.spinner("Analyzing image using deep learning model..."):
            prediction, confidence = predict(image, model)

        # ---------------- RESULT ----------------
        if prediction == "Cancer":
            st.error("⚠️ **Cancer Detected**")
        else:
            st.success("✅ **No Cancer Detected**")

        st.metric(
            label="Confidence",
            value=f"{confidence:.2%}"
        )

        # ---------------- GRAD-CAM ----------------
        st.divider()
        st.subheader("🔬 Model Explainability (Grad-CAM)")

        cam_generator = GradCAM(model)
        image_tensor = preprocess_image(image)
        cam = cam_generator.generate(image_tensor)
        cam_image = overlay_gradcam(image, cam)

        st.image(
            cam_image,
            caption="Grad-CAM Visualization",
            use_column_width=True
        )

# =====================================================
# FOOTER
# =====================================================
st.divider()
st.caption(
    "🧪 This tool is intended for **academic and research demonstration only**. "
    "Do NOT use it for real medical diagnosis."
)

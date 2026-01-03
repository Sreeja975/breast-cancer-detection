# 🩺 Breast Cancer Detection Using Deep Learning (ResNet18 + Grad-CAM)

## 📌 Project Overview
Breast cancer is one of the leading causes of cancer-related deaths among women worldwide. Early and accurate detection is critical for improving survival rates. This project presents a **deep learning–based breast cancer detection system** using **transfer learning with ResNet18**, enhanced with **Grad-CAM visual explanations** for model interpretability.

The project is designed to be **GitHub-ready**, **hackathon-friendly**, and suitable for **academic research and viva presentations**.

---

## 🎯 Objectives
- Automatically classify breast images into **Cancer** or **Normal**
- Utilize **transfer learning** to improve performance on limited medical datasets
- Provide **explainable AI (XAI)** using Grad-CAM
- Ensure reproducibility and ease of deployment

---

## 📂 Dataset Description
The dataset follows a **folder-based structure (ImageFolder format)** and does not rely on CSV annotation files.

```
train/
├── cancer/     # Malignant cases
│   ├── img_001.png
│   └── ...
└── normal/     # Benign cases
    ├── img_101.png
    └── ...
```

- Images are resized to **224 × 224**
- Grayscale images are converted to **3-channel format**
- Fully compatible with Kaggle datasets and local datasets

---

## 🧠 Methodology

### Model Architecture
- **Base Model:** ResNet18 (pretrained on ImageNet)
- **Framework:** PyTorch
- **Modification:** Final fully connected layer replaced for binary classification

### Training Setup
- **Loss Function:** Binary Cross Entropy with Logits (`BCEWithLogitsLoss`)
- **Optimizer:** Adam
- **Hardware Support:** GPU (CUDA / Kaggle)

---

## 🔍 Explainable AI – Grad-CAM
To improve trust and transparency in medical AI systems, this project integrates **Gradient-weighted Class Activation Mapping (Grad-CAM)**.

Grad-CAM highlights the regions of breast images that most influenced the model’s prediction, enabling:
- Better clinical interpretability
- Insight into model decision-making
- Research validation and visualization

---

## 📊 Results & Evaluation
- Training loss and accuracy are monitored per epoch
- Validation can be enabled using a validation split
- The pipeline can be extended to include:
  - Precision
  - Recall
  - ROC-AUC

*Exact performance depends on dataset size and training configuration.*

---

## 💾 Model Saving & Reproducibility
The trained model is saved automatically during training:

```
/kaggle/working/breast_cancer_cnn.pth
```

This model can be reloaded for inference, evaluation, or deployment.

---

## 🚀 How to Run
1. Open a Kaggle Notebook or local Python environment
2. Add the dataset with the specified folder structure
3. Run the training notebook or script sequentially
4. Download the trained model from **Kaggle → Notebook Outputs**

---

## 🏆 Key Highlights
- Transfer learning with ResNet18
- Explainable AI using Grad-CAM
- Folder-based dataset handling (no CSV dependency)
- Kaggle-optimized and reproducible workflow
- Easily extendable to deployment (Streamlit / Flask)

---

## 🔮 Future Work
- Grad-CAM++ or Score-CAM integration
- Hyperparameter tuning and cross-validation
- Multi-class breast cancer classification
- Web-based deployment for clinical demos
- Clinical validation with expert feedback

---

## 👩‍💻 Author & Usage
This project is intended for **educational, research, and hackathon use**.

You are free to fork, modify, and extend this repository.

⭐ *If you find this project useful, consider starring the repository!*


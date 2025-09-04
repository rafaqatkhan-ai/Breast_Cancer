# app.py
# ================================
# STREAMLIT APP: Breast Cancer Pipeline
# ================================

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image

# -------------------------------
# CONFIG
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = 224
num_classes = 2

# -------------------------------
# MODEL DEFINITIONS
# -------------------------------
class EfficientNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        effnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.features = effnet.features
        self.pooling = nn.AdaptiveAvgPool2d((7,7))
    def forward(self, x):
        x = self.features(x)
        x = self.pooling(x)
        return x

class VisionTransformerHead(nn.Module):
    def __init__(self, input_dim=1280, num_classes=2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        x = x.mean(dim=1)
        return self.classifier(x)

class HybridModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.feature_extractor = EfficientNetFeatureExtractor()
        self.flatten = nn.Flatten(2)
        self.transpose = lambda x: x.transpose(1,2)
        self.deit_head = VisionTransformerHead(input_dim=1280, num_classes=num_classes)
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.flatten(x)
        x = self.transpose(x)
        return self.deit_head(x)

# -------------------------------
# TRANSFORMS
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.title("🩺 Breast Cancer Detection with AI")
st.write("Upload a histopathology image to classify it as **Benign** or **Malignant**.")

# Load model
model = HybridModel(num_classes=num_classes).to(device)
model.load_state_dict(torch.load("hybrid_model_full.pth", map_location=device))
model.eval()

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_class = np.argmax(probs)

    classes = ["Benign", "Malignant"]
    st.subheader(f"🔍 Prediction: **{classes[pred_class]}**")
    st.write(f"Confidence: {probs[pred_class]*100:.2f}%")

    # Visualization (Grad-CAM placeholder)
    st.info("Grad-CAM and U-Net tumor segmentation can be added for boundary visualization.")

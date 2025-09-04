# app.py
# ================================
# STREAMLIT APP: Breast Cancer Pipeline (Styled Research Demo)
# ================================

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
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
# CUSTOM FRONTEND STYLING
# -------------------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
    }

    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }

    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }

    [data-testid="stSidebar"] {
        background: #f0f2f6;
    }

    .main-title {
        background: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-size: 24px;
        font-weight: 600;
        margin-bottom: 20px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.2);
    }

    .card {
        background: white;
        padding: 15px 25px;
        border-radius: 15px;
        margin-top: 20px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
    }

    h2, h3 {
        color: #2c3e50;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# HEADER
# -------------------------------
st.markdown(
    '<div class="main-title">A Hybrid EfficientNet–Transformer Model for Breast Cancer Detection<br>from Multispectral Histopathology Images</div>',
    unsafe_allow_html=True
)

st.write("Upload a histopathology image to classify it as **Benign** or **Malignant**.")

# Sidebar
st.sidebar.title("About this App")
st.sidebar.info(
    "This AI model combines **EfficientNet** with a **Vision Transformer head** "
    "to classify breast cancer histopathology images into **Benign** or **Malignant**. "
    "It also provides **Grad-CAM visualization** to highlight important regions."
)

# -------------------------------
# LOAD MODEL
# -------------------------------
model = HybridModel(num_classes=num_classes).to(device)
model.load_state_dict(torch.load("hybrid_model_full.pth", map_location=device))
model.eval()

# -------------------------------
# FILE UPLOADER
# -------------------------------
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

    # -------------------------------
    # CLASSIFICATION RESULT (CARD)
    # -------------------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔍 Classification Result")
    st.metric(label="Prediction", value=classes[pred_class], delta=f"{probs[pred_class]*100:.2f}% confidence")
    st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------
    # GRAD-CAM
    # -------------------------------
    class GradCAM:
        def __init__(self, model, target_layer):
            self.model = model
            self.target_layer = target_layer
            self.gradients = None
            self.activations = None
            self.hook_handles = []
            self._register_hooks()

        def _register_hooks(self):
            def forward_hook(module, input, output):
                self.activations = output.detach()
            def backward_hook(module, grad_in, grad_out):
                self.gradients = grad_out[0].detach()
            self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
            self.hook_handles.append(self.target_layer.register_backward_hook(backward_hook))

        def remove_hooks(self):
            for handle in self.hook_handles:
                handle.remove()

        def generate(self, input_tensor, class_idx=None):
            output = self.model(input_tensor)
            if class_idx is None:
                class_idx = output.argmax(dim=1).item()
            self.model.zero_grad()
            target = output[:, class_idx]
            target.backward()

            weights = self.gradients.mean(dim=(2, 3), keepdim=True)
            cam = (weights * self.activations).sum(dim=1, keepdim=True)
            cam = torch.relu(cam)
            cam = cam.squeeze().cpu().numpy()
            cam = cv2.resize(cam, (image_size, image_size))
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            return cam

    # Run Grad-CAM only after prediction
    target_layer = model.feature_extractor.features[-1]  # last conv block
    grad_cam = GradCAM(model, target_layer)
    cam = grad_cam.generate(input_tensor)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    overlay = heatmap + np.float32(image.resize((image_size, image_size))) / 255
    overlay = overlay / np.max(overlay)

    # -------------------------------
    # GRAD-CAM VISUALIZATION (CARD)
    # -------------------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔥 Grad-CAM Visualization")
    st.image((overlay * 255).astype(np.uint8), use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

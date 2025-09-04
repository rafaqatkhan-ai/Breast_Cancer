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

    st.subheader("🔥 Grad-CAM Visualization")
    st.image((overlay * 255).astype(np.uint8), use_column_width=True)
    import segmentation_models_pytorch as smp

# -------------------------------
# U-NET SEGMENTATION (placeholder)
# -------------------------------
# Assume you trained and saved unet_model.pth
# -------------------------------
# SIMPLE CUSTOM U-NET (no SMP needed)
# -------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(UNet, self).__init__()
        self.down1 = DoubleConv(n_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(128, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv1 = DoubleConv(128, 64)

        self.final = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)

        bn = self.bottleneck(p2)

        u2 = self.up2(bn)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.conv2(u2)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.conv1(u1)

        return torch.sigmoid(self.final(u1))

# Load lightweight U-Net
unet_model = UNet(n_channels=3, n_classes=1).to(device)
unet_model.load_state_dict(torch.load("hybrid_model_full.pth", map_location=device))
unet_model.eval()


with torch.no_grad():
    mask_pred = torch.sigmoid(unet_model(input_tensor)).cpu().numpy()[0,0]
    mask_pred = (mask_pred > 0.5).astype(np.uint8) * 255

# Overlay mask
mask_resized = cv2.resize(mask_pred, (image_size, image_size))
image_np = np.array(image.resize((image_size, image_size)))
overlay_mask = cv2.addWeighted(image_np, 0.7, cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2RGB), 0.3, 0)

st.subheader("🩸 U-Net Segmentation")
st.image(overlay_mask, use_column_width=True)

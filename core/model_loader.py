import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
from torchvision.models.detection import ssd
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
import streamlit as st
import os

# --- DEFINISI ARSITEKTUR (WAJIB ADA) ---
class SSDExtraBlocks(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.blocks.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU6(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU6(inplace=True),
        ))
        for _ in range(3):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels), nn.ReLU6(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels), nn.ReLU6(inplace=True),
            ))
    def forward(self, x, features):
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features

class MobileNetV2SSDBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = mobilenet_v2(weights=None)
        self.features = base_model.features
        self.indices = [13, 18]
        self.extra_blocks = SSDExtraBlocks(1280, 512)
        self.out_channels = [96, 1280, 512, 512, 512, 512]
    def forward(self, x):
        features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.indices: features.append(x)
        features = self.extra_blocks(features[-1], features)
        return {str(i): f for i, f in enumerate(features)}

def create_model(num_classes):
    backbone = MobileNetV2SSDBackbone()
    anchor_generator = DefaultBoxGenerator([[2], [2, 3], [2, 3], [2, 3], [2], [2]],
                                          scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],
                                          steps=[8, 16, 32, 64, 100, 300])
    return ssd.SSD(backbone=backbone, anchor_generator=anchor_generator, size=(300, 300), num_classes=num_classes)

@st.cache_resource
def load_detection_model():
    CLASS_NAMES = ['background', 'Bacterial', 'Downy_mildew', 'Powdery_mildew', 'Septoria_Blight', 'Viral', 'Wilt_leaf_blight', 'healthy']
    device = torch.device('cpu')
    model = create_model(num_classes=len(CLASS_NAMES))
    # Pastikan path model benar (naik satu tingkat dari core/ ke model/)
    model_path = os.path.join(os.path.dirname(__file__), "..", "model", "best_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, CLASS_NAMES, device
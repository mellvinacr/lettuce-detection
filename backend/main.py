import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import mobilenet_v2
from torchvision.models.detection import ssd
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse, FileResponse
import cv2
import numpy as np
import base64
import os
import tempfile
from collections import Counter

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- ARSITEKTUR MODEL (Sesuai Training Anda) ---
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

# --- INIT MODEL ---
CLASS_NAMES = ['background', 'Bacterial', 'Downy_mildew', 'Powdery_mildew', 'Septoria_Blight', 'Viral', 'Wilt_leaf_blight', 'healthy']
device = torch.device('cpu')
model_detection = create_model(num_classes=8)
model_path = os.path.join(os.path.dirname(__file__), "model", "best_model.pth")
model_detection.load_state_dict(torch.load(model_path, map_location=device))
model_detection.eval()

# --- FUNGSI DRAWING (Mencegah Teks Terpotong) ---
def draw_detections(img, prediction):
    h_orig, w_orig = img.shape[:2]
    detected_labels = []
    
    # Skala font otomatis sesuai resolusi gambar
    font_scale = max(0.6, w_orig / 1000)
    thickness = max(2, int(w_orig / 500))

    for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
        if score > 0.4:
            x1, y1, x2, y2 = box.int().tolist()
            name = CLASS_NAMES[label.item()]
            detected_labels.append(name)
            
            color = (0, 255, 0) if name == 'healthy' else (0, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness * 2)
            
            label_txt = f"{name} {score:.2f}"
            (w_txt, h_txt), baseline = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            # Logika agar teks tidak keluar frame atas
            text_y = y1 + h_txt + 10 if y1 - h_txt - 15 < 0 else y1 - 10
            
            cv2.rectangle(img, (x1, text_y - h_txt - 5), (x1 + w_txt, text_y + baseline), color, -1)
            cv2.putText(img, label_txt, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
            
    return img, detected_labels

# --- ENDPOINT GAMBAR ---
@app.post("/detect")
async def detect_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    img_tensor = T.ToTensor()(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).to(device)
    with torch.no_grad():
        prediction = model_detection([img_tensor])[0]

    img, labels = draw_detections(img, prediction)
    _, buffer = cv2.imencode('.jpg', img)
    
    return {
        "image": f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}",
        "summary": dict(Counter(labels))
    }

# --- 2. ENDPOINT DETEKSI VIDEO YANG DIPERBAIKI ---
@app.post("/detect-video")
async def detect_video(file: UploadFile = File(...)):
    # Simpan input sementara
    temp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_in.write(await file.read())
    temp_in_path = temp_in.name
    temp_in.close()

    cap = cv2.VideoCapture(temp_in_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    # Gunakan Codec XVID (.avi) agar stabil tanpa OpenH264
    temp_out_path = temp_in_path + "_out.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID') 
    out = cv2.VideoWriter(temp_out_path, fourcc, fps, (width, height))

    all_labels = []

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Deteksi frame (Gunakan fungsi draw_detections Anda)
            img_t = T.ToTensor()(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).to(device)
            with torch.no_grad():
                pred = model_detection([img_t])[0]
            
            frame, labels_in_frame = draw_detections(frame, pred)
            all_labels.extend(labels_in_frame)
            out.write(frame)
    finally:
        # Lepaskan kunci file agar tidak PermissionError
        cap.release()
        out.release()

    # Baca hasil deteksi dan ubah ke Base64 agar bisa dikirim via JSON
    with open(temp_out_path, "rb") as v_file:
        v_encoded = base64.b64encode(v_file.read()).decode('utf-8')

    # Bersihkan file fisik
    if os.path.exists(temp_in_path): os.remove(temp_in_path)
    if os.path.exists(temp_out_path): os.remove(temp_out_path)

    # Kirim respons JSON yang mengandung Video dan Summary
    return JSONResponse(content={
        "video": f"data:video/x-msvideo;base64,{v_encoded}",
        "summary": dict(Counter(all_labels))
    })
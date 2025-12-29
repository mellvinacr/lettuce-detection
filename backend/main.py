import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import mobilenet_v2
from torchvision.models.detection import ssd
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
# PERBAIKAN: Import JSONResponse agar tidak NameError
from fastapi.responses import JSONResponse, FileResponse 
from collections import Counter
import cv2
import numpy as np
import base64
import os
import tempfile
import json
import datetime

app = FastAPI()

# Konfigurasi CORS agar React bisa baca header kustom dan JSON
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Summary"] 
)

# Path file riwayat untuk menyimpan informasi hasil deteksi di backend
HISTORY_FILE = os.path.join(os.path.dirname(__file__), "history.json")

def save_to_history(filename, summary, media_type):
    """Menyimpan hasil deteksi ke file JSON di backend"""
    history_data = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                history_data = json.load(f)
        except:
            history_data = []

    new_entry = {
        "id": len(history_data) + 1,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "fileName": filename,
        "type": media_type,
        "results": summary
    }
    history_data.append(new_entry)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history_data, f, indent=4)

# --- ARSITEKTUR MODEL (Tetap Sesuai Training) ---
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

def draw_detections(img, prediction):
    h_orig, w_orig = img.shape[:2]
    detected_labels = []
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
            text_y = y1 + h_txt + 10 if y1 - h_txt - 15 < 0 else y1 - 10
            cv2.rectangle(img, (x1, text_y - h_txt - 5), (x1 + w_txt, text_y + baseline), color, -1)
            cv2.putText(img, label_txt, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    return img, detected_labels

# --- ENDPOINTS ---

@app.post("/detect")
async def detect_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_tensor = T.ToTensor()(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).to(device)
    with torch.no_grad():
        prediction = model_detection([img_tensor])[0]
    
    img, labels = draw_detections(img, prediction)
    summary = dict(Counter(labels))
    
    # Simpan informasi ke JSON di backend
    save_to_history(file.filename, summary, "image")
    
    _, buffer = cv2.imencode('.jpg', img)
    return JSONResponse(content={
        "image": f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}",
        "summary": summary
    })

@app.post("/detect-video")
async def detect_video(file: UploadFile = File(...)):
    temp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_in.write(await file.read())
    temp_in_path = temp_in.name
    temp_in.close()

    cap = cv2.VideoCapture(temp_in_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    temp_out_path = temp_in_path + "_out.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(temp_out_path, fourcc, fps, (width, height))

    all_labels = []
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            img_t = T.ToTensor()(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).to(device)
            with torch.no_grad():
                pred = model_detection([img_t])[0]
            frame, labels_in_frame = draw_detections(frame, pred)
            all_labels.extend(labels_in_frame)
            out.write(frame)
    finally:
        # PERBAIKAN: Melepaskan file agar tidak WinError 32
        cap.release()
        out.release()

    summary_data = dict(Counter(all_labels))
    
    # Simpan informasi ke JSON di backend
    save_to_history(file.filename, summary_data, "video")
    
    if os.path.exists(temp_in_path): os.remove(temp_in_path)

    return FileResponse(
        temp_out_path, 
        media_type="video/mp4",
        headers={
            "X-Summary": json.dumps(summary_data),
            "Access-Control-Expose-Headers": "X-Summary" 
        }
    )

@app.get("/history")
async def get_history():
    """Melihat riwayat deteksi yang tersimpan di backend"""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return {"message": "Belum ada riwayat deteksi"}
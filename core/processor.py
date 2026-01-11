import cv2
import numpy as np
import torch
import torchvision.transforms as T
import tempfile
import os
import av
from collections import Counter

def draw_detections(img, prediction, class_names):
    """Fungsi standar untuk menggambar bounding box tebal dan teks"""
    detected_labels = []
    # Ketebalan kotak 4 dan font 0.7 dengan thickness 3 agar terlihat jelas
    for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
        if score > 0.4:
            x1, y1, x2, y2 = map(int, box)
            name = class_names[label.item()]
            detected_labels.append(name)
            
            # Hijau untuk sehat (Healthy), Merah untuk penyakit (Disease)
            # Menyesuaikan dengan logika warna pada desain UI Anda
            color = (0, 255, 0) if name.lower() == 'healthy' else (0, 0, 255)
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 4) 
            cv2.putText(img, f"{name} {score:.2f}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 3)
    return img, detected_labels

def process_video_in_memory(input_path, model, device, class_names):
    """Logika Video Analysis: In-Memory & Auto-Cleanup"""
    temp_final = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    temp_raw = temp_final.replace(".mp4", "_raw.mp4")
    
    cap = cv2.VideoCapture(input_path)
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    out = cv2.VideoWriter(temp_raw, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    all_labels = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = T.ToTensor()(img_rgb).to(device)
        with torch.no_grad():
            pred = model([img_tensor])[0]
        
        frame, labels = draw_detections(frame, pred, class_names)
        all_labels.extend(labels)
        out.write(frame)
        
    cap.release()
    out.release()
    
    # Konversi codec ke H.264 agar video dapat diputar langsung di browser Streamlit
    os.system(f"ffmpeg -i {temp_raw} -vcodec libx264 -f mp4 {temp_final} -y -loglevel quiet")
    
    with open(temp_final, "rb") as f:
        video_bytes = f.read()
    
    # Pembersihan file sementara di backend
    if os.path.exists(temp_raw): os.remove(temp_raw)
    if os.path.exists(temp_final): os.remove(temp_final)
    
    return video_bytes, dict(Counter(all_labels))

class RealTimeProcessor:
    """Logika Real-time: Processing stream frame demi frame"""
    def __init__(self, model, device, class_names):
        self.model = model
        self.device = device
        self.class_names = class_names
        # VARIABEL BARU: Untuk menyimpan log deteksi terbaru yang akan ditampilkan di UI
        self.latest_detections = {} 

    def recv(self, frame):
        # Konversi frame WebRTC ke format ndarray (BGR)
        img = frame.to_ndarray(format="bgr24")
        
        # Preprocessing gambar untuk model SSD
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = T.ToTensor()(img_rgb).to(self.device)
        
        with torch.no_grad():
            prediction = self.model([img_tensor])[0]
        
        # Gambar deteksi dan tangkap label yang muncul di frame saat ini
        img_result, labels = draw_detections(img, prediction, self.class_names)
        
        # Update log summary secara real-time untuk ditampilkan di kolom ScanRealtime
        self.latest_detections = dict(Counter(labels))
        
        # Kembalikan frame yang sudah diproses ke feed video
        return av.VideoFrame.from_ndarray(img_result, format="bgr24")
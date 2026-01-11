import cv2
import numpy as np
import torch
import torchvision.transforms as T
import tempfile
import os
import av
from collections import Counter

def draw_detections(img, prediction, class_names):
    """Fungsi dengan perbaikan visibilitas bounding box agar tidak terpotong"""
    detected_labels = []
    h, w, _ = img.shape # Ambil dimensi gambar untuk pembatasan koordinat

    for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
        if score > 0.4:
            # Pastikan koordinat box tidak keluar dari batas gambar (clamping)
            x1 = max(0, int(box[0]))
            y1 = max(0, int(box[1]))
            x2 = min(w, int(box[2]))
            y2 = min(h, int(box[3]))
            
            name = class_names[label.item()]
            detected_labels.append(name)
            
            # Warna kontras: Hijau (Sehat) atau Merah (Penyakit)
            color = (0, 255, 0) if name.lower() == 'healthy' else (0, 0, 255)
            
            # --- PENINGKATAN VISIBILITAS ---
            # 1. Gunakan ketebalan (thickness) minimal 4 atau 5 untuk garis
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 5) 
            
            # 2. Tambahkan background untuk teks label agar terbaca jelas
            label_text = f"{name} {score:.2f}"
            (t_w, t_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            
            # Atur posisi label: jika objek terlalu di atas, taruh label di dalam kotak
            text_y = y1 - 10 if y1 - 10 > t_h else y1 + t_h + 10
            
            # Gambar background hitam di belakang teks (opsional, untuk kontras maksimal)
            cv2.rectangle(img, (x1, text_y - t_h - 5), (x1 + t_w, text_y + 5), color, -1)
            
            # Tulis teks putih di atas background warna
            cv2.putText(img, label_text, (x1, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        
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
    def __init__(self, model, device, class_names):
        self.model = model
        self.device = device
        self.class_names = class_names
        self.latest_detections = {}
        self.all_frame_labels = [] # Untuk log akumulasi
        
        # Logika Rekaman
        self.recording = False
        self.out = None
        self.temp_path = None

    def start_recording(self, width, height, fps=20.0):
        self.temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(self.temp_path, fourcc, fps, (width, height))
        self.recording = True
        self.all_frame_labels = [] # Reset log saat mulai rekam

    def stop_recording(self):
        self.recording = False
        if self.out:
            self.out.release()
            self.out = None
        return self.temp_path, dict(Counter(self.all_frame_labels))

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = T.ToTensor()(img_rgb).to(self.device)
        
        with torch.no_grad():
            prediction = self.model([img_tensor])[0]
        
        img_result, labels = draw_detections(img, prediction, self.class_names)
        
        # Update UI Log (Realtime)
        self.latest_detections = dict(Counter(labels))
        
        # Jika sedang merekam, simpan frame dan akumulasi log
        if self.recording and self.out:
            self.out.write(img_result)
            self.all_frame_labels.extend(labels)
            
        return av.VideoFrame.from_ndarray(img_result, format="bgr24")
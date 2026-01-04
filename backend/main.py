import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import mobilenet_v2
from torchvision.models.detection import ssd
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
import torchvision.ops as ops
from fastapi import FastAPI, File, UploadFile, HTTPException
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

def letterbox_resize(img, target_size=300):
    """Resize image preserving aspect ratio, pad dengan gray color"""
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create canvas dan paste di tengah
    canvas = np.full((target_size, target_size, 3), 128, dtype=np.uint8)
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas, scale, (x_offset, y_offset)

def draw_detections(img, prediction):
    h_orig, w_orig = img.shape[:2]
    detected_labels = []
    font_scale = max(0.6, w_orig / 1000)
    thickness = max(2, int(w_orig / 500))

    for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
        if score > 0.4:
            # ensure ints
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
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


def postprocess_prediction(prediction, orig_w, orig_h, input_w=300, input_h=300, conf_thresh=0.2, iou_thresh=0.45, scale=1.0, offset=(0,0)):
    # prediction contains 'boxes','labels','scores' in model input scale
    boxes = prediction['boxes']
    labels = prediction['labels']
    scores = prediction['scores']

    if boxes.numel() == 0:
        return []

    # Remove letterbox offset first, then scale to original size
    scale_x = orig_w / input_w
    scale_y = orig_h / input_h
    boxes_scaled = boxes.clone()
    # First: remove offset (letterbox padding)
    offset_x, offset_y = offset
    boxes_scaled[:, 0] = (boxes[:, 0] - offset_x) / scale
    boxes_scaled[:, 2] = (boxes[:, 2] - offset_x) / scale
    boxes_scaled[:, 1] = (boxes[:, 1] - offset_y) / scale
    boxes_scaled[:, 3] = (boxes[:, 3] - offset_y) / scale
    # Clamp to valid range after offset removal
    boxes_scaled[:, 0] = torch.clamp(boxes_scaled[:, 0], 0, orig_w)
    boxes_scaled[:, 2] = torch.clamp(boxes_scaled[:, 2], 0, orig_w)
    boxes_scaled[:, 1] = torch.clamp(boxes_scaled[:, 1], 0, orig_h)
    boxes_scaled[:, 3] = torch.clamp(boxes_scaled[:, 3], 0, orig_h)

    results = []
    # Apply per-class NMS and thresholding
    unique_labels = labels.unique()
    for lbl in unique_labels:
        if lbl.item() == 0:
            continue
        inds = (labels == lbl).nonzero(as_tuple=True)[0]
        cls_boxes = boxes_scaled[inds]
        cls_scores = scores[inds]
        keep = ops.nms(cls_boxes, cls_scores, iou_thresh)
        for k in keep:
            s = float(cls_scores[k])
            if s >= conf_thresh:
                b = cls_boxes[k].tolist()
                results.append({
                    'box': [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                    'label': CLASS_NAMES[int(labels[inds[k]].item())],
                    'score': s,
                })

    return results

# --- ENDPOINTS ---

@app.post("/detect")
async def detect_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    h_orig, w_orig = img.shape[:2]
    print(f"[DETECT] Received image: {file.filename}, size: {w_orig}x{h_orig}, bytes: {len(contents)}")

    # Preprocess using letterbox (preserve aspect ratio)
    input_size = 300
    resized, scale, offset = letterbox_resize(img, target_size=input_size)
    img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_tensor = T.ToTensor()(img_rgb)
    img_tensor = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img_tensor).to(device)

    with torch.no_grad():
        prediction = model_detection([img_tensor])[0]

    # Postprocess: scale boxes back (account for letterbox offset)
    detections = postprocess_prediction(prediction, w_orig, h_orig, input_w=input_size, input_h=input_size, scale=scale, offset=offset)
    print(f"[DETECT] Detections found: {len(detections)}")
    for i, det in enumerate(detections[:5]):  # Log first 5
        print(f"  [{i}] {det['label']}: {det['score']:.3f} at {det['box']}")

    # Draw detections onto original image for returning preview
    for det in detections:
        x1, y1, x2, y2 = map(int, det['box'])
        name = det['label']
        score = det['score']
        color = (0, 255, 0) if name == 'healthy' else (0, 0, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label_txt = f"{name} {score:.2f}"
        cv2.putText(img, label_txt, (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    summary = dict(Counter([d['label'] for d in detections]))
    print(f"[DETECT] Summary: {summary}")

    # Simpan informasi ke JSON di backend
    save_to_history(file.filename, summary, "image")

    _, buffer = cv2.imencode('.jpg', img)
    return JSONResponse(content={
        "image": f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}",
        "summary": summary,
        "detections": detections
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

    # Fallbacks for bad metadata from input video
    if fps is None or fps <= 0 or not np.isfinite(fps):
        fps = 25.0
    if width == 0 or height == 0:
        # try to read one frame to infer size
        ret, frame_tmp = cap.read()
        if ret and frame_tmp is not None:
            height, width = frame_tmp.shape[:2]
            # rewind capture
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            cap.release()
            raise HTTPException(status_code=400, detail="Could not determine video dimensions")

    temp_out_path = temp_in_path + "_out.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_out_path, fourcc, float(fps), (width, height))

    # If writer failed to open, try an alternate codec
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(temp_out_path, fourcc, float(fps), (width, height))
        if not out.isOpened():
            cap.release()
            raise HTTPException(status_code=500, detail="Failed to initialize video writer (codec issue)")

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

    # Do NOT remove temp_in_path yet — we may need to fall back to it.

    # Log output file size for debugging and decide whether output is valid
    out_size = None
    try:
        if os.path.exists(temp_out_path):
            out_size = os.path.getsize(temp_out_path)
        print(f"[DETECT-VIDEO] Produced output: {temp_out_path}, size={out_size} bytes, fps={fps}, resolution={width}x{height}")
    except Exception as e:
        print(f"[DETECT-VIDEO] Produced output: {temp_out_path} (size unknown) - {e}")

    # If output seems too small or missing, fallback to returning the original uploaded file
    MIN_VALID_SIZE = 2000
    if out_size is None or out_size < MIN_VALID_SIZE:
        print(f"[DETECT-VIDEO] Output file too small or missing (size={out_size}). Falling back to original upload: {temp_in_path}")
        # Return original uploaded file so frontend still receives a playable video
        return FileResponse(
            temp_in_path,
            media_type="video/mp4",
            headers={
                "X-Summary": json.dumps(summary_data),
                "X-Fallback": json.dumps({"reason": "output_too_small", "out_size": out_size}),
                "Access-Control-Expose-Headers": "X-Summary, X-Fallback"
            }
        )

    # Otherwise return processed output
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

@app.post("/debug")
async def debug_detections(file: UploadFile = File(...)):
    """Debug endpoint untuk melihat raw predictions dari model sebelum postprocessing"""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    h_orig, w_orig = img.shape[:2]

    # Preprocess
    input_size = 300
    resized, scale, offset = letterbox_resize(img, target_size=input_size)
    img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_tensor = T.ToTensor()(img_rgb)
    img_tensor = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img_tensor).to(device)

    with torch.no_grad():
        prediction = model_detection([img_tensor])[0]

    # Tampilkan raw predictions
    raw_boxes = prediction['boxes'].tolist()
    raw_labels = [CLASS_NAMES[int(l.item())] for l in prediction['labels']]
    raw_scores = prediction['scores'].tolist()

    # Summary dengan berbagai threshold
    summaries = {}
    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        count = sum(1 for s in raw_scores if s >= thresh)
        summaries[f"count_above_{thresh}"] = count

    return JSONResponse(content={
        "raw_predictions": {
            "num_detections": len(raw_boxes),
            "scores": raw_scores[:10],  # Top 10
            "labels": raw_labels[:10],
            "boxes": raw_boxes[:10],
        },
        "score_stats": {
            "min": min(raw_scores) if raw_scores else 0,
            "max": max(raw_scores) if raw_scores else 0,
            "mean": sum(raw_scores) / len(raw_scores) if raw_scores else 0,
        },
        "thresholds_summary": summaries
    })

@app.post("/detect-realtime")
async def detect_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    h_orig, w_orig = img.shape[:2]
    print(f"[REALTIME DEBUG] Received frame size: {w_orig}x{h_orig}")
    input_size = 300
    resized, scale, offset = letterbox_resize(img, target_size=input_size)
    img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_tensor = T.ToTensor()(img_rgb)
    img_tensor = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img_tensor).to(device)

    with torch.no_grad():
        prediction = model_detection([img_tensor])[0]

    # Postprocess with letterbox handling
    detections = postprocess_prediction(prediction, w_orig, h_orig, input_w=input_size, input_h=input_size, scale=scale, offset=offset)

    # Render overlay for debug preview
    img_rendered = img.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det['box'])
        name = det['label']
        score = det['score']
        color = (0, 255, 0) if name == 'healthy' else (0, 0, 255)
        cv2.rectangle(img_rendered, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_rendered, f"{name} {score:.2f}", (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    summary = dict(Counter([d['label'] for d in detections]))
    _, buffer = cv2.imencode('.jpg', img_rendered)
    return JSONResponse(content={
        "image": f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}",
        "summary": summary,
        "detections": detections
    })
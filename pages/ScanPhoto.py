import streamlit as st
import datetime
from collections import Counter
import torch
import torchvision.transforms as T
import cv2
import numpy as np
import io

# Mengimpor fungsi dan loader dari folder core Anda
from core.processor import draw_detections
from core.model_loader import load_detection_model

def get_risk_status(summary):
    """Meniru logika getDetectionStatus dari React"""
    total = sum(summary.values())
    disease_types = list(summary.keys())
    actual_diseases = [d for d in disease_types if d.lower() != 'healthy']
    actual_count = sum(summary[d] for d in actual_diseases)

    if total == 0 or (len(disease_types) == 1 and disease_types[0].lower() == "healthy"):
        return {"status": "✓ Healthy", "color": "#10b981", "bg": "rgba(16,185,129,0.1)", "desc": "Tanaman dalam kondisi sehat"}
    
    if actual_count <= 2 and len(actual_diseases) == 1:
        return {"status": "⚠️ Low Risk", "color": "#3b82f6", "bg": "rgba(59,130,246,0.1)", "desc": "Risiko rendah, monitoring disarankan"}
    elif actual_count <= 5 and len(actual_diseases) <= 2:
        return {"status": "⚠️ Medium Risk", "color": "#f59e0b", "bg": "rgba(245,158,11,0.1)", "desc": "Risiko sedang, lakukan treatment segera"}
    else:
        return {"status": "🚨 High Risk", "color": "#ef4444", "bg": "rgba(239,68,68,0.1)", "desc": "Risiko tinggi - segera ambil tindakan!"}

def show():
    # MEMUAT MODEL
    try:
        model_detection, CLASS_NAMES, device = load_detection_model()
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return

    st.title("Photo Disease Detection")
    st.markdown("<p style='color: #64748b;'>Upload lettuce photos for AI-powered disease analysis</p>", unsafe_allow_html=True)

    # --- HERO CARD ---
    st.markdown("""
        <div class="hero-card">
            <div style="flex: 1;">
                <div class="ai-badge">AI POWER DETECTION</div>
                <h2 style="color: #0f172a; font-weight: 800; margin-bottom: 12px;">Analisis Gambar dengan Computer Vision</h2>
                <p style="color: #64748b;">Sistem akan mendeteksi objek pada gambar dengan hasil yang cepat dan akurat.</p>
                <div style="display: flex; gap: 12px; margin-top: 15px;">
                    <span style="padding: 8px 16px; background: rgba(16, 185, 129, 0.1); color: #065f46; border-radius: 12px; font-size: 13px; font-weight: 600;">✓ Object Detection</span>
                    <span style="padding: 8px 16px; background: rgba(0, 188, 212, 0.1); color: #00838f; border-radius: 12px; font-size: 13px; font-weight: 600;">✓ Bounding Box</span>
                </div>
            </div>
            <h1 style="font-size: 80px; margin: 0;">🖼️</h1>
        </div>
    """, unsafe_allow_html=True)


    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key="photo_scanner", label_visibility="collapsed")

    if uploaded_file:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Image Preview")
            st.image(uploaded_file, use_container_width=True)
            
            # Button dengan type="primary" agar dideteksi oleh CSS khusus di styles.py
            if st.button("🔍 Start Detection", use_container_width=True, type="primary"):
                with st.spinner("Analyzing..."):
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    img = cv2.imdecode(file_bytes, 1)
                    
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_t = T.ToTensor()(img_rgb).to(device)
                    
                    with torch.no_grad():
                        prediction = model_detection([img_t])[0]
                    
                    # res_img mengandung Bounding Box dari core/processor.py
                    res_img, labels_list = draw_detections(img, prediction, CLASS_NAMES)
                    summary = dict(Counter(labels_list))
                    
                    # Konversi untuk display (RGB) & download (BGR asli)
                    final_rgb = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
                    _, buffer = cv2.imencode('.jpg', res_img)
                    
                    st.session_state.photo_result = {
                        "img": final_rgb, 
                        "summary": summary,
                        "download_bytes": buffer.tobytes()
                    }

        if "photo_result" in st.session_state:
            res_data = st.session_state.photo_result
            risk = get_risk_status(res_data['summary'])

            with col2:
                st.subheader("Detection Result")
                st.image(res_data['img'], use_container_width=True)
                
                # Fitur Download Hasil
                st.download_button(
                    label="📥 Download Result Image",
                    data=res_data['download_bytes'],
                    file_name=f"detected_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
                    mime="image/jpeg",
                    use_container_width=True
                )

                # Render Risk Badge
                st.markdown(f"""
                    <div style="padding: 12px; background: {risk['bg']}; border-radius: 12px; border-left: 5px solid {risk['color']}; margin-top: 10px;">
                        <h4 style="color: {risk['color']}; margin: 0;">{risk['status']}</h4>
                        <p style="color: #64748b; font-size: 14px; margin: 5px 0 0 0;">{risk['desc']}</p>
                    </div>
                """, unsafe_allow_html=True)

            # Bagian Summary bawah
            st.write("---")
            sum_col1, sum_col2 = st.columns(2)
            
            with sum_col1:
                st.markdown('<div class="summary-card">📊 DETECTION SUMMARY</div>', unsafe_allow_html=True)
                if res_data['summary']:
                    for label, count in res_data['summary'].items():
                        st.metric(label, f"{count} instances")
                else:
                    st.info("Tidak ada objek terdeteksi.")

            with sum_col2:
                st.markdown('<div class="summary-card">📋 DETECTION LOG</div>', unsafe_allow_html=True)
                st.write(f"**Analysis Type:** Photo Detection")
                st.write(f"**Scan Date:** {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}")
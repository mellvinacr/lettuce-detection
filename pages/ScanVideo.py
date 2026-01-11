import streamlit as st
import tempfile
import os
import datetime
import json
import time
import numpy as np
import base64
from core.processor import process_video_in_memory
from core.model_loader import load_detection_model

# Path file riwayat
HISTORY_FILE = "history.json"

def get_base64_image(image_path):
    """Mengonversi file gambar lokal ke format base64 untuk HTML."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        return ""

def save_to_history(filename, summary):
    """Menyimpan hasil deteksi video ke file JSON"""
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
        "type": "video",
        "results": summary
    }
    history_data.append(new_entry)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history_data, f, indent=4)

def show():
    # 1. Load Resources
    model, classes, device = load_detection_model()
    
    # Konversi Gambar untuk Hero Card
    image_path = "D:/Semester 5/Comvis/lettuce_detection/components/lettuce.jpeg"
    img_base64 = get_base64_image(image_path)
    
    st.title("Video Disease Detection")
    st.markdown("<p style='color: #64748b;'>Analisis video selada menggunakan AI-powered detection</p>", unsafe_allow_html=True)

    # 2. Hero Card (Visual Header)
    st.markdown(f"""
        <div class="hero-card">
            <div style="flex: 1;">
                <div class="ai-badge" style="background: rgba(34, 197, 94, 0.1); color: #166534; border-color: rgba(34, 197, 94, 0.2);">AI-POWERED DETECTION</div>
                <h2 style="color: #14532d; font-weight: 800; margin-bottom: 12px;">Analisis Video dengan Computer Vision</h2>
                <p style="color: #166534; line-height: 1.6;">Sistem akan memproses setiap frame video untuk mendeteksi penyakit tanaman selada secara komprehensif.</p>
                <div style="display: flex; gap: 12px; margin-top: 15px;">
                    <span style="padding: 8px 16px; background: #dcfce7; color: #166534; border-radius: 12px; font-size: 13px; font-weight: 600;">✓ Video Analysis</span>
                    <span style="padding: 8px 16px; background: rgba(34, 197, 94, 0.1); color: #166534; border-radius: 12px; font-size: 13px; font-weight: 600;">✓ Multi-Frame Detection</span>
                </div>
            </div>
            <div style="width: 280px; height: 280px; border-radius: 24px; overflow: hidden; border: 4px solid white; box-shadow: 0 20px 40px rgba(34, 197, 94, 0.15);">
                <img src="data:image/jpeg;base64,{img_base64}" style="width: 100%; height: 100%; object-fit: cover;" alt="Lettuce Mascot">
            </div>
        </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["mp4", "mov", "avi"], key="video_scanner", label_visibility="collapsed")

    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile_path = tfile.name
        tfile.close()

        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Video Preview")
            st.video(tfile_path)
            
            if st.button("🚀 Start Analysis", use_container_width=True, type="primary"):
                with st.container():
                    with st.status("🎬 Inisialisasi AI Engine...", expanded=True) as status:
                        progress_bar = st.progress(0)
                        
                        time.sleep(0.5)
                        status.update(label="🔍 Memindai Frame Video...")
                        st.write("Menganalisis kesehatan tanaman per frame...")
                        progress_bar.progress(30)
                        
                        # Proses Deteksi Sebenarnya
                        video_bytes, summary = process_video_in_memory(tfile_path, model, device, classes)
                        
                        progress_bar.progress(80)
                        status.update(label="💾 Menyimpan Hasil Analisis...", state="running")
                        st.write("Mencatat data ke riwayat deteksi...")
                        save_to_history(uploaded_file.name, summary)
                        
                        time.sleep(0.5)
                        progress_bar.progress(100)
                        status.update(label="✅ Analisis Selesai!", state="complete", expanded=False)
                    
                    st.session_state.video_result = {"bytes": video_bytes, "summary": summary}
                    st.rerun()

        # 4. Result Section
        if "video_result" in st.session_state:
            res_data = st.session_state.video_result
            
            with col2:
                st.subheader("Detection Result")
                st.video(res_data['bytes'])
                st.download_button(
                    label="📥 Download Result Video",
                    data=res_data['bytes'],
                    file_name=f"detected_{uploaded_file.name}",
                    mime="video/mp4",
                    use_container_width=True
                )

            # 5. Summary & Logs Section
            st.write("---")
            sum_col1, sum_col2 = st.columns(2)
            
            with sum_col1:
                st.markdown('<div class="summary-card" style="border-left: 4px solid #22c55e;">📊 DETECTION SUMMARY</div>', unsafe_allow_html=True)
                if res_data['summary']:
                    for label, count in res_data['summary'].items():
                        st.metric(label, f"{count} Occurrences")
                else:
                    st.info("No objects detected in video.")

            with sum_col2:
                st.markdown('<div class="summary-card" style="border-left: 4px solid #22c55e;">📋 DETECTION LOG</div>', unsafe_allow_html=True)
                st.write(f"**File Name:** `{uploaded_file.name}`")
                st.write(f"**Status:** Analysis Complete")
                st.markdown(f"""
                    <div style="margin-top: 12px; padding: 12px; background: #f0fdf4; border-radius: 8px; border-left: 4px solid #22c55e;">
                        <span style="font-size: 11px; color: #166534; font-weight: 700; display: block;">TIMESTAMP</span>
                        <span style="font-size: 14px; color: #14532d;">{datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</span>
                    </div>
                """, unsafe_allow_html=True)

        if os.path.exists(tfile_path):
            os.remove(tfile_path)
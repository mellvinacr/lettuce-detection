import streamlit as st
import tempfile
import os
import datetime
import json
import numpy as np
from core.processor import process_video_in_memory
from core.model_loader import load_detection_model

# Path file riwayat
HISTORY_FILE = "history.json"

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
    # 1. Load Resources (Model SSD MobileNetV2)
    model, classes, device = load_detection_model()
    
    st.title("Video Disease Detection")
    st.markdown("<p style='color: #64748b;'>Analisis video selada menggunakan AI-powered detection</p>", unsafe_allow_html=True)

    # 2. Hero Card (Desain Modern Selaras Dashboard)
    st.markdown("""
        <div class="hero-card">
            <div style="flex: 1;">
                <div class="ai-badge" style="background: rgba(168, 85, 247, 0.1); color: #7e22ce; border-color: rgba(168, 85, 247, 0.2);">AI-POWERED DETECTION</div>
                <h2 style="color: #0f172a; font-weight: 800; margin-bottom: 12px;">Analisis Video dengan Computer Vision</h2>
                <p style="color: #64748b; line-height: 1.6;">Sistem akan memproses setiap frame video untuk mendeteksi penyakit tanaman selada secara komprehensif.</p>
                <div style="display: flex; gap: 12px; margin-top: 15px;">
                    <span style="padding: 8px 16px; background: rgba(168, 85, 247, 0.1); color: #7e22ce; border-radius: 12px; font-size: 13px; font-weight: 600;">✓ Video Analysis</span>
                    <span style="padding: 8px 16px; background: rgba(16, 185, 129, 0.1); color: #065f46; border-radius: 12px; font-size: 13px; font-weight: 600;">✓ Multi-Frame Detection</span>
                </div>
            </div>
            <h1 style="font-size: 80px; margin: 0;">🎥</h1>
        </div>
    """, unsafe_allow_html=True)


    uploaded_file = st.file_uploader("", type=["mp4", "mov", "avi"], key="video_scanner")

    if uploaded_file:
        # Gunakan tempfile agar OpenCV bisa membaca file path
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile_path = tfile.name
        tfile.close()

        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Video Preview")
            st.video(tfile_path)
            
            # Tombol Start Analysis dengan Type Primary (Warna Ungu diatur di styles.py)
            if st.button("🚀 Start Analysis", use_container_width=True, type="primary"):
                with st.spinner("● AI is Scanning Frames..."):
                    # Proses In-Memory
                    video_bytes, summary = process_video_in_memory(tfile_path, model, device, classes)
                    
                    # Simpan ke riwayat JSON
                    save_to_history(uploaded_file.name, summary)
                    
                    # Simpan ke session state agar persisten
                    st.session_state.video_result = {"bytes": video_bytes, "summary": summary}
                    st.success("Analysis Complete & Saved to History!")

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
                st.markdown('<div class="summary-card" style="border-left: 4px solid #a855f7;">📊 DETECTION SUMMARY</div>', unsafe_allow_html=True)
                if res_data['summary']:
                    for label, count in res_data['summary'].items():
                        st.metric(label, f"{count} Occurrences")
                else:
                    st.info("No objects detected in video.")

            with sum_col2:
                st.markdown('<div class="summary-card" style="border-left: 4px solid #a855f7;">📋 DETECTION LOG</div>', unsafe_allow_html=True)
                st.write(f"**File Name:** `{uploaded_file.name}`")
                st.write(f"**Status:** Analysis Complete")
                st.markdown(f"""
                    <div style="margin-top: 12px; padding: 12px; background: #f1f5f9; border-radius: 8px; border-left: 4px solid #a855f7;">
                        <span style="font-size: 11px; color: #64748b; font-weight: 700; display: block;">TIMESTAMP</span>
                        <span style="font-size: 14px; color: #1e293b;">{datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</span>
                    </div>
                """, unsafe_allow_html=True)

        # 6. Pembersihan file input
        if os.path.exists(tfile_path):
            os.remove(tfile_path)
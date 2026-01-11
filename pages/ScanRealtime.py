import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from core.model_loader import load_detection_model
from core.processor import RealTimeProcessor
import time
import os
import datetime

# Fungsi simpan log (sama dengan yang ada di ScanVideo)
from pages.ScanVideo import save_to_history 

def show():
    # 1. Memuat model dan resources
    model, classes, device = load_detection_model()
    
    st.title("Real-time Disease Detection")
    
    # 2. Hero Card (Visual Header)
    st.markdown("""
        <div class="hero-card">
            <div style="flex: 1;">
                <div class="ai-badge">REAL-TIME AI DETECTION</div>
                <h2 style="color: #0f172a; font-weight: 800; margin-bottom: 12px;">Pemantauan Langsung dengan Computer Vision</h2>
                <p style="color: #64748b;">Sistem akan melakukan analisis setiap frame untuk hasil yang akurat.</p>
                <div style="display: flex; gap: 12px; margin-top: 15px;">
                    <span style="padding: 8px 16px; background: rgba(16, 185, 129, 0.1); color: #065f46; border-radius: 12px; font-size: 13px; font-weight: 600;">✓ Live Detection</span>
                    <span style="padding: 8px 16px; background: rgba(0, 188, 212, 0.1); color: #00838f; border-radius: 12px; font-size: 13px; font-weight: 600;">✓ Bounding Box</span>
                </div>
            </div>
            <h1 style="font-size: 80px; margin: 0;">📷</h1>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""<div class="summary-card">📹 Visual Detection Feed</div>""", unsafe_allow_html=True)
        ctx = webrtc_streamer(
            key="lettuce-realtime",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=lambda: RealTimeProcessor(model, device, classes),
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False},
        )

    with col2:
        st.markdown("""<div class="summary-card">📊 Detection Log & Tools</div>""", unsafe_allow_html=True)
        
        if ctx.video_processor:
            # PERBAIKAN: Menggunakan satu variabel container untuk kolom tunggal agar tidak ValueError
            rec_area = st.container()
            
            with rec_area:
                if not ctx.video_processor.recording:
                    # Tombol Start Record
                    if st.button("🔴 Start Record", use_container_width=True):
                        ctx.video_processor.start_recording(640, 480)
                        st.rerun()
                else:
                    # Tombol Stop & Save dengan gaya Primary
                    if st.button("⏹️ Stop & Save", use_container_width=True, type="primary"):
                        with st.spinner("Processing Record..."):
                            path, summary = ctx.video_processor.stop_recording()
                            
                            # Membuat nama file berdasarkan waktu rekaman
                            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            custom_filename = f"Detection_{timestamp_str}.mp4"
                            
                            # Simpan ke history.json menggunakan nama file baru
                            save_to_history(custom_filename, summary)
                            
                            # Konversi codec agar bisa didownload (H.264)
                            final_path = path.replace(".mp4", f"_{timestamp_str}_final.mp4")
                            os.system(f"ffmpeg -i {path} -vcodec libx264 -f mp4 {final_path} -y -loglevel quiet")
                            
                            with open(final_path, "rb") as f:
                                st.download_button(
                                    label="📥 Download Record", 
                                    data=f, 
                                    file_name=custom_filename,
                                    mime="video/mp4", 
                                    use_container_width=True
                                )
                            st.success(f"Log disimpan sebagai {custom_filename}!")

            # --- DISPLAY LOG REALTIME ---
            log_placeholder = st.empty()
            while ctx.state.playing:
                # Mengambil data deteksi terbaru dari variabel di dalam RealTimeProcessor
                latest_results = getattr(ctx.video_processor, 'latest_detections', {})
                with log_placeholder.container():
                    if latest_results:
                        for label, qty in latest_results.items():
                            # Menentukan warna border berdasarkan status kesehatan tanaman
                            status_color = '#10b981' if label.lower() == 'healthy' else '#ef4444'
                            st.markdown(f"""
                                <div style="display: flex; justify-content: space-between; background: white; padding: 10px; border-radius: 8px; margin-bottom: 8px; border-left: 4px solid {status_color}; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                                    <span style="font-weight: 600;">{label}</span>
                                    <span style="font-weight: 800;">{qty}</span>
                                </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("Mencari objek...")
                time.sleep(1)
        else:
            st.warning("Webcam Offline")
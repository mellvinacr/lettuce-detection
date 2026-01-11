import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from core.model_loader import load_detection_model
from core.processor import RealTimeProcessor
import time
import os
import datetime
import base64

# Fungsi simpan log (sama dengan yang ada di ScanVideo)
from pages.ScanVideo import save_to_history 

def get_base64_image(image_path):
    """Mengonversi file gambar lokal ke format base64 untuk HTML."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        return ""

def show():
    # 1. Memuat model dan resources
    model, classes, device = load_detection_model()
    
    # Path gambar sesuai permintaan Anda
    image_path = "D:/Semester 5/Comvis/lettuce_detection/components/lettuce.jpeg"
    img_base64 = get_base64_image(image_path)
    
    st.title("Real-time Disease Detection")
    st.markdown("<p style='color: #64748b;'>Analisis kesehatan tanaman selada secara langsung melalui kamera</p>", unsafe_allow_html=True)
    
    # 2. Hero Card (Visual Header) - Diperbarui dengan Gambar Base64 dan Warna Soft
    st.markdown(f"""
        <div class="hero-card">
            <div style="flex: 1;">
                <div class="ai-badge">REAL-TIME AI DETECTION</div>
                <h2 style="color: #14532d; font-weight: 800; margin-bottom: 12px;">Pemantauan Langsung dengan Computer Vision</h2>
                <p style="color: #166534;">Sistem akan melakukan analisis setiap frame untuk hasil yang akurat.</p>
                <div style="display: flex; gap: 12px; margin-top: 15px;">
                    <span style="padding: 8px 16px; background: #dcfce7; color: #166534; border-radius: 12px; font-size: 13px; font-weight: 600;">✓ Live Detection</span>
                    <span style="padding: 8px 16px; background: rgba(34, 197, 94, 0.1); color: #166534; border-radius: 12px; font-size: 13px; font-weight: 600;">✓ Bounding Box</span>
                </div>
            </div>
            <div style="width: 280px; height: 280px; border-radius: 24px; overflow: hidden; border: 4px solid white; box-shadow: 0 20px 40px rgba(34, 197, 94, 0.15);">
                <img src="data:image/jpeg;base64,{img_base64}" style="width: 100%; height: 100%; object-fit: cover;" alt="Lettuce Mascot">
            </div>
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
            rec_area = st.container()
            
            with rec_area:
                if not ctx.video_processor.recording:
                    if st.button("🔴 Start Record", use_container_width=True):
                        ctx.video_processor.start_recording(640, 480)
                        st.rerun()
                else:
                    if st.button("⏹️ Stop & Save", use_container_width=True, type="primary"):
                        with st.spinner("Processing Record..."):
                            path, summary = ctx.video_processor.stop_recording()
                            
                            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            custom_filename = f"Detection_{timestamp_str}.mp4"
                            
                            save_to_history(custom_filename, summary)
                            
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
                latest_results = getattr(ctx.video_processor, 'latest_detections', {})
                with log_placeholder.container():
                    if latest_results:
                        for label, qty in latest_results.items():
                            # Warna border disesuaikan: Hijau untuk Healthy, Merah untuk Penyakit
                            status_color = '#10b981' if label.lower() == 'healthy' else '#ef4444'
                            st.markdown(f"""
                                <div style="display: flex; justify-content: space-between; background: white; padding: 10px; border-radius: 8px; margin-bottom: 8px; border-left: 4px solid {status_color}; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                                    <span style="font-weight: 600; color: #1e293b;">{label}</span>
                                    <span style="font-weight: 800; color: {status_color};">{qty}</span>
                                </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("Mencari objek...")
                time.sleep(1)
        else:
            st.warning("Webcam Offline. Pastikan izin kamera telah diberikan.")
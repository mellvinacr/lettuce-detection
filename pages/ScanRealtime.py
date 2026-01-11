import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from core.model_loader import load_detection_model
from core.processor import RealTimeProcessor

def show():
    # 1. Load Model Resources (Cached)
    model, classes, device = load_detection_model()
    
    st.title("Real-time Disease Detection")
    st.markdown("<p style='color: #64748b;'>Live webcam monitoring with AI-powered disease detection</p>", unsafe_allow_html=True)

    # 2. Hero Card
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
        
        # 3. WebRTC Streamer dengan RealTimeProcessor
        ctx = webrtc_streamer(
            key="lettuce-realtime",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=lambda: RealTimeProcessor(model, device, classes),
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False},
        )

    with col2:
        st.markdown("""<div class="summary-card">📊 Detection Log Summary</div>""", unsafe_allow_html=True)
        
        # Placeholder untuk Log Dinamis (Mirip gambar image_fa346b.jpg)
        log_placeholder = st.empty()
        
        # Logika Penampilan Log dari Processor
        if ctx.video_processor:
            # Ambil data deteksi terbaru dari variabel di dalam RealTimeProcessor
            # Pastikan di core/processor.py Anda memiliki self.latest_detections = {}
            latest_results = getattr(ctx.video_processor, 'latest_detections', {})
            
            if latest_results:
                with log_placeholder.container():
                    st.markdown("""
                        <div style="display: flex; justify-content: space-between; font-size: 12px; font-weight: bold; color: #64748b; margin-bottom: 10px; border-bottom: 1px solid #e2e8f0; padding-bottom: 5px;">
                            <span>CLASSIFICATION</span>
                            <span>QTY</span>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    for label, qty in latest_results.items():
                        bg_color = "rgba(16, 185, 129, 0.1)" if label.lower() == "healthy" else "rgba(239, 68, 68, 0.1)"
                        text_color = "#065f46" if label.lower() == "healthy" else "#ef4444"
                        
                        st.markdown(f"""
                            <div style="display: flex; justify-content: space-between; align-items: center; background: white; padding: 10px; border-radius: 8px; margin-bottom: 8px; border: 1px solid #f1f5f9;">
                                <span style="background: {bg_color}; color: {text_color}; padding: 4px 12px; border-radius: 6px; font-size: 11px; font-weight: 700; text-transform: uppercase;">
                                    {label}
                                </span>
                                <span style="font-weight: 800; color: #1e293b;">{qty}</span>
                            </div>
                        """, unsafe_allow_html=True)
            else:
                log_placeholder.info("Mencari objek selada...")
        else:
            st.warning("Webcam Offline. Klik 'Start' untuk memulai.")

        # Info Box (Cara Menggunakan)
        st.markdown("""
            <div style="background: rgba(59, 130, 246, 0.05); padding: 20px; border-radius: 16px; border: 1px solid rgba(59, 130, 246, 0.15); margin-top: 20px;">
                <h5 style="color: #1e293b; margin-top:0;">💡 Cara Menggunakan</h5>
                <p style="color: #64748b; font-size: 13px; margin:0; line-height: 1.5;">
                    Klik tombol <b>Start</b> pada jendela feed untuk mengaktifkan kamera. 
                    <br><br>
                    <span style="color: #10b981; font-weight: bold;">● Healthy:</span> Kondisi Optimal<br>
                    <span style="color: #ef4444; font-weight: bold;">● Disease:</span> Perlu Penanganan
                </p>
            </div>
        """, unsafe_allow_html=True)
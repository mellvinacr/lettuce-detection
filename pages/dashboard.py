import streamlit as st
import base64
from pathlib import Path

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def show():
    st.title("Deteksi Penyakit Selada Hijau")
    
    # Jalur file lokal
    image_path = "D:/Semester 5/Comvis/lettuce_detection/components/lettuce.jpeg"
    
    try:
        # Konversi gambar ke base64
        img_base64 = get_base64_image(image_path)
        
        st.markdown(f"""
            <div class="hero-card">
                <div style="flex: 1;">
                    <div class="ai-badge">AI-POWERED DETECTION</div>
                    <h2 style="color: #0f172a; font-weight: 800; margin-bottom: 12px;">Teknologi Computer Vision untuk Pertanian Modern</h2>
                    <p style="color: #64748b; line-height: 1.6;">Deteksi dini penyakit pada tanaman selada menggunakan deep learning untuk meningkatkan hasil panen dan kualitas tanaman.</p>
                </div>
                <div style="width: 280px; height: 280px; border-radius: 24px; overflow: hidden; border: 4px solid white; box-shadow: 0 20px 40px rgba(0,188,212,0.2);">
                    <img src="data:image/jpeg;base64,{img_base64}" style="width: 100%; height: 100%; object-fit: cover;" alt="Lettuce Mascot">
                </div>
            </div>
        """, unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("File gambar tidak ditemukan. Pastikan path sudah benar.")

    # 3. Method Cards Grid
    st.subheader("Metode Deteksi")
    col1, col2, col3 = st.columns(3)

    with col1:
        # Visual Card
        st.markdown("""
            <div class="method-card">
                <div style="font-size: 30px; margin-bottom: 15px;">📷</div>
                <h4 style="margin: 0; color: #1e293b;">Scan Real-time</h4>
                <p style="color: #64748b; font-size: 14px;">Deteksi langsung menggunakan kamera untuk analisis instant.</p>
                <p style="color: #10b981; font-weight: 600; margin-bottom: 0;">Mulai Scan →</p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        # Visual Card
        st.markdown("""
            <div class="method-card">
                <div style="font-size: 30px; margin-bottom: 15px;">🖼️</div>
                <h4 style="margin: 0; color: #1e293b;">Upload Photo</h4>
                <p style="color: #64748b; font-size: 14px;">Unggah foto tanaman untuk analisis hasil penyakit mendalam.</p>
                <p style="color: #3b82f6; font-weight: 600; margin-bottom: 0;">Upload Foto →</p>
            </div>
        """, unsafe_allow_html=True)
    

    with col3:
        # Visual Card
        st.markdown("""
            <div class="method-card">
                <div style="font-size: 30px; margin-bottom: 15px;">🎥</div>
                <h4 style="margin: 0; color: #1e293b;">Video Analysis</h4>
                <p style="color: #64748b; font-size: 14px;">Proses frame video untuk deteksi komprehensif.</p>
                <p style="color: #a855f7; font-weight: 600; margin-bottom: 0;">Proses Video →</p>
            </div>
        """, unsafe_allow_html=True)
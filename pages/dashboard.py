import streamlit as st
import base64
from pathlib import Path

def get_base64_image(image_path):
    """Mengonversi file gambar lokal ke format base64 untuk HTML."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        return ""

def show():
    st.title("Deteksi Penyakit Selada Hijau")
    st.markdown("<p style='color: #64748b; font-size: 16px;'>Pilih metode deteksi untuk menganalisis kesehatan tanaman selada Anda</p>", unsafe_allow_html=True)
    
    # Jalur file lokal sesuai folder proyek Anda
    image_path = "D:/Semester 5/Comvis/lettuce_detection/components/lettuce.jpeg"
    img_base64 = get_base64_image(image_path)
    
    # 1. Hero Card Section - Diperbarui dengan Tema Hijau Muda Soft
    st.markdown(f"""
        <div class="hero-card">
            <div style="flex: 1;">
                <div class="ai-badge" style="background: #dcfce7; color: #166534; border: 1px solid #bbf7d0;">AI-POWERED DETECTION</div>
                <h2 style="color: #14532d; font-weight: 800; margin-bottom: 12px;">Teknologi Computer Vision untuk Pertanian Modern</h2>
                <p style="color: #166534; line-height: 1.6;">Deteksi dini penyakit pada tanaman selada menggunakan deep learning untuk meningkatkan hasil panen dan kualitas tanaman secara efisien.</p>
                <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-top: 10px;">
                    <span style="padding: 8px 16px; background: #f0fdf4; color: #166534; border-radius: 12px; font-size: 13px; font-weight: 600; border: 1px solid #dcfce7;">✓ Akurasi Tinggi</span>
                    <span style="padding: 8px 16px; background: rgba(34, 197, 94, 0.1); color: #15803d; border-radius: 12px; font-size: 13px; font-weight: 600; border: 1px solid rgba(34, 197, 94, 0.2);">✓ Real-time Analysis</span>
                </div>
            </div>
            <div style="width: 280px; height: 280px; border-radius: 24px; overflow: hidden; border: 4px solid white; box-shadow: 0 20px 40px rgba(34, 197, 94, 0.15);">
                <img src="data:image/jpeg;base64,{img_base64}" style="width: 100%; height: 100%; object-fit: cover;" alt="Lettuce Mascot">
            </div>
        </div>
    """, unsafe_allow_html=True)

    # 2. Method Cards Grid
    st.subheader("Metode Deteksi")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
            <div class="method-card">
                <div style="font-size: 35px; margin-bottom: 15px;">📷</div>
                <h4 style="margin: 0; color: #14532d;">Scan Real-time</h4>
                <p style="color: #64748b; font-size: 14px; margin: 10px 0;">Deteksi langsung menggunakan kamera untuk analisis instant pada tanaman.</p>
                <p style="color: #22c55e; font-weight: 700; margin-bottom: 0;">Mulai Scan →</p>
            </div>
        """, unsafe_allow_html=True)
        # Tombol Navigasi Transparan (Sesuai Logika styles.py Anda)
        if st.button("Mulai Scan", key="nav_realtime", help="Buka Kamera Real-time"):
            st.session_state.page = "Scan Realtime"
            st.rerun()

    with col2:
        st.markdown("""
            <div class="method-card">
                <div style="font-size: 35px; margin-bottom: 15px;">🖼️</div>
                <h4 style="margin: 0; color: #14532d;">Upload Photo</h4>
                <p style="color: #64748b; font-size: 14px; margin: 10px 0;">Unggah foto tanaman untuk analisis hasil penyakit secara mendalam.</p>
                <p style="color: #22c55e; font-weight: 700; margin-bottom: 0;">Upload Foto →</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Upload Foto", key="nav_photo", help="Buka Unggah Foto"):
            st.session_state.page = "Upload Photo"
            st.rerun()

    with col3:
        st.markdown("""
            <div class="method-card">
                <div style="font-size: 35px; margin-bottom: 15px;">🎥</div>
                <h4 style="margin: 0; color: #14532d;">Video Analysis</h4>
                <p style="color: #64748b; font-size: 14px; margin: 10px 0;">Proses setiap frame video untuk pemindaian penyakit yang komprehensif.</p>
                <p style="color: #22c55e; font-weight: 700; margin-bottom: 0;">Proses Video →</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Proses Video", key="nav_video", help="Buka Analisis Video"):
            st.session_state.page = "Video Analysis"
            st.rerun()
import streamlit as st

def show():
    # 1. Header Section
    st.title("Deteksi Penyakit Selada Hijau")
    st.markdown("<p style='color: #64748b; font-size: 16px;'>Pilih metode deteksi untuk menganalisis kesehatan tanaman selada Anda</p>", unsafe_allow_html=True)

    # 2. Hero Card Section
    st.markdown("""
        <div class="hero-card">
            <div style="flex: 1;">
                <div class="ai-badge">AI-POWERED DETECTION</div>
                <h2 style="color: #0f172a; font-weight: 800; margin-bottom: 12px;">Teknologi Computer Vision untuk Pertanian Modern</h2>
                <p style="color: #64748b; line-height: 1.6;">Deteksi dini penyakit pada tanaman selada menggunakan deep learning untuk meningkatkan hasil panen dan kualitas tanaman.</p>
                <div style="display: flex; gap: 12px; flex-wrap: wrap;">
                    <span style="padding: 8px 16px; background: rgba(16, 185, 129, 0.1); color: #065f46; border-radius: 12px; font-size: 13px; font-weight: 600;">✓ Akurasi Tinggi</span>
                    <span style="padding: 8px 16px; background: rgba(59, 130, 246, 0.1); color: #1e3a8a; border-radius: 12px; font-size: 13px; font-weight: 600;">✓ Real-time Analysis</span>
                </div>
            </div>
            <div style="width: 280px; height: 280px; border-radius: 24px; overflow: hidden; border: 4px solid white; box-shadow: 0 20px 40px rgba(0,188,212,0.2);">
                <img src="https://via.placeholder.com/280x280" style="width: 100%; height: 100%; object-fit: cover;" alt="Lettuce Mascot">
            </div>
        </div>
    """, unsafe_allow_html=True)

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
        
        # Tombol Overlay (Transparan)
        if st.button("GoRealtime", key="btn_realtime"):
            st.session_state.page = "Real-time Scan"
            st.rerun()

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
        
        # Tombol Overlay (Transparan)
        if st.button("GoPhoto", key="btn_photo"):
            st.session_state.page = "Photo Upload"
            st.rerun()

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
        
        # Tombol Overlay (Transparan)
        if st.button("GoVideo", key="btn_video"):
            st.session_state.page = "Video Analysis"
            st.rerun()
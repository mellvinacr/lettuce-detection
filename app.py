import streamlit as st
from components.sidebar import draw_sidebar
from components.styles import apply_custom_css
from pages import dashboard, ScanPhoto, ScanVideo, ScanRealtime

# 1. Konfigurasi Awal
st.set_page_config(
    page_title="LettuceEye AI", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None
)

# 2. Sembunyikan Navigasi Bawaan Streamlit agar Sidebar Kustom Bersih
st.markdown("""
    <style>
        [data-testid="stSidebarNav"] {display: none;}
    </style>
""", unsafe_allow_html=True)

# 3. Terapkan CSS Global (Termasuk gaya Card dan Overlay Button)
apply_custom_css()

# 4. Inisialisasi State Halaman (Default ke Dashboard)
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

# 5. Render Sidebar & Sinkronisasi dengan Session State
# Kirimkan st.session_state.page sebagai nilai awal jika sidebar mendukungnya
selected_page = draw_sidebar()

# Jika ada perubahan di Sidebar, update Session State
if selected_page != st.session_state.page:
    st.session_state.page = selected_page

# 6. Routing Logic Berdasarkan Session State
if st.session_state.page == "Dashboard":
    dashboard.show()
elif st.session_state.page == "Photo Upload":
    ScanPhoto.show()
elif st.session_state.page == "Video Analysis":
    ScanVideo.show()
elif st.session_state.page == "Real-time Scan":
    ScanRealtime.show()
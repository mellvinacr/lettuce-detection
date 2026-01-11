import streamlit as st

def apply_custom_css():
    st.markdown("""
        <style>
        /* 1. Base App Styling - Hijau Mint Pucat (Sangat Jernih) */
        .stApp {
            background-color: #f0fdf4; 
            font-family: 'Inter', -apple-system, sans-serif;
        }

        /* 2. Sidebar Styling - Putih Bersih agar tetap kontras */
        [data-testid="stSidebar"] {
            background-color: #f0f2f6 !important;
            border-right: 1px solid #dcfce7 !important;
        }
        
        iframe[title="streamlit_option_menu.option_menu"] {
            background-color: transparent !important;
        }

        /* 3. Hero Card - Putih dengan aksen Hijau Muda */
        .hero-card {
            background: #ffffff;
            border-radius: 20px;
            padding: 35px;
            margin-bottom: 32px;
            box-shadow: 0 10px 25px rgba(22, 163, 74, 0.05);
            border: 1px solid #dcfce7;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 40px;
        }

        .ai-badge {
            display: inline-block;
            padding: 6px 14px;
            background: #dcfce7; /* Hijau Muda Cerah */
            border-radius: 8px;
            margin-bottom: 16px;
            color: #166534; 
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }

        /* 4. Method Card - Clean White dengan Hover Hijau */
        .method-card {
            background: #ffffff;
            padding: 28px;
            border-radius: 18px;
            border: 1px solid #eef2ef;
            transition: all 0.3s ease;
            position: relative;
        }

        .method-card:hover {
            border-color: #4ade80;
            background: #f0fdf4;
            transform: translateY(-5px);
        }

        /* Navigasi Overlay Transparan */
        .stButton > button[p-styled="false"] {
            position: absolute; top: 0; left: 0; width: 100%; height: 100%;
            background: transparent !important; border: none !important;
            color: transparent !important; z-index: 10;
        }

        /* 5. Action Buttons - Hijau Emerald (Vibran namun tetap Soft) */
        div.stButton > button[kind="primary"] {
            background: #22c55e !important;
            color: white !important;
            border-radius: 12px !important;
            border: none !important;
            padding: 0.6rem 1.5rem !important;
            font-weight: 600 !important;
            width: 100% !important;
            box-shadow: 0 4px 12px rgba(34, 197, 94, 0.2) !important;
        }

        div.stButton > button[kind="primary"]:hover {
            background: #16a34a !important;
            box-shadow: 0 6px 15px rgba(34, 197, 94, 0.3) !important;
        }

        /* Metric & Tipografi - Warna Hijau Hutan Tua */
        [data-testid="stMetricValue"] {
            color: #166534 !important;
        }
        
        h1, h2, h3, h4 {
            color: #14532d !important; /* Hijau Gelap Jernih */
        }
        
        p {
            color: #166534 !important;
        }
        </style>
    """, unsafe_allow_html=True)
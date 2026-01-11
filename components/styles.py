import streamlit as st

def apply_custom_css():
    st.markdown("""
        <style>
        /* 1. Base App Styling */
        .stApp {
            background-color: #f1f5f9;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }

        /* 2. Sidebar Styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%) !important;
        }

        /* 3. Hero Card Styling */
        .hero-card {
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
            border-radius: 24px;
            padding: 40px;
            margin-bottom: 32px;
            box-shadow: 0 4px 24px rgba(0, 0, 0, 0.06);
            border: 1px solid rgba(226, 232, 240, 0.8);
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 40px;
        }

        .ai-badge {
            display: inline-block;
            padding: 8px 16px;
            background: linear-gradient(135deg, rgba(0, 188, 212, 0.1) 0%, rgba(0, 188, 212, 0.05) 100%);
            border-radius: 100px;
            margin-bottom: 16px;
            border: 1px solid rgba(0, 188, 212, 0.2);
            color: #00838f;
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 0.5px;
        }

        /* 4. Method Card & Navigation Logic */
        .method-card {
            background: white;
            padding: 28px;
            border-radius: 20px;
            border: 2px solid #e2e8f0;
            transition: all 0.3s ease;
            position: relative;
            cursor: pointer;
        }

        .method-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.1);
        }

        /* TOMBOL NAVIGASI: Hanya targetkan tombol dengan label "Go" (key di app) */
        /* Kita menggunakan selektor yang tidak mengganggu tombol deteksi */
        .stButton > button[p-styled="false"], 
        div[p-styled="false"] > button {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: transparent !important;
            border: none !important;
            color: transparent !important;
            z-index: 10;
        }

        /* 5. Start Detection Button (Targeted by Kind Primary) */
        /* Menyesuaikan dengan tampilan Gambar 2 (Biru Cerah/Cyan) */
        div.stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #00bcd4 0%, #00acc1 100%) !important;
            color: white !important;
            border-radius: 12px !important;
            border: none !important;
            padding: 0.6rem 2rem !important;
            font-weight: 700 !important;
            box-shadow: 0 4px 14px rgba(0, 188, 212, 0.39) !important;
            transition: all 0.2s ease !important;
            position: relative !important; /* Kembalikan ke posisi normal */
            width: 100% !important;
            height: auto !important;
            visibility: visible !important;
            color: white !important;
        }

        div.stButton > button[kind="primary"]:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(0, 188, 212, 0.5) !important;
            background: #00acc1 !important;
        }

        /* 6. Summary Card Styling */
        .summary-card {
            background: #1e293b;
            color: white;
            padding: 10px 16px;
            border-radius: 10px;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 1px;
            margin-bottom: 20px;
        }
        </style>
    """, unsafe_allow_html=True)
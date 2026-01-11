import streamlit as st
from streamlit_option_menu import option_menu

def draw_sidebar():
    with st.sidebar:
        # 1. CSS untuk membersihkan Sidebar dari warna gelap bawaan
        st.markdown("""
            <style>
                /* Mengubah background sidebar menjadi putih bersih */
                [data-testid="stSidebar"] {
                    background-color: #ffffff !important;
                    border-right: 1px solid #f0f2f6;
                }
                
                /* Menghilangkan sisa background pada iframe option_menu */
                iframe[title="streamlit_option_menu.option_menu"] {
                    background-color: transparent !important;
                }

                /* Styling teks sidebar bawaan */
                [data-testid="stSidebar"] p {
                    color: #475569 !important;
                }
            </style>
        """, unsafe_allow_html=True)
        
        # 2. Header Logo (Clean & Minimalist)
        st.markdown("""
            <div style="
                padding: 10px 5px 25px 5px;
                display: flex;
                align-items: center;
                gap: 12px;
            ">
                <div style="
                    background: #10b981;
                    width: 40px;
                    height: 40px;
                    border-radius: 10px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: white;
                    font-size: 22px;
                    box-shadow: 0 4px 12px rgba(16, 185, 129, 0.2);
                ">🥬</div>
                <div>
                    <h2 style="
                        color: #1e293b;
                        margin: 0;
                        font-size: 18px;
                        font-weight: 700;
                        letter-spacing: -0.5px;
                    ">LettuceEye AI</h2>
                    <span style="color: #94a3b8; font-size: 11px; font-weight: 500;">Intelligent Agriculture</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # 3. Navigation Menu (White Theme)
        selected = option_menu(
            menu_title=None,
            options=["Dashboard", "Real-time Scan", "Photo Upload", "Video Analysis"],
            icons=['grid-1x2', 'camera', 'image', 'play-circle'],
            default_index=0,
            styles={
                "container": {
                    "padding": "0px",
                    "background-color": "transparent",
                },
                "icon": {
                    "font-size": "16px",
                    "margin-right": "10px"
                },
                "nav-link": {
                    "color": "#64748b",
                    "font-size": "14px",
                    "margin": "5px 0px",
                    "padding": "12px 15px",
                    "border-radius": "10px",
                    "font-family": "sans-serif",
                    "font-weight": "500",
                    "--hover-color": "#f8fafc",
                },
                "nav-link-selected": {
                    "background-color": "#f0fdf4",
                    "color": "#10b981",
                    "font-weight": "600",
                    "border": "1px solid #dcfce7"
                }
            }
        )
        
        # 4. Info Section (Soft Badge style)
        st.markdown("""
            <div style="margin-top: 50px;"></div>
            <div style="
                padding: 15px;
                background: #f8fafc;
                border-radius: 12px;
                border: 1px solid #f1f5f9;
                margin: 0 5px;
            ">
                <p style="
                    color: #64748b;
                    font-size: 12px;
                    margin: 0;
                    line-height: 1.6;
                ">
                    <strong style="color: #334155;">💡 Pro Tip:</strong><br>
                    Gunakan pencahayaan yang cukup saat melakukan pemindaian realtime.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        return selected
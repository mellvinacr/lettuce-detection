import streamlit as st
from streamlit_option_menu import option_menu

def draw_sidebar():
    with st.sidebar:
        st.markdown("""
            <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                <h2 style="color: white; margin: 0;">🥬 LettuceEye AI</h2>
                <small style="color: #64748b;">Disease Detection System</small>
            </div>
        """, unsafe_allow_html=True)
        
        selected = option_menu(
            "",
            ["Dashboard", "Real-time Scan", "Photo Upload", "Video Analysis"],
            icons=['house', 'camera', 'image', 'play-btn'],
            menu_icon="cast", default_index=0,
            styles={
                "container": {"background-color": "transparent"},
                "nav-link": {"color": "#14171a", "font-size": "14px"},
                "nav-link-selected": {"background-color": "rgba(0,188,212,0.2)", "color": "#00bcd4", "border": "1px solid #00bcd4"}
            }
        )
        return selected
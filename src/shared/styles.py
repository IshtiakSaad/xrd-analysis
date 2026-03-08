
import streamlit as st

def inject_custom_css():
    # Force sleek Dark theme for the UI
    bg_main   = "#0E1117"
    bg_sec    = "#1E1E1E"
    text_main = "#E0E0E0"
    text_muted= "#999999"
    accent    = "#FF4B4B"
    border    = "#333333"
    card_shadow = "0 4px 20px rgba(0,0,0,0.3)"
    
    css = f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600&family=EB+Garamond:ital,wght@0,400;0,500;1,400&family=JetBrains+Mono:wght@400;500&display=swap');

      /* Global resets */
      html, body, [data-testid="stAppViewContainer"] {{
        font-family: 'Outfit', sans-serif;
        background-color: {bg_main};
        color: {text_main};
      }}
      
      /* Typography */
      h1, h2, h3 {{ font-family: 'Outfit', sans-serif; letter-spacing: -0.02em; }}
      h1 {{ font-size: 2.8rem !important; font-weight: 600; margin-bottom: 2rem !important; color: {text_main}; }}
      h2 {{ font-size: 1.6rem !important; font-weight: 500; border-bottom: 2px solid {accent}33; padding-bottom: 0.5rem; margin-top: 2.5rem !important; color: {text_main}; }}
      
      /* Academic text styles */
      p, li {{ font-size: 1.1rem; line-height: 1.6; }}

      /* Sidebar Refinement */
      section[data-testid="stSidebar"] {{ 
        background-color: {bg_sec} !important; 
        border-right: 1px solid {border}; 
        box-shadow: 10px 0 30px rgba(0,0,0,0.02);
      }}
      
      /* Card-like components */
      div[data-testid="metric-container"] {{
        background-color: {bg_sec} !important;
        border: 1px solid {border};
        border-radius: 16px;
        padding: 1.2rem;
        box-shadow: {card_shadow};
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
      }}
      div[data-testid="metric-container"]:hover {{
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.08);
        border-color: {accent}55;
      }}
      
      div[data-testid="stMetricValue"] {{
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        color: {accent} !important;
      }}

      /* File Uploader styling */
      div[data-testid="stFileUploader"] {{
        border: 2px dashed {border};
        background-color: {bg_sec};
        border-radius: 16px;
        padding: 2rem;
        transition: all 0.3s ease;
      }}
      div[data-testid="stFileUploader"]:hover {{
        border-color: {accent};
        background-color: {accent}05;
      }}

      /* Tab Styling */
      button[data-baseweb="tab"] {{
        font-family: 'Outfit', sans-serif;
        font-size: 1rem;
        font-weight: 500;
        color: {text_muted};
        transition: all 0.2s ease;
      }}
      button[data-baseweb="tab"][aria-selected="true"] {{
        color: {accent} !important;
        border-bottom-color: {accent} !important;
      }}
      
      /* Expander Styling */
      div[data-testid="stExpander"] {{
        border: 1px solid {border};
        border-radius: 12px;
        background-color: {bg_sec} !important;
        margin-bottom: 1rem;
      }}

      /* Button Styling */
      .stButton > button {{
        border-radius: 8px;
        font-weight: 500;
        padding: 0.5rem 1rem;
        transition: all 0.2s ease;
      }}
      
      /* Figure Captions */
      .fig-caption {{
        font-family: 'EB Garamond', serif;
        font-size: 1rem;
        color: {text_muted};
        text-align: center;
        margin: 1rem 0 3rem 0;
        font-style: italic;
      }}

      /* Hide scientific notation in metrics if any */
      small {{ color: {text_muted}; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

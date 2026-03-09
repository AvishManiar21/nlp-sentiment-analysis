"""Header component for the dashboard."""

import streamlit as st
from utils.theme import inject_custom_css, get_theme_tokens, is_dark_mode


def render_header():
    """Render page header with configuration and theme-aware styling."""
    st.set_page_config(
        page_title="NLP Sentiment Analysis | Amazon Reviews",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    inject_custom_css()
    
    tokens = get_theme_tokens()
    
    st.title("NLP Sentiment Analysis & Opinion Mining")
    st.markdown(
        f"<p style='color: {tokens['text_secondary']}; margin-top: -0.5rem;'>"
        "Analyzing real Amazon product reviews with VADER, TextBlob, "
        "and Machine Learning models (Logistic Regression, Naive Bayes)."
        "</p>",
        unsafe_allow_html=True
    )
    
    st.divider()

"""Header component for the dashboard."""

import streamlit as st
from utils.theme import inject_custom_css


def render_header():
    """Render page header with configuration."""
    st.set_page_config(
        page_title="NLP Sentiment Analysis | Amazon Reviews",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    inject_custom_css()
    
    col1, col2 = st.columns([6, 1])
    
    with col1:
        st.title("NLP Sentiment Analysis & Opinion Mining")
        st.markdown(
            "*Analyzing real Amazon product reviews with VADER, TextBlob, "
            "and Machine Learning models (Logistic Regression, Naive Bayes).*"
        )
    
    with col2:
        st.write("")
    
    st.divider()

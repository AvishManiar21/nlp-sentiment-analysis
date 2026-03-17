"""
Streamlit Dashboard for NLP Sentiment Analysis & Opinion Mining.
Interactive exploration of Amazon product reviews with ML model comparison.
Supports automatic data generation for Streamlit Cloud deployment.
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from components.header import render_header
from components.sidebar import render_sidebar
from components.tabs.overview import render_overview_tab
from components.tabs.categories import render_categories_tab
from components.tabs.aspects import render_aspects_tab
from components.tabs.trends import render_trends_tab
from components.tabs.performance import render_performance_tab
from components.tabs.deep_dive import render_deep_dive_tab
from components.tabs.insights import render_insights_tab
from components.tabs.compare import render_compare_tab
from utils.cache import load_data, load_evaluation_results
from utils.model_storage import ensure_models_available, get_model_info


def main():
    """Main dashboard function."""
    render_header()

    # Ensure deep learning models are available
    # On Streamlit Cloud, this will download models from HuggingFace Hub
    # On local development, this checks if models exist locally
    model_info = get_model_info()

    if not model_info["models_exist_locally"] and model_info["hf_repo_configured"]:
        with st.spinner("Downloading deep learning models from HuggingFace Hub..."):
            st.info(
                f"🤗 Downloading models from: {model_info['hf_repo']}\n\n"
                f"This happens once on first run. Models will be cached for future use."
            )

            # Progress callback for user feedback
            progress_placeholder = st.empty()

            def progress_callback(current, total, filename):
                progress_placeholder.progress(
                    current / total,
                    text=f"Downloading {filename} ({current}/{total})"
                )

            success = ensure_models_available(progress_callback=progress_callback)
            progress_placeholder.empty()

            if success:
                st.success("✓ Models downloaded successfully!")
            else:
                st.warning(
                    "⚠️ Some models could not be downloaded. "
                    "Deep learning model features may be limited."
                )

    df = load_data()
    eval_results = load_evaluation_results()
    
    if len(df) == 0:
        st.error("No review data found. Run `python main.py` to generate data, or check the data folder.")
        st.stop()
    
    filtered_df = render_sidebar(df)
    
    if len(filtered_df) == 0:
        st.warning(
            "No reviews match your filters. Try adjusting the sidebar filters "
            "(Category, Brand, Sentiment, Rating, or Date Range) to see data."
        )
        st.stop()
    
    tabs = st.tabs([
        "Overview",
        "Business Insights",
        "Compare",
        "Categories & Brands",
        "Aspects & Drivers",
        "Trends",
        "Model Performance",
        "Deep Dive",
    ])
    
    with tabs[0]:
        render_overview_tab(filtered_df)
    
    with tabs[1]:
        render_insights_tab(filtered_df)
    
    with tabs[2]:
        render_compare_tab(df)
    
    with tabs[3]:
        render_categories_tab(filtered_df)
    
    with tabs[4]:
        render_aspects_tab(filtered_df)
    
    with tabs[5]:
        render_trends_tab(filtered_df)
    
    with tabs[6]:
        render_performance_tab(filtered_df, eval_results)
    
    with tabs[7]:
        render_deep_dive_tab(filtered_df)


if __name__ == "__main__":
    main()

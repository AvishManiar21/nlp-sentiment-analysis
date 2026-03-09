"""Categories & Brands tab component."""

import streamlit as st
import pandas as pd
from components.charts.category import render_category_analysis, render_brand_comparison


def render_categories_tab(df: pd.DataFrame):
    """Render the Categories & Brands tab content."""
    render_category_analysis(df)
    st.divider()
    render_brand_comparison(df)

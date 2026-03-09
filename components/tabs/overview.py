"""Overview tab component."""

import streamlit as st
import pandas as pd
from components.kpi_cards import render_kpi_row
from components.charts.sentiment import render_sentiment_overview, render_ground_truth_comparison


def render_overview_tab(df: pd.DataFrame):
    """Render the Overview tab content."""
    render_kpi_row(df)
    st.divider()
    render_sentiment_overview(df)
    st.divider()
    render_ground_truth_comparison(df)

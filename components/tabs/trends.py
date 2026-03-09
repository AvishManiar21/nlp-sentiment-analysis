"""Trends tab component."""

import streamlit as st
import pandas as pd
from components.charts.temporal import render_temporal_trends, render_vader_vs_textblob


def render_trends_tab(df: pd.DataFrame):
    """Render the Trends tab content."""
    render_temporal_trends(df)
    st.divider()
    render_vader_vs_textblob(df)

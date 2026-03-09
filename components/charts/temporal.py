"""Temporal analysis chart components."""

import os

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.theme import apply_chart_theme, get_sentiment_colors, CATEGORY_PALETTE


IS_CLOUD_MODE = os.getenv("CLOUD_MODE", "").lower() == "true"


def render_temporal_trends(df: pd.DataFrame):
    """Render sentiment trends over time."""
    st.subheader("Sentiment Trends Over Time")
    
    if "review_date" not in df.columns or df["review_date"].isna().all():
        st.info("No date data available for temporal analysis.")
        return
    
    df_temp = df.copy()
    df_temp = df_temp.dropna(subset=["review_date"])
    df_temp["year_month"] = df_temp["review_date"].dt.to_period("M").astype(str)
    
    if "category" in df_temp.columns:
        temporal = df_temp.groupby(["year_month", "category"]).agg(
            avg_sentiment=("ensemble_score", "mean") if "ensemble_score" in df_temp.columns else ("sentiment_label", "count"),
            count=("sentiment_label", "count"),
        ).reset_index()
        color_col = "category"
    else:
        temporal = df_temp.groupby("year_month").agg(
            avg_sentiment=("ensemble_score", "mean") if "ensemble_score" in df_temp.columns else ("sentiment_label", "count"),
            count=("sentiment_label", "count"),
        ).reset_index()
        color_col = None
    
    if "ensemble_score" in df_temp.columns:
        fig = px.line(
            temporal,
            x="year_month",
            y="avg_sentiment",
            color=color_col,
            title="Monthly Sentiment Trends",
            labels={"year_month": "Month", "avg_sentiment": "Avg Sentiment"},
            color_discrete_sequence=CATEGORY_PALETTE,
        )
        apply_chart_theme(fig, height=450, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)


def render_vader_vs_textblob(df: pd.DataFrame):
    """Render VADER vs TextBlob comparison scatter plot."""
    st.subheader("VADER vs TextBlob Comparison")
    
    if "vader_compound" not in df.columns or "textblob_polarity" not in df.columns:
        st.info("VADER/TextBlob scores not available.")
        return
    
    colors = get_sentiment_colors()
    
    max_points = 1500 if IS_CLOUD_MODE else 3000
    sample = df.sample(min(max_points, len(df)), random_state=42)
    hover_cols = ["rating", "category"] if "category" in sample.columns else ["rating"]
    
    fig = px.scatter(
        sample,
        x="vader_compound",
        y="textblob_polarity",
        color="sentiment_label",
        opacity=0.5,
        title="VADER Compound vs TextBlob Polarity",
        labels={"vader_compound": "VADER Compound", "textblob_polarity": "TextBlob Polarity"},
        color_discrete_map=colors,
        hover_data=hover_cols,
    )
    fig.add_shape(
        type="line",
        x0=-1, y0=-1, x1=1, y1=1,
        line=dict(color=colors["neutral"], dash="dash", width=1)
    )
    fig.update_traces(
        marker=dict(size=6, line=dict(width=0.3, color="white")),
        selector=dict(mode="markers")
    )
    apply_chart_theme(fig, height=500)
    st.plotly_chart(fig, use_container_width=True)

"""Category analysis chart components."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.theme import apply_chart_theme, get_sentiment_colors, CATEGORY_PALETTE


def render_category_sentiment_bar(df: pd.DataFrame):
    """Render average sentiment by category bar chart."""
    if "category" not in df.columns or "ensemble_score" not in df.columns:
        return None
    
    cat_stats = df.groupby("category").agg(
        avg_sentiment=("ensemble_score", "mean"),
        count=("sentiment_label", "count"),
    ).reset_index()
    
    fig = px.bar(
        cat_stats.sort_values("avg_sentiment"),
        x="avg_sentiment",
        y="category",
        orientation="h",
        color="avg_sentiment",
        color_continuous_scale="RdYlGn",
        title="Average Sentiment by Category",
    )
    apply_chart_theme(fig, coloraxis_colorbar=dict(title="Sentiment"))
    return fig


def render_category_breakdown(df: pd.DataFrame):
    """Render sentiment breakdown by category stacked bar chart."""
    if "category" not in df.columns:
        return None
    
    colors = get_sentiment_colors()
    
    cat_stats = df.groupby("category").agg(
        positive_pct=("sentiment_label", lambda x: (x == "positive").mean() * 100),
        negative_pct=("sentiment_label", lambda x: (x == "negative").mean() * 100),
    ).reset_index()
    
    cat_stats["neutral_pct"] = 100 - cat_stats["positive_pct"] - cat_stats["negative_pct"]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Positive",
        x=cat_stats["category"],
        y=cat_stats["positive_pct"],
        marker_color=colors["positive"],
    ))
    fig.add_trace(go.Bar(
        name="Neutral",
        x=cat_stats["category"],
        y=cat_stats["neutral_pct"],
        marker_color=colors["neutral"],
    ))
    fig.add_trace(go.Bar(
        name="Negative",
        x=cat_stats["category"],
        y=cat_stats["negative_pct"],
        marker_color=colors["negative"],
    ))
    
    apply_chart_theme(
        fig,
        barmode="stack",
        title="Sentiment Breakdown by Category",
        yaxis_title="Percentage (%)"
    )
    return fig


def render_category_analysis(df: pd.DataFrame):
    """Render complete category analysis section."""
    st.subheader("Category Analysis")
    
    if "category" not in df.columns:
        st.info("No category data available.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = render_category_sentiment_bar(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = render_category_breakdown(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)


def render_brand_comparison(df: pd.DataFrame):
    """Render brand comparison scatter plot."""
    st.subheader("Brand Comparison")
    
    if "brand" not in df.columns or "category" not in df.columns:
        st.info("No brand/category data available.")
        return
    
    if "ensemble_score" not in df.columns or "rating" not in df.columns:
        st.info("Missing sentiment or rating data.")
        return
    
    brand_stats = df.groupby(["category", "brand"]).agg(
        avg_sentiment=("ensemble_score", "mean"),
        avg_rating=("rating", "mean"),
        count=("sentiment_label", "count"),
    ).reset_index()
    
    brand_stats = brand_stats[brand_stats["count"] >= 20]
    
    if brand_stats.empty:
        st.info("Not enough data for brand comparison (need at least 20 reviews per brand).")
        return
    
    fig = px.scatter(
        brand_stats,
        x="avg_rating",
        y="avg_sentiment",
        color="category",
        size="count",
        hover_name="brand",
        title="Brand Positioning: Rating vs Sentiment",
        labels={"avg_rating": "Average Rating", "avg_sentiment": "Average Sentiment"},
        color_discrete_sequence=CATEGORY_PALETTE,
        size_max=45,
    )
    fig.update_traces(
        marker=dict(line=dict(width=0.5, color="white")),
        selector=dict(mode="markers")
    )
    apply_chart_theme(fig, height=500)
    st.plotly_chart(fig, use_container_width=True)

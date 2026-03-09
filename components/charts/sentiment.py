"""Sentiment-related chart components."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.theme import apply_chart_theme, get_sentiment_colors


def render_sentiment_pie(df: pd.DataFrame):
    """Render sentiment distribution pie chart."""
    colors = get_sentiment_colors()
    counts = df["sentiment_label"].value_counts().reindex(
        ["positive", "neutral", "negative"]
    ).fillna(0)
    
    fig = go.Figure(data=[go.Pie(
        labels=counts.index,
        values=counts.values,
        marker=dict(colors=[colors["positive"], colors["neutral"], colors["negative"]]),
        hole=0.45,
        textinfo="label+percent",
        textfont_size=14,
    )])
    apply_chart_theme(fig, title="Sentiment Split", showlegend=False)
    return fig


def render_sentiment_histogram(df: pd.DataFrame):
    """Render sentiment score distribution histogram."""
    if "ensemble_score" not in df.columns:
        return None
    
    colors = get_sentiment_colors()
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df["ensemble_score"],
        nbinsx=60,
        marker_color=colors["positive"],
        opacity=0.7,
        name="Ensemble Score",
    ))
    fig.add_vline(x=0, line_dash="dash", line_color=colors["negative"], annotation_text="Neutral")
    apply_chart_theme(
        fig,
        title="Sentiment Score Distribution",
        xaxis_title="Ensemble Sentiment Score",
        yaxis_title="Count"
    )
    return fig


def render_sentiment_overview(df: pd.DataFrame):
    """Render sentiment distribution overview section."""
    st.subheader("Sentiment Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        fig = render_sentiment_pie(df)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = render_sentiment_histogram(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)


def render_ground_truth_comparison(df: pd.DataFrame):
    """Render ground truth vs predicted sentiment comparison."""
    if "ground_truth" not in df.columns:
        return
    
    st.subheader("Ground Truth vs Predicted Sentiment")
    
    colors = get_sentiment_colors()
    col1, col2 = st.columns(2)
    
    with col1:
        gt_counts = df["ground_truth"].value_counts().reindex(
            ["positive", "neutral", "negative"]
        ).fillna(0)
        pred_counts = df["sentiment_label"].value_counts().reindex(
            ["positive", "neutral", "negative"]
        ).fillna(0)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Ground Truth (from ratings)",
            x=["positive", "neutral", "negative"],
            y=gt_counts.values,
            marker_color=[colors["positive"], colors["neutral"], colors["negative"]],
            opacity=0.7,
        ))
        fig.add_trace(go.Bar(
            name="Predicted (ensemble)",
            x=["positive", "neutral", "negative"],
            y=pred_counts.values,
            marker_color=[colors["positive"], colors["neutral"], colors["negative"]],
            opacity=1.0,
            marker_pattern_shape="/",
        ))
        apply_chart_theme(fig, title="Label Distribution Comparison", barmode="group")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        from sklearn.metrics import confusion_matrix
        
        labels = ["positive", "neutral", "negative"]
        cm = confusion_matrix(df["ground_truth"], df["sentiment_label"], labels=labels)
        
        fig = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=labels,
            y=labels,
            color_continuous_scale="Blues",
            text_auto=True,
        )
        apply_chart_theme(fig, title="Confusion Matrix (Ensemble)")
        st.plotly_chart(fig, use_container_width=True)

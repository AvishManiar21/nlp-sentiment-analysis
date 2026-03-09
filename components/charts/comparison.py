"""Comparison chart components for side-by-side analysis."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.theme import apply_chart_theme, get_sentiment_colors, CATEGORY_PALETTE


def render_comparison_radar(items_data: list, labels: list):
    """
    Render radar chart comparing multiple items.
    
    Args:
        items_data: List of dicts with metrics for each item
        labels: List of item labels/names
    """
    if not items_data or len(items_data) < 2:
        st.info("Select at least 2 items to compare.")
        return None
    
    categories = ['Avg Rating', 'Positive %', 'Sentiment Score', 'Review Count (scaled)']
    
    fig = go.Figure()
    
    for i, (data, label) in enumerate(zip(items_data, labels)):
        max_count = max(d.get('count', 1) for d in items_data)
        scaled_count = (data.get('count', 0) / max_count) * 5 if max_count > 0 else 0
        
        values = [
            data.get('avg_rating', 0),
            data.get('positive_pct', 0) / 20,
            (data.get('avg_sentiment', 0) + 1) * 2.5,
            scaled_count,
        ]
        values.append(values[0])
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill='toself',
            name=label,
            line_color=CATEGORY_PALETTE[i % len(CATEGORY_PALETTE)],
            opacity=0.7,
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 5])
        ),
        showlegend=True,
        title="Multi-dimensional Comparison",
    )
    apply_chart_theme(fig, height=450)
    
    return fig


def render_comparison_bars(df: pd.DataFrame, group_col: str, selected_items: list):
    """
    Render comparison bar charts for selected items.
    
    Args:
        df: Full DataFrame
        group_col: Column to group by ('category' or 'brand')
        selected_items: List of selected item values to compare
    """
    if len(selected_items) < 2:
        st.info(f"Select at least 2 {group_col}s to compare.")
        return
    
    colors = get_sentiment_colors()
    filtered = df[df[group_col].isin(selected_items)]
    
    stats = filtered.groupby(group_col).agg(
        count=("sentiment_label", "count"),
        avg_rating=("rating", "mean") if "rating" in filtered.columns else ("sentiment_label", "count"),
        avg_sentiment=("ensemble_score", "mean") if "ensemble_score" in filtered.columns else ("sentiment_label", "count"),
        positive_pct=("sentiment_label", lambda x: (x == "positive").mean() * 100),
        neutral_pct=("sentiment_label", lambda x: (x == "neutral").mean() * 100),
        negative_pct=("sentiment_label", lambda x: (x == "negative").mean() * 100),
    ).reset_index()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if "rating" in filtered.columns:
            fig = px.bar(
                stats,
                x=group_col,
                y="avg_rating",
                color=group_col,
                title="Average Rating Comparison",
                color_discrete_sequence=CATEGORY_PALETTE,
            )
            apply_chart_theme(fig, height=350)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if "ensemble_score" in filtered.columns:
            fig = px.bar(
                stats,
                x=group_col,
                y="avg_sentiment",
                color=group_col,
                title="Average Sentiment Comparison",
                color_discrete_sequence=CATEGORY_PALETTE,
            )
            apply_chart_theme(fig, height=350)
            st.plotly_chart(fig, use_container_width=True)
    
    fig = go.Figure()
    for item in selected_items:
        item_data = stats[stats[group_col] == item].iloc[0]
        fig.add_trace(go.Bar(
            name=item,
            x=["Positive", "Neutral", "Negative"],
            y=[item_data["positive_pct"], item_data["neutral_pct"], item_data["negative_pct"]],
        ))
    
    apply_chart_theme(
        fig,
        barmode="group",
        title="Sentiment Distribution Comparison",
        yaxis_title="Percentage (%)",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_aspect_comparison(df: pd.DataFrame, group_col: str, selected_items: list):
    """Render aspect-level comparison between items."""
    if len(selected_items) < 2:
        return
    
    try:
        from src.opinion_miner import extract_aspect_sentiments
    except ImportError:
        st.warning("Opinion miner not available for aspect comparison.")
        return
    
    st.subheader("Aspect-Level Comparison")
    
    aspect_data = {}
    for item in selected_items:
        item_df = df[df[group_col] == item]
        if len(item_df) >= 50:
            try:
                aspects = extract_aspect_sentiments(item_df, dynamic=False)
                if not aspects.empty:
                    aspect_data[item] = aspects.head(8)
            except Exception:
                continue
    
    if len(aspect_data) < 2:
        st.info("Not enough data for aspect comparison (need at least 50 reviews per item).")
        return
    
    all_aspects = set()
    for aspects_df in aspect_data.values():
        all_aspects.update(aspects_df["aspect"].tolist())
    
    common_aspects = list(all_aspects)[:6]
    
    if not common_aspects:
        st.info("No common aspects found for comparison.")
        return
    
    comparison_data = []
    for item, aspects_df in aspect_data.items():
        for aspect in common_aspects:
            aspect_row = aspects_df[aspects_df["aspect"] == aspect]
            if not aspect_row.empty:
                comparison_data.append({
                    "item": item,
                    "aspect": aspect,
                    "sentiment": aspect_row["avg_sentiment"].values[0] if "avg_sentiment" in aspect_row.columns else 0,
                })
    
    if comparison_data:
        comp_df = pd.DataFrame(comparison_data)
        fig = px.bar(
            comp_df,
            x="aspect",
            y="sentiment",
            color="item",
            barmode="group",
            title="Aspect Sentiment Comparison",
            color_discrete_sequence=CATEGORY_PALETTE,
        )
        apply_chart_theme(fig, height=400)
        st.plotly_chart(fig, use_container_width=True)

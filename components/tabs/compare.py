"""Comparison Mode tab component."""

import streamlit as st
import pandas as pd
from components.charts.comparison import render_comparison_radar, render_comparison_bars, render_aspect_comparison
from components.kpi_cards import render_kpi_comparison


def render_compare_tab(df: pd.DataFrame):
    """Render the Comparison Mode tab content."""
    st.subheader("Comparison Mode")
    
    st.markdown("""
    Compare sentiment and ratings across different categories or brands. 
    Select items to analyze side-by-side.
    """)
    
    compare_by = st.radio(
        "Compare by:",
        ["Category", "Brand"],
        horizontal=True,
    )
    
    group_col = "category" if compare_by == "Category" else "brand"
    
    if group_col not in df.columns:
        st.warning(f"No {group_col} data available for comparison.")
        return
    
    available_items = df[group_col].dropna().unique().tolist()
    
    if len(available_items) < 2:
        st.info(f"Need at least 2 {compare_by.lower()}s to compare. Current data has {len(available_items)}.")
        return
    
    item_stats = df.groupby(group_col).agg(
        count=("sentiment_label", "count"),
    ).reset_index()
    
    valid_items = item_stats[item_stats["count"] >= 10][group_col].tolist()
    
    if len(valid_items) < 2:
        st.info(f"Need at least 2 {compare_by.lower()}s with 10+ reviews each.")
        return
    
    selected_items = st.multiselect(
        f"Select {compare_by}s to compare (2-4 recommended):",
        options=valid_items,
        default=valid_items[:min(2, len(valid_items))],
        max_selections=4,
    )
    
    if len(selected_items) < 2:
        st.info(f"Select at least 2 {compare_by.lower()}s to start comparison.")
        return
    
    st.divider()
    
    st.markdown("### Overview Comparison")
    
    items_data = []
    for item in selected_items:
        item_df = df[df[group_col] == item]
        data = {
            "name": item,
            "count": len(item_df),
            "avg_rating": item_df["rating"].mean() if "rating" in item_df.columns else 0,
            "avg_sentiment": item_df["ensemble_score"].mean() if "ensemble_score" in item_df.columns else 0,
            "positive_pct": (item_df["sentiment_label"] == "positive").mean() * 100,
            "negative_pct": (item_df["sentiment_label"] == "negative").mean() * 100,
        }
        items_data.append(data)
    
    cols = st.columns(len(selected_items))
    for i, (item, data) in enumerate(zip(selected_items, items_data)):
        with cols[i]:
            st.markdown(f"#### {item}")
            st.metric("Reviews", f"{data['count']:,}")
            if "rating" in df.columns:
                st.metric("Avg Rating", f"{data['avg_rating']:.2f}")
            if "ensemble_score" in df.columns:
                st.metric("Avg Sentiment", f"{data['avg_sentiment']:.3f}")
            st.metric("Positive %", f"{data['positive_pct']:.1f}%")
    
    st.divider()
    
    st.markdown("### Visual Comparison")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        fig = render_comparison_radar(items_data, selected_items)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Key Metrics")
        comparison_df = pd.DataFrame(items_data)
        comparison_df = comparison_df.rename(columns={
            "name": compare_by,
            "count": "Reviews",
            "avg_rating": "Avg Rating",
            "avg_sentiment": "Avg Sentiment",
            "positive_pct": "Positive %",
            "negative_pct": "Negative %",
        })
        
        display_cols = [compare_by, "Reviews"]
        if "Avg Rating" in comparison_df.columns:
            display_cols.append("Avg Rating")
        if "Avg Sentiment" in comparison_df.columns:
            display_cols.append("Avg Sentiment")
        display_cols.extend(["Positive %", "Negative %"])
        
        st.dataframe(
            comparison_df[display_cols].style.format({
                "Avg Rating": "{:.2f}",
                "Avg Sentiment": "{:.3f}",
                "Positive %": "{:.1f}%",
                "Negative %": "{:.1f}%",
            }).background_gradient(
                cmap="RdYlGn",
                subset=["Positive %"] if "Positive %" in display_cols else []
            ),
            use_container_width=True,
            hide_index=True,
        )
    
    st.divider()
    
    st.markdown("### Detailed Comparison")
    render_comparison_bars(df, group_col, selected_items)
    
    st.divider()
    
    render_aspect_comparison(df, group_col, selected_items)
    
    st.divider()
    
    st.markdown("### Winner Summary")
    
    if items_data:
        best_rating = max(items_data, key=lambda x: x.get("avg_rating", 0))
        best_sentiment = max(items_data, key=lambda x: x.get("avg_sentiment", 0))
        most_positive = max(items_data, key=lambda x: x.get("positive_pct", 0))
        most_reviews = max(items_data, key=lambda x: x.get("count", 0))
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("**Best Rating**")
            st.success(f"🏆 {best_rating['name']}")
            st.caption(f"Rating: {best_rating['avg_rating']:.2f}")
        
        with col2:
            st.markdown("**Best Sentiment**")
            st.success(f"🏆 {best_sentiment['name']}")
            st.caption(f"Score: {best_sentiment['avg_sentiment']:.3f}")
        
        with col3:
            st.markdown("**Most Positive**")
            st.success(f"🏆 {most_positive['name']}")
            st.caption(f"Positive: {most_positive['positive_pct']:.1f}%")
        
        with col4:
            st.markdown("**Most Reviews**")
            st.info(f"📊 {most_reviews['name']}")
            st.caption(f"Count: {most_reviews['count']:,}")

"""KPI metrics cards component."""

import streamlit as st
import pandas as pd


def render_kpi_row(df: pd.DataFrame):
    """Render KPI metrics row with key statistics."""
    if len(df) == 0:
        st.info("No data to display. Adjust filters to see metrics.")
        return
    
    cols = st.columns(6)
    
    with cols[0]:
        st.metric("Total Reviews", f"{len(df):,}")
    
    with cols[1]:
        if "rating" in df.columns:
            st.metric("Avg Rating", f"{df['rating'].mean():.2f}")
    
    with cols[2]:
        if "ensemble_score" in df.columns:
            st.metric("Avg Sentiment", f"{df['ensemble_score'].mean():.3f}")
    
    with cols[3]:
        pos_pct = (df["sentiment_label"] == "positive").mean() * 100
        st.metric("Positive %", f"{pos_pct:.1f}%")
    
    with cols[4]:
        neg_pct = (df["sentiment_label"] == "negative").mean() * 100
        st.metric("Negative %", f"{neg_pct:.1f}%")
    
    with cols[5]:
        if "textblob_subjectivity" in df.columns:
            st.metric("Avg Subjectivity", f"{df['textblob_subjectivity'].mean():.3f}")


def render_kpi_comparison(df1: pd.DataFrame, df2: pd.DataFrame, label1: str, label2: str):
    """Render KPI comparison between two DataFrames."""
    cols = st.columns(4)
    
    def _calc_delta(val1, val2):
        if val2 == 0:
            return 0
        return ((val1 - val2) / val2) * 100
    
    with cols[0]:
        count1, count2 = len(df1), len(df2)
        delta = _calc_delta(count1, count2)
        st.metric(
            "Reviews",
            f"{count1:,} vs {count2:,}",
            delta=f"{delta:+.1f}%" if delta != 0 else None,
        )
    
    with cols[1]:
        if "rating" in df1.columns and "rating" in df2.columns:
            avg1, avg2 = df1['rating'].mean(), df2['rating'].mean()
            delta = avg1 - avg2
            st.metric(
                "Avg Rating",
                f"{avg1:.2f} vs {avg2:.2f}",
                delta=f"{delta:+.2f}" if abs(delta) > 0.01 else None,
            )
    
    with cols[2]:
        if "ensemble_score" in df1.columns and "ensemble_score" in df2.columns:
            sent1, sent2 = df1['ensemble_score'].mean(), df2['ensemble_score'].mean()
            delta = sent1 - sent2
            st.metric(
                "Avg Sentiment",
                f"{sent1:.3f} vs {sent2:.3f}",
                delta=f"{delta:+.3f}" if abs(delta) > 0.001 else None,
            )
    
    with cols[3]:
        pos1 = (df1["sentiment_label"] == "positive").mean() * 100
        pos2 = (df2["sentiment_label"] == "positive").mean() * 100
        delta = pos1 - pos2
        st.metric(
            "Positive %",
            f"{pos1:.1f}% vs {pos2:.1f}%",
            delta=f"{delta:+.1f}%" if abs(delta) > 0.1 else None,
        )

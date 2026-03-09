"""Export functionality for dashboard data."""

import streamlit as st
import pandas as pd
from io import BytesIO
from datetime import datetime


def export_to_csv(df: pd.DataFrame, filename_prefix: str = "sentiment_analysis") -> bytes:
    """Export DataFrame to CSV bytes."""
    return df.to_csv(index=False).encode('utf-8')


def export_to_excel(df: pd.DataFrame, summary_stats: dict = None, filename_prefix: str = "sentiment_analysis") -> bytes:
    """Export DataFrame to Excel with optional summary sheet."""
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Reviews', index=False)
        
        if summary_stats:
            summary_df = pd.DataFrame([summary_stats])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    output.seek(0)
    return output.getvalue()


def get_summary_stats(df: pd.DataFrame) -> dict:
    """Calculate summary statistics for export."""
    stats = {
        "Total Reviews": len(df),
        "Export Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    
    if "rating" in df.columns:
        stats["Average Rating"] = round(df["rating"].mean(), 2)
    
    if "ensemble_score" in df.columns:
        stats["Average Sentiment"] = round(df["ensemble_score"].mean(), 3)
    
    if "sentiment_label" in df.columns:
        sentiment_counts = df["sentiment_label"].value_counts()
        total = len(df)
        stats["Positive %"] = round((sentiment_counts.get("positive", 0) / total) * 100, 1)
        stats["Neutral %"] = round((sentiment_counts.get("neutral", 0) / total) * 100, 1)
        stats["Negative %"] = round((sentiment_counts.get("negative", 0) / total) * 100, 1)
    
    if "category" in df.columns:
        stats["Categories"] = df["category"].nunique()
    
    if "brand" in df.columns:
        stats["Brands"] = df["brand"].nunique()
    
    return stats


def render_export_section(df: pd.DataFrame):
    """Render export buttons in sidebar."""
    st.sidebar.divider()
    st.sidebar.subheader("Export Data")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    csv_data = export_to_csv(df)
    st.sidebar.download_button(
        label="Download CSV",
        data=csv_data,
        file_name=f"sentiment_analysis_{timestamp}.csv",
        mime="text/csv",
        use_container_width=True,
    )
    
    summary_stats = get_summary_stats(df)
    
    try:
        excel_data = export_to_excel(df, summary_stats)
        st.sidebar.download_button(
            label="Download Excel",
            data=excel_data,
            file_name=f"sentiment_analysis_{timestamp}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    except ImportError:
        st.sidebar.info("Install openpyxl for Excel export")
    
    with st.sidebar.expander("Export Summary"):
        for key, value in summary_stats.items():
            st.write(f"**{key}:** {value}")

"""Aspects & Drivers tab component."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.theme import apply_chart_theme, get_sentiment_colors, get_theme_tokens, is_dark_mode

try:
    from wordcloud import WordCloud
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False


def render_aspect_analysis(df: pd.DataFrame):
    """Render aspect-level opinion mining."""
    st.subheader("Aspect-Level Opinion Mining")
    
    try:
        from src.opinion_miner import extract_aspect_sentiments
        aspect_df = extract_aspect_sentiments(df, dynamic=False)
    except Exception as e:
        st.warning(f"Could not perform aspect analysis: {e}")
        return
    
    if aspect_df.empty:
        st.info("Not enough data for aspect analysis with current filters.")
        return
    
    colors = get_sentiment_colors()
    top_aspects = aspect_df.head(12)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if "avg_sentiment" in top_aspects.columns:
            fig = px.bar(
                top_aspects.sort_values("mention_count"),
                x="mention_count",
                y="aspect",
                orientation="h",
                color="avg_sentiment",
                color_continuous_scale="RdYlGn",
                title="Most Discussed Aspects",
            )
        else:
            fig = px.bar(
                top_aspects.sort_values("mention_count"),
                x="mention_count",
                y="aspect",
                orientation="h",
                title="Most Discussed Aspects",
            )
        apply_chart_theme(fig, height=450)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if all(col in top_aspects.columns for col in ["positive_pct", "neutral_pct", "negative_pct"]):
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="Positive",
                y=top_aspects["aspect"],
                x=top_aspects["positive_pct"],
                orientation="h",
                marker_color=colors["positive"],
            ))
            fig.add_trace(go.Bar(
                name="Neutral",
                y=top_aspects["aspect"],
                x=top_aspects["neutral_pct"],
                orientation="h",
                marker_color=colors["neutral"],
            ))
            fig.add_trace(go.Bar(
                name="Negative",
                y=top_aspects["aspect"],
                x=top_aspects["negative_pct"],
                orientation="h",
                marker_color=colors["negative"],
            ))
            apply_chart_theme(
                fig,
                height=450,
                barmode="stack",
                title="Sentiment Breakdown per Aspect",
                xaxis_title="Percentage (%)"
            )
            st.plotly_chart(fig, use_container_width=True)


def render_wordclouds(df: pd.DataFrame):
    """Render word clouds for positive and negative reviews with theme-aware styling."""
    if not WORDCLOUD_AVAILABLE:
        st.info("WordCloud library not available. Install with: pip install wordcloud")
        return
    
    st.subheader("Word Clouds")
    
    tokens = get_theme_tokens()
    dark = is_dark_mode()
    
    bg_color = tokens["bg_secondary"]
    text_color = tokens["text_primary"]
    
    text_col = "review_text" if "review_text" in df.columns else "cleaned_text"
    
    def _render_wc(text, colormap, title):
        if not text.strip():
            st.info(f"No text available for {title.lower()}")
            return
        wc = WordCloud(
            width=800,
            height=380,
            background_color=bg_color,
            colormap=colormap,
            max_words=100,
            collocations=False,
            min_font_size=10,
            max_font_size=120,
        ).generate(text)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10, color=text_color)
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        st.pyplot(fig)
        plt.close(fig)
    
    col1, col2 = st.columns(2)
    
    with col1:
        pos_text = " ".join(
            df[df["sentiment_label"] == "positive"][text_col].dropna().astype(str).tolist()
        )
        _render_wc(pos_text, "Greens", "Positive Reviews — Key Words")
    
    with col2:
        neg_text = " ".join(
            df[df["sentiment_label"] == "negative"][text_col].dropna().astype(str).tolist()
        )
        _render_wc(neg_text, "Reds", "Negative Reviews — Key Words")


def render_aspects_tab(df: pd.DataFrame):
    """Render the Aspects & Drivers tab content."""
    render_aspect_analysis(df)
    st.divider()
    render_wordclouds(df)

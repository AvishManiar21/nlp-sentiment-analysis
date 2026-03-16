"""Sidebar component with filters and controls."""

import streamlit as st
import pandas as pd
from utils.export import render_export_section
from utils.cache import check_dl_models_available

BRAND_EXCLUDE_PATTERNS = [
    "unknown",
    "format: audio cd",
    "format: audio cd library binding",
    "format: mp3 music",
    "format: vinyl",
]
BRAND_MAX_LENGTH = 50
BRAND_TOP_N = 50


def _is_valid_brand(brand):
    """Return True if brand is a real brand (not format/metadata noise)."""
    if pd.isna(brand) or not str(brand).strip():
        return False
    s = str(brand).strip()
    if len(s) > BRAND_MAX_LENGTH:
        return False
    lower = s.lower()
    if lower == "unknown":
        return False
    for pat in BRAND_EXCLUDE_PATTERNS:
        if pat in lower:
            return False
    return True


def _filter_valid_brands(df):
    """Filter df to only rows with valid brands."""
    if "brand" not in df.columns:
        return df
    mask = df["brand"].apply(_is_valid_brand)
    return df[mask].copy()


def _get_filter_brands(df, top_n=BRAND_TOP_N):
    """Get clean brand list for filter."""
    if "brand" not in df.columns:
        return []
    brands = df["brand"].dropna().astype(str).str.strip()
    if brands.empty:
        return []
    mask = brands.str.len() <= BRAND_MAX_LENGTH
    for pat in BRAND_EXCLUDE_PATTERNS:
        mask &= ~brands.str.lower().str.contains(pat, regex=False)
    mask &= (brands != "") & (brands.str.lower() != "unknown")
    valid = brands[mask]
    if valid.empty:
        return []
    top = valid.value_counts().head(top_n)
    return ["All"] + sorted(top.index.tolist())


def render_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    """Render sidebar filters and return filtered DataFrame."""
    st.sidebar.header("Filters")
    st.sidebar.caption("Narrow results by category, brand, sentiment, rating, or date.")
    st.sidebar.divider()
    
    categories = ["All"] + sorted(df["category"].dropna().unique().tolist())
    selected_category = st.sidebar.selectbox("Category", categories)
    
    if "brand" in df.columns:
        if selected_category != "All":
            sub = df[df["category"] == selected_category]
            brands = _get_filter_brands(sub)
        else:
            brands = _get_filter_brands(df)
        selected_brand = st.sidebar.selectbox("Brand", brands) if brands else "All"
    else:
        selected_brand = "All"
    
    sentiments = ["All", "positive", "neutral", "negative"]
    selected_sentiment = st.sidebar.selectbox("Sentiment", sentiments)
    
    if "rating" in df.columns:
        rating_range = st.sidebar.slider("Rating Range", 1, 5, (1, 5))
    else:
        rating_range = (1, 5)
    
    date_range = None
    if "review_date" in df.columns and df["review_date"].notna().any():
        date_min = df["review_date"].min()
        date_max = df["review_date"].max()
        if pd.notna(date_min) and pd.notna(date_max):
            date_range = st.sidebar.date_input(
                "Date Range",
                value=(date_min.date(), date_max.date()),
                min_value=date_min.date(),
                max_value=date_max.date()
            )
    
    filtered = df.copy()
    if selected_category != "All":
        filtered = filtered[filtered["category"] == selected_category]
    if selected_brand != "All" and "brand" in filtered.columns:
        filtered = filtered[filtered["brand"] == selected_brand]
    if selected_sentiment != "All":
        filtered = filtered[filtered["sentiment_label"] == selected_sentiment]
    if "rating" in filtered.columns:
        filtered = filtered[filtered["rating"].between(rating_range[0], rating_range[1])]
    
    if date_range and len(date_range) == 2 and "review_date" in filtered.columns:
        filtered = filtered[
            (filtered["review_date"].dt.date >= date_range[0]) &
            (filtered["review_date"].dt.date <= date_range[1])
        ]
    
    filtered = _filter_valid_brands(filtered)
    
    st.sidebar.divider()
    st.sidebar.metric("Filtered Reviews", f"{len(filtered):,}")
    if len(df) > 0 and len(filtered) != len(df):
        st.sidebar.caption(f"Showing {len(filtered):,} of {len(df):,} total reviews")

    render_export_section(filtered)

    # Deep Learning Models Section
    st.sidebar.divider()
    st.sidebar.subheader("🧠 Deep Learning")

    dl_models = check_dl_models_available()

    if dl_models:
        st.sidebar.success(f"✓ {len(dl_models)} DL model(s) trained")

        with st.sidebar.expander("View DL Models"):
            for model in dl_models:
                st.markdown(f"""
                **{model['name']}**
                - Framework: {model['framework']}
                - Type: {model['type']}
                """)
    else:
        st.sidebar.info("No DL models trained yet")
        with st.sidebar.expander("Train DL Models"):
            st.code("""python main.py --train-dl \\
  --use-embeddings""", language="bash")

    return filtered

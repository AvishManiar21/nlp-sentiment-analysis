"""Deep Dive tab component for sample reviews."""

import streamlit as st
import pandas as pd


def render_sample_reviews(df: pd.DataFrame):
    """Render sample reviews section."""
    st.subheader("Sample Reviews")
    
    if len(df) == 0:
        st.info("No reviews to display. Adjust filters to see sample reviews.")
        return
    
    text_col = "review_text" if "review_text" in df.columns else "cleaned_text"
    product_col = "product" if "product" in df.columns else None
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Most Positive**")
        if "ensemble_score" in df.columns and len(df) > 0:
            top_pos = df.nlargest(5, "ensemble_score")
            for _, row in top_pos.iterrows():
                product_name = row.get(product_col, "Review")[:30] if product_col else "Review"
                title = f"⭐ {row['rating']} | {product_name}... ({row['ensemble_score']:.3f})"
                with st.expander(title):
                    st.write(row[text_col][:500] if pd.notna(row[text_col]) else "No text")
    
    with col2:
        st.markdown("**Most Neutral**")
        if "ensemble_score" in df.columns and len(df) > 0:
            df_copy = df.copy()
            df_copy["abs_score"] = df_copy["ensemble_score"].abs()
            top_neu = df_copy.nsmallest(5, "abs_score")
            for _, row in top_neu.iterrows():
                product_name = row.get(product_col, "Review")[:30] if product_col else "Review"
                title = f"⭐ {row['rating']} | {product_name}... ({row['ensemble_score']:.3f})"
                with st.expander(title):
                    st.write(row[text_col][:500] if pd.notna(row[text_col]) else "No text")
    
    with col3:
        st.markdown("**Most Negative**")
        if "ensemble_score" in df.columns and len(df) > 0:
            top_neg = df.nsmallest(5, "ensemble_score")
            for _, row in top_neg.iterrows():
                product_name = row.get(product_col, "Review")[:30] if product_col else "Review"
                title = f"⭐ {row['rating']} | {product_name}... ({row['ensemble_score']:.3f})"
                with st.expander(title):
                    st.write(row[text_col][:500] if pd.notna(row[text_col]) else "No text")


def render_review_search(df: pd.DataFrame):
    """Render review search functionality."""
    st.subheader("Search Reviews")
    
    if len(df) == 0:
        st.info("No reviews to search. Adjust filters first.")
        return
    
    text_col = "review_text" if "review_text" in df.columns else "cleaned_text"
    
    search_query = st.text_input("Search for keywords in reviews", placeholder="e.g., battery, shipping, quality")
    
    if search_query:
        mask = df[text_col].str.contains(search_query, case=False, na=False)
        results = df[mask]
        
        st.write(f"Found **{len(results):,}** reviews containing '{search_query}'")
        
        if len(results) == 0:
            st.caption("Try different keywords or broaden your filters.")
        elif len(results) > 0:
            for _, row in results.head(10).iterrows():
                sentiment_emoji = "🟢" if row["sentiment_label"] == "positive" else (
                    "🔴" if row["sentiment_label"] == "negative" else "🟡"
                )
                with st.expander(f"{sentiment_emoji} Rating: {row['rating']} | {row[text_col][:60]}..."):
                    st.write(row[text_col])
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Sentiment", row["sentiment_label"])
                    with col2:
                        if "ensemble_score" in df.columns:
                            st.metric("Score", f"{row['ensemble_score']:.3f}")
                    with col3:
                        if "category" in df.columns:
                            st.metric("Category", row["category"])


def render_deep_dive_tab(df: pd.DataFrame):
    """Render the Deep Dive tab content."""
    render_sample_reviews(df)
    st.divider()
    render_review_search(df)

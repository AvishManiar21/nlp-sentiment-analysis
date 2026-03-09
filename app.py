"""
Streamlit Dashboard for NLP Sentiment Analysis & Opinion Mining.
Interactive exploration of Amazon product reviews with ML model comparison.
Supports automatic data generation for Streamlit Cloud deployment.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent))

try:
    from wordcloud import WordCloud
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "outputs"
RESULTS_DIR = Path(__file__).parent / "results"
MODELS_DIR = Path(__file__).parent / "models"

CLOUD_SAMPLE_SIZE = 30000

COLORS = {
    "positive": "#22c55e",
    "negative": "#ef4444",
    "neutral": "#94a3b8",
}
CATEGORY_PALETTE = ["#3b82f6", "#8b5cf6", "#22c55e", "#f59e0b", "#ec4899", "#06b6d4"]
MODEL_PALETTE = ["#3b82f6", "#22c55e", "#ef4444", "#f59e0b", "#8b5cf6", "#06b6d4"]

# Unified Plotly chart theme
PLOTLY_THEME = dict(
    template="plotly_white",
    font=dict(family="Inter, system-ui, sans-serif", size=12, color="#334155"),
    title_font=dict(size=16, color="#1e293b"),
    paper_bgcolor="white",
    plot_bgcolor="#f8fafc",
    margin=dict(t=60, b=50, l=50, r=30),
    hoverlabel=dict(bgcolor="white", font_size=12, font_family="Inter"),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="#e2e8f0",
    ),
    xaxis=dict(showgrid=True, gridcolor="#e2e8f0", zeroline=False),
    yaxis=dict(showgrid=True, gridcolor="#e2e8f0", zeroline=False),
    colorway=CATEGORY_PALETTE,
)


def _apply_chart_theme(fig, height=420, **overrides):
    """Apply unified theme to a Plotly figure. Overrides merge with theme."""
    layout = {**PLOTLY_THEME, "height": height, **overrides}
    fig.update_layout(**layout)
    return fig


def generate_data_for_cloud(sample_size=CLOUD_SAMPLE_SIZE):
    """
    Generate data for Streamlit Cloud deployment.
    Downloads from HuggingFace, preprocesses, and runs sentiment analysis.
    """
    import nltk
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    st.info(f"Downloading {sample_size:,} Amazon reviews from HuggingFace...")
    
    from src.data_loader import load_amazon_reviews
    df = load_amazon_reviews(
        sample_size=sample_size,
        output_path=DATA_DIR / "amazon_reviews.csv",
        force_reload=True,
        verbose=False,
    )
    
    st.info("Preprocessing text data...")
    
    from src.preprocessor import preprocess_dataframe
    df = preprocess_dataframe(df, verbose=False)
    df.to_csv(DATA_DIR / "preprocessed_reviews.csv", index=False)
    
    st.info("Running sentiment analysis (VADER + TextBlob)...")
    
    from src.sentiment_analyzer import run_sentiment_analysis
    df = run_sentiment_analysis(df, verbose=False)
    df.to_csv(DATA_DIR / "reviews_with_sentiment.csv", index=False)
    
    st.info("Training ML model...")
    
    from src.ml_models import train_model, evaluate_model, save_model, prepare_data
    
    X_train, X_test, y_train, y_test = prepare_data(
        df, 
        text_column="processed_text", 
        label_column="ground_truth",
        test_size=0.2
    )
    
    pipeline = train_model(X_train, y_train, "logistic_regression", verbose=False)
    results = evaluate_model(pipeline, X_test, y_test, "Logistic Regression")
    results["pipeline"] = pipeline
    save_model(pipeline, "logistic_regression", MODELS_DIR)
    
    st.info("Generating evaluation results...")
    
    from src.model_evaluator import compare_all_models, save_evaluation_results
    comparison = compare_all_models(df, ml_results={"logistic_regression": results}, verbose=False)
    save_evaluation_results(comparison, RESULTS_DIR)
    
    st.success(f"Data generation complete! Processed {len(df):,} reviews.")
    
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    """Load analyzed review data. Generate if not found (for cloud deployment)."""
    path = DATA_DIR / "reviews_with_sentiment.csv"
    
    if not path.exists():
        with st.spinner("First run: Downloading and processing data (this may take 2-3 minutes)..."):
            generate_data_for_cloud()
    
    df = pd.read_csv(path)
    
    if "review_date" in df.columns:
        df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
    
    if "ground_truth" not in df.columns and "rating" in df.columns:
        df["ground_truth"] = df["rating"].apply(
            lambda r: "negative" if r <= 2 else ("neutral" if r == 3 else "positive")
        )
    
    return df


@st.cache_data
def load_evaluation_results():
    """Load model evaluation results."""
    summary_path = RESULTS_DIR / "evaluation_summary.json"
    comparison_path = RESULTS_DIR / "evaluation_comparison.csv"
    
    results = {}
    
    if summary_path.exists():
        with open(summary_path, "r") as f:
            results["summary"] = json.load(f)
    
    if comparison_path.exists():
        results["comparison"] = pd.read_csv(comparison_path)
    
    return results if results else None


def render_header():
    """Render page header."""
    st.set_page_config(
        page_title="NLP Sentiment Analysis | Amazon Reviews",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("NLP Sentiment Analysis & Opinion Mining")
    st.markdown(
        "*Analyzing real Amazon product reviews with VADER, TextBlob, "
        "and Machine Learning models (Logistic Regression, Naive Bayes).*"
    )
    st.divider()


# Non-brands to exclude (McAuley store field often has format/metadata for Digital Music)
BRAND_EXCLUDE_PATTERNS = [
    "unknown",
    "format: audio cd",
    "format: audio cd library binding",
    "format: mp3 music",
    "format: vinyl",
]
BRAND_MAX_LENGTH = 50  # Exclude very long strings (artist lists, etc.)
BRAND_TOP_N = 50  # Show top N brands by review count in dropdown


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
    """Filter df to only rows with valid brands. Keeps rows without brand column."""
    if "brand" not in df.columns:
        return df
    mask = df["brand"].apply(_is_valid_brand)
    return df[mask].copy()


def _get_filter_brands(df, top_n=BRAND_TOP_N):
    """Get clean brand list for filter: exclude noise, limit to top N by count."""
    if "brand" not in df.columns:
        return []
    brands = df["brand"].dropna().astype(str).str.strip()
    if brands.empty:
        return []
    # Exclude known non-brands
    mask = brands.str.len() <= BRAND_MAX_LENGTH
    for pat in BRAND_EXCLUDE_PATTERNS:
        mask &= ~brands.str.lower().str.contains(pat, regex=False)
    mask &= (brands != "") & (brands.str.lower() != "unknown")
    valid = brands[mask]
    if valid.empty:
        return []
    # Top N by count
    top = valid.value_counts().head(top_n)
    return ["All"] + sorted(top.index.tolist())


def render_sidebar(df):
    """Render sidebar filters."""
    st.sidebar.header("Filters")
    
    categories = ["All"] + sorted(df["category"].dropna().unique().tolist())
    selected_category = st.sidebar.selectbox("Category", categories)
    
    if "brand" in df.columns:
        if selected_category != "All":
            sub = df[df["category"] == selected_category]
            brands = _get_filter_brands(sub)
        else:
            brands = _get_filter_brands(df)
        selected_brand = st.sidebar.selectbox("Brand", brands)
    else:
        selected_brand = "All"
    
    sentiments = ["All", "positive", "neutral", "negative"]
    selected_sentiment = st.sidebar.selectbox("Sentiment", sentiments)
    
    if "rating" in df.columns:
        rating_range = st.sidebar.slider("Rating Range", 1, 5, (1, 5))
    else:
        rating_range = (1, 5)
    
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
        else:
            date_range = None
    else:
        date_range = None
    
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
    st.sidebar.metric("Filtered Reviews", f"{len(filtered):,}")
    return filtered


def render_kpi_row(df):
    """Render KPI metrics row."""
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


def render_sentiment_overview(df):
    """Render sentiment distribution overview."""
    st.subheader("Sentiment Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        counts = df["sentiment_label"].value_counts().reindex(["positive", "neutral", "negative"]).fillna(0)
        fig = go.Figure(data=[go.Pie(
            labels=counts.index,
            values=counts.values,
            marker=dict(colors=[COLORS["positive"], COLORS["neutral"], COLORS["negative"]]),
            hole=0.45,
            textinfo="label+percent",
            textfont_size=14,
        )])
        _apply_chart_theme(fig, title="Sentiment Split", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if "ensemble_score" in df.columns:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df["ensemble_score"], nbinsx=60,
                marker_color=COLORS["positive"], opacity=0.7,
                name="Ensemble Score",
            ))
            fig.add_vline(x=0, line_dash="dash", line_color="#ef4444", annotation_text="Neutral")
            _apply_chart_theme(fig, title="Sentiment Score Distribution",
                xaxis_title="Ensemble Sentiment Score", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)


def render_ground_truth_comparison(df):
    """Render ground truth vs predicted sentiment comparison."""
    if "ground_truth" not in df.columns:
        return
    
    st.subheader("Ground Truth vs Predicted Sentiment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gt_counts = df["ground_truth"].value_counts().reindex(["positive", "neutral", "negative"]).fillna(0)
        pred_counts = df["sentiment_label"].value_counts().reindex(["positive", "neutral", "negative"]).fillna(0)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Ground Truth (from ratings)",
            x=["positive", "neutral", "negative"],
            y=gt_counts.values,
            marker_color=[COLORS["positive"], COLORS["neutral"], COLORS["negative"]],
            opacity=0.7,
        ))
        fig.add_trace(go.Bar(
            name="Predicted (ensemble)",
            x=["positive", "neutral", "negative"],
            y=pred_counts.values,
            marker_color=[COLORS["positive"], COLORS["neutral"], COLORS["negative"]],
            opacity=1.0,
            marker_pattern_shape="/",
        ))
        _apply_chart_theme(fig, title="Label Distribution Comparison", barmode="group")
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
        _apply_chart_theme(fig, title="Confusion Matrix (Ensemble)")
        st.plotly_chart(fig, use_container_width=True)


def render_category_analysis(df):
    """Render category analysis."""
    st.subheader("Category Analysis")
    
    if "category" not in df.columns:
        st.info("No category data available.")
        return
    
    cat_stats = df.groupby("category").agg(
        avg_sentiment=("ensemble_score", "mean") if "ensemble_score" in df.columns else ("sentiment_label", "count"),
        avg_rating=("rating", "mean") if "rating" in df.columns else ("sentiment_label", "count"),
        count=("sentiment_label", "count"),
        positive_pct=("sentiment_label", lambda x: (x == "positive").mean() * 100),
        negative_pct=("sentiment_label", lambda x: (x == "negative").mean() * 100),
    ).reset_index()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if "ensemble_score" in df.columns:
            fig = px.bar(
                cat_stats.sort_values("avg_sentiment"),
                x="avg_sentiment", y="category",
                orientation="h",
                color="avg_sentiment",
                color_continuous_scale="RdYlGn",
                title="Average Sentiment by Category",
            )
            _apply_chart_theme(fig, coloraxis_colorbar=dict(title="Sentiment"))
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Positive",
            x=cat_stats["category"],
            y=cat_stats["positive_pct"],
            marker_color=COLORS["positive"],
        ))
        neutral_pct = 100 - cat_stats["positive_pct"] - cat_stats["negative_pct"]
        fig.add_trace(go.Bar(
            name="Neutral",
            x=cat_stats["category"],
            y=neutral_pct,
            marker_color=COLORS["neutral"],
        ))
        fig.add_trace(go.Bar(
            name="Negative",
            x=cat_stats["category"],
            y=cat_stats["negative_pct"],
            marker_color=COLORS["negative"],
        ))
        _apply_chart_theme(fig, barmode="stack", title="Sentiment Breakdown by Category",
            yaxis_title="Percentage (%)")
        st.plotly_chart(fig, use_container_width=True)


def render_temporal_trends(df):
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
        _apply_chart_theme(fig, height=450, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)


def render_aspect_analysis(df):
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
    
    top_aspects = aspect_df.head(12)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if "avg_sentiment" in top_aspects.columns:
            fig = px.bar(
                top_aspects.sort_values("mention_count"),
                x="mention_count", y="aspect",
                orientation="h",
                color="avg_sentiment",
                color_continuous_scale="RdYlGn",
                title="Most Discussed Aspects",
            )
        else:
            fig = px.bar(
                top_aspects.sort_values("mention_count"),
                x="mention_count", y="aspect",
                orientation="h",
                title="Most Discussed Aspects",
            )
        _apply_chart_theme(fig, height=450)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if all(col in top_aspects.columns for col in ["positive_pct", "neutral_pct", "negative_pct"]):
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="Positive", y=top_aspects["aspect"],
                x=top_aspects["positive_pct"],
                orientation="h", marker_color=COLORS["positive"],
            ))
            fig.add_trace(go.Bar(
                name="Neutral", y=top_aspects["aspect"],
                x=top_aspects["neutral_pct"],
                orientation="h", marker_color=COLORS["neutral"],
            ))
            fig.add_trace(go.Bar(
                name="Negative", y=top_aspects["aspect"],
                x=top_aspects["negative_pct"],
                orientation="h", marker_color=COLORS["negative"],
            ))
            _apply_chart_theme(fig, height=450, barmode="stack",
                title="Sentiment Breakdown per Aspect", xaxis_title="Percentage (%)")
            st.plotly_chart(fig, use_container_width=True)


def render_brand_comparison(df):
    """Render brand comparison."""
    st.subheader("Brand Comparison")
    
    if "brand" not in df.columns or "category" not in df.columns:
        st.info("No brand/category data available.")
        return
    
    brand_stats = df.groupby(["category", "brand"]).agg(
        avg_sentiment=("ensemble_score", "mean") if "ensemble_score" in df.columns else ("sentiment_label", "count"),
        avg_rating=("rating", "mean") if "rating" in df.columns else ("sentiment_label", "count"),
        count=("sentiment_label", "count"),
    ).reset_index()
    
    brand_stats = brand_stats[brand_stats["count"] >= 20]
    
    if "ensemble_score" in df.columns and "rating" in df.columns:
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
        fig.update_traces(marker=dict(line=dict(width=0.5, color="white")), selector=dict(mode="markers"))
        _apply_chart_theme(fig, height=500)
        st.plotly_chart(fig, use_container_width=True)


def render_wordclouds(df):
    """Render word clouds."""
    if not WORDCLOUD_AVAILABLE:
        st.info("WordCloud library not available.")
        return
    
    st.subheader("Word Clouds")
    
    col1, col2 = st.columns(2)
    
    text_col = "review_text" if "review_text" in df.columns else "cleaned_text"
    
    def _render_wc(text, colormap, title):
        if not text.strip():
            return
        wc = WordCloud(
            width=800, height=380, background_color="#fafafa",
            colormap=colormap, max_words=100, collocations=False,
            min_font_size=10, max_font_size=120,
        ).generate(text)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
        fig.patch.set_facecolor("white")
        st.pyplot(fig)
        plt.close(fig)
    
    with col1:
        pos_text = " ".join(df[df["sentiment_label"] == "positive"][text_col].dropna().astype(str).tolist())
        _render_wc(pos_text, "Greens", "Positive Reviews — Key Words")
    
    with col2:
        neg_text = " ".join(df[df["sentiment_label"] == "negative"][text_col].dropna().astype(str).tolist())
        _render_wc(neg_text, "Reds", "Negative Reviews — Key Words")


def render_vader_vs_textblob(df):
    """Render VADER vs TextBlob comparison."""
    st.subheader("VADER vs TextBlob Comparison")
    
    if "vader_compound" not in df.columns or "textblob_polarity" not in df.columns:
        st.info("VADER/TextBlob scores not available.")
        return
    
    sample = df.sample(min(3000, len(df)), random_state=42)
    hover_cols = ["rating", "category"] if "category" in sample.columns else ["rating"]
    fig = px.scatter(
        sample,
        x="vader_compound",
        y="textblob_polarity",
        color="sentiment_label",
        opacity=0.5,
        title="VADER Compound vs TextBlob Polarity",
        labels={"vader_compound": "VADER Compound", "textblob_polarity": "TextBlob Polarity"},
        color_discrete_map=COLORS,
        hover_data=hover_cols,
    )
    fig.add_shape(type="line", x0=-1, y0=-1, x1=1, y1=1,
                  line=dict(color="#94a3b8", dash="dash", width=1))
    fig.update_traces(marker=dict(size=6, line=dict(width=0.3, color="white")), selector=dict(mode="markers"))
    _apply_chart_theme(fig, height=500)
    st.plotly_chart(fig, use_container_width=True)


def render_sample_reviews(df):
    """Render sample reviews."""
    st.subheader("Sample Reviews")
    
    text_col = "review_text" if "review_text" in df.columns else "cleaned_text"
    product_col = "product" if "product" in df.columns else None
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Most Positive**")
        if "ensemble_score" in df.columns:
            top_pos = df.nlargest(5, "ensemble_score")[[product_col, text_col, "rating", "ensemble_score"] if product_col else [text_col, "rating", "ensemble_score"]]
            for _, row in top_pos.iterrows():
                title = f"⭐ {row['rating']} | {row.get(product_col, 'Review')[:30]}... ({row['ensemble_score']:.3f})"
                with st.expander(title):
                    st.write(row[text_col][:500])
    
    with col2:
        st.markdown("**Most Neutral**")
        if "ensemble_score" in df.columns:
            df_copy = df.copy()
            df_copy["abs_score"] = df_copy["ensemble_score"].abs()
            top_neu = df_copy.nsmallest(5, "abs_score")
            for _, row in top_neu.iterrows():
                title = f"⭐ {row['rating']} | {row.get(product_col, 'Review')[:30]}... ({row['ensemble_score']:.3f})"
                with st.expander(title):
                    st.write(row[text_col][:500])
    
    with col3:
        st.markdown("**Most Negative**")
        if "ensemble_score" in df.columns:
            top_neg = df.nsmallest(5, "ensemble_score")
            for _, row in top_neg.iterrows():
                title = f"⭐ {row['rating']} | {row.get(product_col, 'Review')[:30]}... ({row['ensemble_score']:.3f})"
                with st.expander(title):
                    st.write(row[text_col][:500])


def render_model_performance(df, eval_results):
    """Render model performance comparison tab."""
    st.subheader("Model Performance Comparison")
    
    if not eval_results:
        st.info(
            "Model evaluation results not found. "
            "Run `python main.py` to train models and generate evaluation metrics."
        )
        return
    
    if "comparison" in eval_results:
        comparison_df = eval_results["comparison"]
        
        st.markdown("### Performance Metrics")
        
        metric_cols = ["Accuracy", "F1 (weighted)", "F1 (macro)", "Precision", "Recall"]
        available_cols = ["Model"] + [c for c in metric_cols if c in comparison_df.columns]
        
        st.dataframe(
            comparison_df[available_cols].style.format({
                col: "{:.4f}" for col in available_cols if col != "Model"
            }).background_gradient(cmap="RdYlGn", subset=[c for c in available_cols if c != "Model"]),
            use_container_width=True,
        )
        
        st.markdown("### Accuracy Comparison")
        
        fig = px.bar(
            comparison_df,
            x="Model",
            y="Accuracy",
            color="Accuracy",
            color_continuous_scale="RdYlGn",
            title="Model Accuracy",
        )
        _apply_chart_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### F1 Score Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if "F1 (weighted)" in comparison_df.columns:
                fig = px.bar(
                    comparison_df,
                    x="Model",
                    y="F1 (weighted)",
                    color="F1 (weighted)",
                    color_continuous_scale="RdYlGn",
                    title="Weighted F1 Score",
                )
                _apply_chart_theme(fig, height=350)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if "F1 (macro)" in comparison_df.columns:
                fig = px.bar(
                    comparison_df,
                    x="Model",
                    y="F1 (macro)",
                    color="F1 (macro)",
                    color_continuous_scale="RdYlGn",
                    title="Macro F1 Score",
                )
                _apply_chart_theme(fig, height=350)
                st.plotly_chart(fig, use_container_width=True)
    
    if "summary" in eval_results:
        summary = eval_results["summary"]
        
        if "best_models" in summary:
            st.markdown("### Best Models by Metric")
            
            best_df = pd.DataFrame([
                {"Metric": metric, "Best Model": info["model"], "Score": info["value"]}
                for metric, info in summary["best_models"].items()
            ])
            
            st.dataframe(
                best_df.style.format({"Score": "{:.4f}"}),
                use_container_width=True,
            )
    
    st.markdown("### Confusion Matrices")
    st.info(
        "Confusion matrix images are saved in the `outputs/` directory. "
        "Check `confusion_matrix_*.png` files."
    )


def main():
    """Main dashboard function."""
    render_header()
    df = load_data()
    eval_results = load_evaluation_results()
    filtered_df = render_sidebar(df)
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Overview", "Categories & Brands", "Aspects & Drivers",
        "Trends", "Model Performance", "Deep Dive"
    ])
    
    with tab1:
        render_kpi_row(filtered_df)
        st.divider()
        render_sentiment_overview(filtered_df)
        st.divider()
        render_ground_truth_comparison(filtered_df)
    
    with tab2:
        render_category_analysis(filtered_df)
        st.divider()
        render_brand_comparison(filtered_df)
    
    with tab3:
        render_aspect_analysis(filtered_df)
        st.divider()
        render_wordclouds(filtered_df)
    
    with tab4:
        render_temporal_trends(filtered_df)
        st.divider()
        render_vader_vs_textblob(filtered_df)
    
    with tab5:
        render_model_performance(filtered_df, eval_results)
    
    with tab6:
        render_sample_reviews(filtered_df)


if __name__ == "__main__":
    main()

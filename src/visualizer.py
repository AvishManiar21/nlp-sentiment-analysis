"""
Visualization module for sentiment analysis results.
Generates static charts (matplotlib/seaborn), interactive plots (plotly),
and model evaluation visualizations (confusion matrices, comparisons, ROC).
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from wordcloud import WordCloud
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

sns.set_theme(style="whitegrid", palette="husl", font_scale=1.1)

COLORS = {
    "positive": "#2ecc71",
    "negative": "#e74c3c",
    "neutral": "#95a5a6",
    "primary": "#3498db",
    "secondary": "#9b59b6",
    "accent": "#f39c12",
}
CATEGORY_PALETTE = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22", "#34495e", "#16a085", "#c0392b", "#8e44ad", "#27ae60"]
MODEL_PALETTE = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6", "#1abc9c"]


def save_fig(fig, name, output_dir, dpi=150):
    """Save matplotlib figure to file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_sentiment_distribution(df, output_dir):
    """Plot overall sentiment distribution charts."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    colors = [COLORS["positive"], COLORS["neutral"], COLORS["negative"]]
    label_order = ["positive", "neutral", "negative"]
    counts = df["sentiment_label"].value_counts().reindex(label_order).fillna(0)
    
    axes[0].bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=1.5)
    axes[0].set_title("Sentiment Distribution", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Number of Reviews")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + max(counts.values) * 0.02, f"{int(v):,}", ha="center", fontweight="bold")
    
    if "ensemble_score" in df.columns:
        axes[1].hist(df["ensemble_score"].dropna(), bins=60, color=COLORS["primary"],
                     alpha=0.7, edgecolor="white")
        axes[1].axvline(x=0, color="red", linestyle="--", alpha=0.7, label="Neutral")
        axes[1].set_title("Ensemble Score Distribution", fontsize=14, fontweight="bold")
        axes[1].set_xlabel("Sentiment Score")
        axes[1].set_ylabel("Frequency")
        axes[1].legend()
    
    if "rating" in df.columns and "ensemble_score" in df.columns:
        rating_sentiment = df.groupby("rating")["ensemble_score"].mean()
        bar_colors = [COLORS["negative"] if r <= 2 else COLORS["neutral"] if r == 3 else COLORS["positive"]
                      for r in rating_sentiment.index]
        axes[2].bar(rating_sentiment.index, rating_sentiment.values,
                    color=bar_colors, edgecolor="white", linewidth=1.5)
        axes[2].set_title("Avg Sentiment by Rating", fontsize=14, fontweight="bold")
        axes[2].set_xlabel("Star Rating")
        axes[2].set_ylabel("Avg Sentiment Score")
    
    fig.suptitle("Overall Sentiment Analysis", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    return save_fig(fig, "sentiment_distribution", output_dir)


def plot_category_comparison(cat_summary, output_dir):
    """Plot category comparison charts."""
    if cat_summary.empty:
        return None
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    if "avg_sentiment" in cat_summary.columns:
        cat_summary_sorted = cat_summary.sort_values("avg_sentiment", ascending=True)
        colors = [COLORS["positive"] if v > 0 else COLORS["negative"]
                  for v in cat_summary_sorted["avg_sentiment"]]
        axes[0].barh(cat_summary_sorted["category"], cat_summary_sorted["avg_sentiment"],
                     color=colors, edgecolor="white")
        axes[0].set_title("Avg Sentiment by Category", fontsize=14, fontweight="bold")
        axes[0].set_xlabel("Average Sentiment Score")
        axes[0].axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    
    if all(col in cat_summary.columns for col in ["positive_pct", "neutral_pct", "negative_pct"]):
        cat_stacked = cat_summary[["category", "positive_pct", "neutral_pct", "negative_pct"]]
        cat_stacked = cat_stacked.set_index("category")
        cat_stacked.plot(kind="barh", stacked=True, ax=axes[1],
                         color=[COLORS["positive"], COLORS["neutral"], COLORS["negative"]])
        axes[1].set_title("Sentiment Breakdown by Category", fontsize=14, fontweight="bold")
        axes[1].set_xlabel("Percentage (%)")
        axes[1].legend(["Positive", "Neutral", "Negative"], loc="lower right")
    
    if all(col in cat_summary.columns for col in ["avg_rating", "avg_sentiment", "total_reviews"]):
        sizes = cat_summary["total_reviews"] / cat_summary["total_reviews"].max() * 500
        axes[2].scatter(cat_summary["avg_rating"], cat_summary["avg_sentiment"],
                        s=sizes, c=CATEGORY_PALETTE[:len(cat_summary)], alpha=0.7, edgecolors="black")
        for _, row in cat_summary.iterrows():
            axes[2].annotate(row["category"], (row["avg_rating"], row["avg_sentiment"]),
                             fontsize=9, ha="center", va="bottom")
        axes[2].set_title("Rating vs Sentiment (size = volume)", fontsize=14, fontweight="bold")
        axes[2].set_xlabel("Average Rating")
        axes[2].set_ylabel("Average Sentiment")
    
    fig.tight_layout()
    return save_fig(fig, "category_comparison", output_dir)


def plot_aspect_analysis(aspect_df, output_dir):
    """Plot aspect sentiment analysis charts."""
    if aspect_df.empty:
        return None
    
    top_aspects = aspect_df.head(15).copy()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    top_aspects_sorted = top_aspects.sort_values("mention_count")
    if "avg_sentiment" in top_aspects_sorted.columns:
        colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in top_aspects_sorted["avg_sentiment"]]
    else:
        colors = ["#3498db"] * len(top_aspects_sorted)
    
    axes[0].barh(top_aspects_sorted["aspect"], top_aspects_sorted["mention_count"], color=colors)
    axes[0].set_title("Most Discussed Aspects\n(Green = Positive, Red = Negative)",
                       fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Mention Count")
    
    if all(col in top_aspects.columns for col in ["positive_pct", "neutral_pct", "negative_pct"]):
        aspects_pivot = top_aspects[["aspect", "positive_pct", "neutral_pct", "negative_pct"]].set_index("aspect")
        aspects_pivot = aspects_pivot.sort_values("positive_pct")
        aspects_pivot.plot(kind="barh", stacked=True, ax=axes[1],
                           color=[COLORS["positive"], COLORS["neutral"], COLORS["negative"]])
        axes[1].set_title("Sentiment Breakdown per Aspect", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("Percentage (%)")
        axes[1].legend(["Positive", "Neutral", "Negative"], loc="lower right")
    
    fig.tight_layout()
    return save_fig(fig, "aspect_analysis", output_dir)


def plot_temporal_trends(temporal_df, output_dir):
    """Plot sentiment trends over time."""
    if temporal_df.empty:
        return None, None
    
    color_col = "category" if "category" in temporal_df.columns else None
    y_col = "avg_sentiment" if "avg_sentiment" in temporal_df.columns else "review_count"
    
    fig = px.line(
        temporal_df,
        x="year_month",
        y=y_col,
        color=color_col,
        title="Sentiment Trends Over Time",
        labels={"year_month": "Month", y_col: "Average Sentiment Score" if y_col == "avg_sentiment" else "Review Count"},
        color_discrete_sequence=CATEGORY_PALETTE,
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        template="plotly_white",
        font=dict(size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500,
    )
    output_path = Path(output_dir) / "temporal_trends.html"
    fig.write_html(str(output_path))
    return output_path, fig


def plot_brand_heatmap(brand_df, output_dir):
    """Plot brand sentiment heatmaps."""
    if brand_df.empty or "brand" not in brand_df.columns:
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    if "avg_sentiment" in brand_df.columns and "category" in brand_df.columns:
        pivot_sentiment = brand_df.pivot_table(
            index="brand", columns="category", values="avg_sentiment", aggfunc="mean"
        )
        pivot_sentiment = pivot_sentiment.dropna(how="all")
        if not pivot_sentiment.empty:
            sns.heatmap(pivot_sentiment, annot=True, fmt=".3f", cmap="RdYlGn", center=0,
                        ax=axes[0], linewidths=0.5)
            axes[0].set_title("Average Sentiment by Brand & Category", fontsize=14, fontweight="bold")
    
    if "avg_rating" in brand_df.columns and "category" in brand_df.columns:
        pivot_rating = brand_df.pivot_table(
            index="brand", columns="category", values="avg_rating", aggfunc="mean"
        )
        pivot_rating = pivot_rating.dropna(how="all")
        if not pivot_rating.empty:
            sns.heatmap(pivot_rating, annot=True, fmt=".2f", cmap="YlOrRd", ax=axes[1],
                        linewidths=0.5, vmin=1, vmax=5)
            axes[1].set_title("Average Rating by Brand & Category", fontsize=14, fontweight="bold")
    
    fig.tight_layout()
    return save_fig(fig, "brand_heatmap", output_dir)


def plot_wordclouds(df, output_dir, text_column="review_text"):
    """Generate word clouds for positive and negative reviews."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    if "sentiment_label" not in df.columns:
        return None
    
    pos_texts = df[df["sentiment_label"] == "positive"][text_column].dropna().astype(str).tolist()
    neg_texts = df[df["sentiment_label"] == "negative"][text_column].dropna().astype(str).tolist()
    
    pos_text = " ".join(pos_texts) if pos_texts else ""
    neg_text = " ".join(neg_texts) if neg_texts else ""
    
    if pos_text.strip():
        wc_pos = WordCloud(
            width=800, height=400, background_color="white",
            colormap="Greens", max_words=100, collocations=False,
        ).generate(pos_text)
        axes[0].imshow(wc_pos, interpolation="bilinear")
        axes[0].set_title("Positive Reviews - Key Words", fontsize=14, fontweight="bold")
        axes[0].axis("off")
    
    if neg_text.strip():
        wc_neg = WordCloud(
            width=800, height=400, background_color="white",
            colormap="Reds", max_words=100, collocations=False,
        ).generate(neg_text)
        axes[1].imshow(wc_neg, interpolation="bilinear")
        axes[1].set_title("Negative Reviews - Key Words", fontsize=14, fontweight="bold")
        axes[1].axis("off")
    
    fig.suptitle("Word Clouds: Positive vs Negative Sentiment", fontsize=16, fontweight="bold")
    fig.tight_layout()
    return save_fig(fig, "wordclouds", output_dir)


def plot_drivers(positive_drivers, negative_drivers, output_dir):
    """Plot sentiment driver analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    if not positive_drivers.empty:
        top_pos = positive_drivers.head(15).sort_values("tfidf_score")
        axes[0].barh(top_pos["phrase"], top_pos["tfidf_score"], color=COLORS["positive"])
        axes[0].set_title("Top Positive Sentiment Drivers", fontsize=14, fontweight="bold")
        axes[0].set_xlabel("TF-IDF Importance Score")
    
    if not negative_drivers.empty:
        top_neg = negative_drivers.head(15).sort_values("tfidf_score")
        axes[1].barh(top_neg["phrase"], top_neg["tfidf_score"], color=COLORS["negative"])
        axes[1].set_title("Top Negative Sentiment Drivers", fontsize=14, fontweight="bold")
        axes[1].set_xlabel("TF-IDF Importance Score")
    
    fig.tight_layout()
    return save_fig(fig, "sentiment_drivers", output_dir)


def plot_rating_vs_sentiment_scatter(df, output_dir):
    """Plot VADER vs TextBlob scatter comparison."""
    if "vader_compound" not in df.columns or "textblob_polarity" not in df.columns:
        return None, None
    
    sample = df.sample(min(5000, len(df)), random_state=42)
    
    color_col = "sentiment_label" if "sentiment_label" in sample.columns else None
    symbol_col = "category" if "category" in sample.columns else None
    
    fig = px.scatter(
        sample,
        x="vader_compound",
        y="textblob_polarity",
        color=color_col,
        symbol=symbol_col,
        opacity=0.5,
        title="VADER vs TextBlob Sentiment Scores",
        labels={"vader_compound": "VADER Compound Score", "textblob_polarity": "TextBlob Polarity"},
        color_discrete_map=COLORS if color_col else None,
        hover_data=["product", "rating"] if "product" in sample.columns else None,
    )
    fig.update_layout(
        template="plotly_white",
        height=600,
        font=dict(size=12),
    )
    output_path = Path(output_dir) / "vader_vs_textblob.html"
    fig.write_html(str(output_path))
    return output_path, fig


def plot_confusion_matrix(conf_matrix, labels, model_name, output_dir):
    """Plot confusion matrix heatmap for a model."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar_kws={"label": "Count"},
    )
    
    ax.set_title(f"Confusion Matrix: {model_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    
    fig.tight_layout()
    
    safe_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    return save_fig(fig, f"confusion_matrix_{safe_name}", output_dir)


def plot_all_confusion_matrices(evaluation_results, output_dir):
    """Plot confusion matrices for all evaluated models."""
    paths = {}
    
    model_results = evaluation_results.get("model_results", {})
    
    for model_key, results in model_results.items():
        conf_matrix = results.get("confusion_matrix")
        labels = results.get("labels", [])
        model_name = results.get("model_name", model_key)
        
        if conf_matrix is not None and len(labels) > 0:
            path = plot_confusion_matrix(conf_matrix, labels, model_name, output_dir)
            paths[model_key] = path
    
    return paths


def plot_model_comparison(evaluation_results, output_dir):
    """Plot model comparison bar chart."""
    comparison_df = evaluation_results.get("comparison_df")
    
    if comparison_df is None or comparison_df.empty:
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    metrics_to_plot = ["Accuracy", "F1 (weighted)", "F1 (macro)"]
    available_metrics = [m for m in metrics_to_plot if m in comparison_df.columns]
    
    if available_metrics:
        x = np.arange(len(comparison_df))
        width = 0.25
        
        for i, metric in enumerate(available_metrics):
            offset = (i - len(available_metrics) / 2 + 0.5) * width
            bars = axes[0].bar(x + offset, comparison_df[metric], width, 
                              label=metric, color=MODEL_PALETTE[i])
            
            for bar in bars:
                height = bar.get_height()
                axes[0].annotate(f'{height:.3f}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3), textcoords="offset points",
                                ha='center', va='bottom', fontsize=8)
        
        axes[0].set_xlabel("Model")
        axes[0].set_ylabel("Score")
        axes[0].set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(comparison_df["Model"], rotation=45, ha="right")
        axes[0].legend()
        axes[0].set_ylim(0, 1.1)
    
    if "Precision" in comparison_df.columns and "Recall" in comparison_df.columns:
        for i, (_, row) in enumerate(comparison_df.iterrows()):
            axes[1].scatter(row["Recall"], row["Precision"], 
                           s=200, c=[MODEL_PALETTE[i % len(MODEL_PALETTE)]],
                           label=row["Model"], edgecolors="black", linewidth=1.5)
        
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        axes[1].set_title("Precision vs Recall", fontsize=14, fontweight="bold")
        axes[1].legend(loc="best")
        axes[1].set_xlim(0, 1.05)
        axes[1].set_ylim(0, 1.05)
        axes[1].plot([0, 1], [0, 1], "k--", alpha=0.3)
    
    fig.tight_layout()
    return save_fig(fig, "model_comparison", output_dir)


def plot_per_class_f1(evaluation_results, output_dir):
    """Plot per-class F1 scores for all models."""
    model_results = evaluation_results.get("model_results", {})
    
    if not model_results:
        return None
    
    data = []
    
    for model_key, results in model_results.items():
        model_name = results.get("model_name", model_key)
        report = results.get("classification_report", {})
        labels = results.get("labels", [])
        
        for label in labels:
            if str(label) in report:
                data.append({
                    "Model": model_name,
                    "Class": str(label),
                    "F1": report[str(label)].get("f1-score", 0),
                })
    
    if not data:
        return None
    
    df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    classes = df["Class"].unique()
    models = df["Model"].unique()
    x = np.arange(len(classes))
    width = 0.8 / len(models)
    
    for i, model in enumerate(models):
        model_data = df[df["Model"] == model]
        f1_scores = [model_data[model_data["Class"] == c]["F1"].values[0] 
                     if c in model_data["Class"].values else 0 for c in classes]
        offset = (i - len(models) / 2 + 0.5) * width
        ax.bar(x + offset, f1_scores, width, label=model, color=MODEL_PALETTE[i % len(MODEL_PALETTE)])
    
    ax.set_xlabel("Sentiment Class")
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 Scores by Model", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.1)
    
    fig.tight_layout()
    return save_fig(fig, "per_class_f1", output_dir)


def plot_ground_truth_distribution(df, output_dir, label_column="ground_truth"):
    """Plot ground truth label distribution."""
    if label_column not in df.columns:
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    label_order = ["positive", "neutral", "negative"]
    counts = df[label_column].value_counts().reindex(label_order).fillna(0)
    colors = [COLORS["positive"], COLORS["neutral"], COLORS["negative"]]
    
    axes[0].bar(counts.index, counts.values, color=colors, edgecolor="white")
    axes[0].set_title("Ground Truth Distribution", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + max(counts.values) * 0.02, f"{int(v):,}", ha="center", fontweight="bold")
    
    if "rating" in df.columns:
        rating_counts = df["rating"].value_counts().sort_index()
        rating_colors = [COLORS["negative"], COLORS["negative"], COLORS["neutral"],
                         COLORS["positive"], COLORS["positive"]]
        axes[1].bar(rating_counts.index, rating_counts.values, 
                   color=rating_colors[:len(rating_counts)], edgecolor="white")
        axes[1].set_title("Rating Distribution (Source of Ground Truth)", fontsize=14, fontweight="bold")
        axes[1].set_xlabel("Star Rating")
        axes[1].set_ylabel("Count")
    
    fig.tight_layout()
    return save_fig(fig, "ground_truth_distribution", output_dir)


def generate_all_visualizations(df, mining_results, output_dir, evaluation_results=None):
    """
    Generate all visualizations for the analysis.
    
    Args:
        df: DataFrame with sentiment analysis results
        mining_results: Dictionary from opinion mining
        output_dir: Directory to save visualizations
        evaluation_results: Optional model evaluation results
    
    Returns:
        Dictionary with paths to all generated files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    paths = {}
    
    print("Generating sentiment distribution plots...")
    paths["sentiment_dist"] = plot_sentiment_distribution(df, output_dir)
    
    if "ground_truth" in df.columns:
        print("Generating ground truth distribution...")
        paths["ground_truth"] = plot_ground_truth_distribution(df, output_dir)
    
    print("Generating category comparison plots...")
    cat_summary = mining_results.get("category_summary", pd.DataFrame())
    result = plot_category_comparison(cat_summary, output_dir)
    if result:
        paths["category_comp"] = result
    
    print("Generating aspect analysis plots...")
    aspect_df = mining_results.get("aspect_sentiments", pd.DataFrame())
    result = plot_aspect_analysis(aspect_df, output_dir)
    if result:
        paths["aspect_analysis"] = result
    
    print("Generating temporal trend plots...")
    temporal_df = mining_results.get("temporal_trends", pd.DataFrame())
    result = plot_temporal_trends(temporal_df, output_dir)
    if result[0]:
        paths["temporal"] = result[0]
    
    print("Generating brand heatmap...")
    brand_df = mining_results.get("brand_summary", pd.DataFrame())
    result = plot_brand_heatmap(brand_df, output_dir)
    if result:
        paths["brand_heatmap"] = result
    
    print("Generating word clouds...")
    result = plot_wordclouds(df, output_dir)
    if result:
        paths["wordclouds"] = result
    
    print("Generating driver analysis plots...")
    pos_drivers = mining_results.get("positive_drivers", pd.DataFrame())
    neg_drivers = mining_results.get("negative_drivers", pd.DataFrame())
    paths["drivers"] = plot_drivers(pos_drivers, neg_drivers, output_dir)
    
    print("Generating VADER vs TextBlob scatter...")
    result = plot_rating_vs_sentiment_scatter(df, output_dir)
    if result[0]:
        paths["scatter"] = result[0]
    
    if evaluation_results:
        print("Generating model evaluation visualizations...")
        
        cm_paths = plot_all_confusion_matrices(evaluation_results, output_dir)
        paths["confusion_matrices"] = cm_paths
        
        result = plot_model_comparison(evaluation_results, output_dir)
        if result:
            paths["model_comparison"] = result
        
        result = plot_per_class_f1(evaluation_results, output_dir)
        if result:
            paths["per_class_f1"] = result
    
    print(f"All visualizations saved to {output_dir}/")
    return paths


if __name__ == "__main__":
    data_path = Path(__file__).parent.parent / "data" / "reviews_with_sentiment.csv"
    output_dir = Path(__file__).parent.parent / "outputs"
    
    if data_path.exists():
        print(f"Loading data from {data_path}")
        df = pd.read_csv(data_path, parse_dates=["review_date"])
        
        from .opinion_miner import run_opinion_mining
        mining_results = run_opinion_mining(df, verbose=True)
        
        paths = generate_all_visualizations(df, mining_results, output_dir)
        
        print("\nGenerated files:")
        for name, path in paths.items():
            print(f"  {name}: {path}")
    else:
        print(f"Data file not found: {data_path}")
        print("Run the main pipeline first.")

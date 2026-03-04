"""
Sentiment analysis pipeline using VADER and TextBlob.
Produces composite sentiment scores, polarity labels, and subjectivity metrics.
Robust handling for real-world text data (NaN, encoding issues, long reviews).
"""

import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import nltk
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

MAX_TEXT_LENGTH = 10000
MIN_TEXT_LENGTH = 3


def ensure_nltk_data():
    """Ensure required NLTK resources are downloaded."""
    resources = [
        ("sentiment/vader_lexicon", "vader_lexicon"),
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


def sanitize_text(text):
    """
    Sanitize text for sentiment analysis.
    Handles NaN, encoding issues, and length constraints.
    """
    if pd.isna(text) or text is None:
        return ""
    
    text = str(text)
    
    text = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    
    text = "".join(char for char in text if char.isprintable() or char in "\n\t")
    
    if len(text) > MAX_TEXT_LENGTH:
        text = text[:MAX_TEXT_LENGTH]
    
    text = text.strip()
    
    return text


def analyze_vader(text, analyzer):
    """
    Analyze sentiment using VADER.
    Returns compound, positive, negative, neutral scores.
    """
    text = sanitize_text(text)
    
    if len(text) < MIN_TEXT_LENGTH:
        return 0.0, 0.0, 0.0, 1.0
    
    try:
        scores = analyzer.polarity_scores(text)
        return (
            float(scores["compound"]),
            float(scores["pos"]),
            float(scores["neg"]),
            float(scores["neu"]),
        )
    except Exception:
        return 0.0, 0.0, 0.0, 1.0


def analyze_textblob(text):
    """
    Analyze sentiment using TextBlob.
    Returns polarity (-1 to 1) and subjectivity (0 to 1).
    """
    text = sanitize_text(text)
    
    if len(text) < MIN_TEXT_LENGTH:
        return 0.0, 0.5
    
    try:
        blob = TextBlob(text)
        polarity = float(blob.sentiment.polarity)
        subjectivity = float(blob.sentiment.subjectivity)
        
        polarity = max(-1.0, min(1.0, polarity))
        subjectivity = max(0.0, min(1.0, subjectivity))
        
        return polarity, subjectivity
    except Exception:
        return 0.0, 0.5


def classify_sentiment(compound_score):
    """Classify sentiment based on ensemble score."""
    if compound_score >= 0.05:
        return "positive"
    elif compound_score <= -0.05:
        return "negative"
    return "neutral"


def sentiment_strength(compound_score):
    """Determine sentiment strength from score magnitude."""
    abs_score = abs(compound_score)
    if abs_score >= 0.6:
        return "strong"
    elif abs_score >= 0.3:
        return "moderate"
    return "weak"


def compute_ensemble_score(vader_compound, textblob_polarity, vader_weight=0.65):
    """
    Compute ensemble sentiment score combining VADER and TextBlob.
    VADER is weighted higher as it's designed for social media text.
    """
    textblob_weight = 1.0 - vader_weight
    ensemble = (vader_weight * vader_compound) + (textblob_weight * textblob_polarity)
    return max(-1.0, min(1.0, ensemble))


def analyze_sentiment_batch(texts, analyzer, batch_size=1000):
    """
    Analyze sentiment for a batch of texts.
    More memory-efficient for large datasets.
    """
    vader_results = []
    textblob_results = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        
        for text in batch:
            compound, pos, neg, neu = analyze_vader(text, analyzer)
            vader_results.append({
                "vader_compound": compound,
                "vader_positive": pos,
                "vader_negative": neg,
                "vader_neutral": neu,
            })
        
        for text in batch:
            polarity, subjectivity = analyze_textblob(text)
            textblob_results.append({
                "textblob_polarity": polarity,
                "textblob_subjectivity": subjectivity,
            })
    
    return vader_results, textblob_results


def run_sentiment_analysis(df, text_column="review_text", verbose=True):
    """
    Run full sentiment analysis pipeline on DataFrame.
    
    Args:
        df: DataFrame containing review text
        text_column: Name of the column containing text to analyze
        verbose: Show progress bars
    
    Returns:
        DataFrame with sentiment columns added:
        - vader_compound, vader_positive, vader_negative, vader_neutral
        - textblob_polarity, textblob_subjectivity
        - ensemble_score, sentiment_label, sentiment_strength
        - rating_sentiment_gap (if 'rating' column exists)
    """
    ensure_nltk_data()
    sia = SentimentIntensityAnalyzer()
    
    df = df.copy()
    
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame")
    
    texts = df[text_column].tolist()
    
    if verbose:
        print("Running VADER sentiment analysis...")
    
    vader_results = []
    iterator = tqdm(texts, desc="VADER") if verbose else texts
    for text in iterator:
        compound, pos, neg, neu = analyze_vader(text, sia)
        vader_results.append({
            "vader_compound": compound,
            "vader_positive": pos,
            "vader_negative": neg,
            "vader_neutral": neu,
        })
    
    if verbose:
        print("Running TextBlob sentiment analysis...")
    
    textblob_results = []
    iterator = tqdm(texts, desc="TextBlob") if verbose else texts
    for text in iterator:
        polarity, subjectivity = analyze_textblob(text)
        textblob_results.append({
            "textblob_polarity": polarity,
            "textblob_subjectivity": subjectivity,
        })
    
    vader_df = pd.DataFrame(vader_results)
    textblob_df = pd.DataFrame(textblob_results)
    
    result = pd.concat([df.reset_index(drop=True), vader_df, textblob_df], axis=1)
    
    result["ensemble_score"] = result.apply(
        lambda row: compute_ensemble_score(
            row["vader_compound"], 
            row["textblob_polarity"]
        ), 
        axis=1
    )
    
    result["sentiment_label"] = result["ensemble_score"].apply(classify_sentiment)
    result["sentiment_strength"] = result["ensemble_score"].apply(sentiment_strength)
    
    if "rating" in result.columns:
        result["rating_sentiment_gap"] = (
            (result["rating"] - 3) / 2 - result["ensemble_score"]
        ).abs()
    
    if verbose:
        print(f"\nSentiment analysis complete for {len(result):,} reviews.")
        print(f"\nSentiment distribution:")
        print(result["sentiment_label"].value_counts().to_string())
        
        print(f"\nSentiment strength distribution:")
        print(result["sentiment_strength"].value_counts().to_string())
    
    return result


def get_sentiment_summary(df):
    """Get summary statistics for sentiment analysis results."""
    summary = {
        "total_reviews": len(df),
        "avg_vader_compound": df["vader_compound"].mean(),
        "avg_textblob_polarity": df["textblob_polarity"].mean(),
        "avg_ensemble_score": df["ensemble_score"].mean(),
        "avg_subjectivity": df["textblob_subjectivity"].mean(),
        "sentiment_distribution": df["sentiment_label"].value_counts().to_dict(),
        "strength_distribution": df["sentiment_strength"].value_counts().to_dict(),
    }
    
    if "rating" in df.columns:
        summary["avg_rating_sentiment_gap"] = df["rating_sentiment_gap"].mean()
    
    return summary


def predict_sentiment_vader(texts, threshold=0.05):
    """Predict sentiment labels using VADER only."""
    ensure_nltk_data()
    sia = SentimentIntensityAnalyzer()
    
    labels = []
    for text in texts:
        compound, _, _, _ = analyze_vader(text, sia)
        if compound >= threshold:
            labels.append("positive")
        elif compound <= -threshold:
            labels.append("negative")
        else:
            labels.append("neutral")
    
    return labels


def predict_sentiment_textblob(texts, threshold=0.05):
    """Predict sentiment labels using TextBlob only."""
    labels = []
    for text in texts:
        polarity, _ = analyze_textblob(text)
        if polarity >= threshold:
            labels.append("positive")
        elif polarity <= -threshold:
            labels.append("negative")
        else:
            labels.append("neutral")
    
    return labels


if __name__ == "__main__":
    from pathlib import Path
    
    data_path = Path(__file__).parent.parent / "data" / "amazon_reviews.csv"
    
    if data_path.exists():
        print(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        
        if len(df) > 1000:
            print(f"Using sample of 1000 reviews for quick test...")
            df = df.sample(n=1000, random_state=42)
        
        result = run_sentiment_analysis(df, verbose=True)
        
        summary = get_sentiment_summary(result)
        print("\n" + "=" * 50)
        print("Sentiment Summary")
        print("=" * 50)
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        out_path = data_path.parent / "reviews_with_sentiment_test.csv"
        result.to_csv(out_path, index=False)
        print(f"\nSaved test results to: {out_path}")
    else:
        print(f"Data file not found: {data_path}")
        print("Run data_loader.py first to fetch Amazon reviews.")

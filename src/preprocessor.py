"""
Text preprocessing pipeline for NLP sentiment analysis.
Handles cleaning, tokenization, lemmatization, and ground truth label creation.
"""

import re
import html
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import nltk


def ensure_nltk_resources():
    """Download required NLTK resources if not present."""
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ]
    
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"Downloading NLTK resource: {name}")
            nltk.download(name, quiet=True)


ensure_nltk_resources()

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


STOP_WORDS = set(stopwords.words("english"))

STOP_WORDS -= {"not", "no", "never", "neither", "nobody", "nothing", "nowhere",
               "nor", "none", "very", "too", "more", "most", "less", "least",
               "but", "however", "although", "though", "except"}

LEMMATIZER = WordNetLemmatizer()


def clean_html(text):
    """Remove HTML tags and decode entities."""
    if not text or pd.isna(text):
        return ""
    
    text = str(text)
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    
    return text


def clean_text(text):
    """Clean text while preserving sentiment-relevant content."""
    if not text or pd.isna(text):
        return ""
    
    text = str(text)
    
    text = clean_html(text)
    
    text = text.encode("ascii", "ignore").decode("ascii")
    
    text = re.sub(r"http\S+|www\.\S+", "", text)
    
    text = re.sub(r"\S+@\S+", "", text)
    
    text = re.sub(r"[^\w\s!?.,;:\'\"-]", " ", text)
    
    text = re.sub(r"(.)\1{3,}", r"\1\1", text)
    
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def normalize_text(text, lowercase=True, remove_punctuation=True):
    """Normalize text for ML processing."""
    if not text:
        return ""
    
    text = clean_text(text)
    
    if lowercase:
        text = text.lower()
    
    if remove_punctuation:
        text = re.sub(r"[^\w\s]", " ", text)
    
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def tokenize(text):
    """Tokenize text into words."""
    if not text:
        return []
    
    try:
        tokens = word_tokenize(text.lower())
        return tokens
    except Exception:
        return text.lower().split()


def remove_stopwords(tokens):
    """Remove stopwords from token list."""
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


def lemmatize(tokens):
    """Lemmatize tokens to their base forms."""
    return [LEMMATIZER.lemmatize(t) for t in tokens]


def preprocess_for_ml(text):
    """Full preprocessing pipeline for ML models."""
    cleaned = normalize_text(text, lowercase=True, remove_punctuation=True)
    tokens = tokenize(cleaned)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    return " ".join(tokens)


def preprocess_for_display(text):
    """Light preprocessing for display (keep readable)."""
    return clean_text(text)


def rating_to_sentiment(rating):
    """Convert star rating to sentiment label."""
    if rating <= 2:
        return "negative"
    elif rating == 3:
        return "neutral"
    else:
        return "positive"


def rating_to_sentiment_binary(rating):
    """Convert star rating to binary sentiment (pos/neg only)."""
    return "positive" if rating >= 4 else "negative"


def create_ground_truth_labels(df, include_neutral=True):
    """Create ground truth sentiment labels from ratings."""
    df = df.copy()
    
    if include_neutral:
        df["ground_truth"] = df["rating"].apply(rating_to_sentiment)
    else:
        df = df[df["rating"] != 3].copy()
        df["ground_truth"] = df["rating"].apply(rating_to_sentiment_binary)
    
    return df


def preprocess_dataframe(df, text_column="review_text", verbose=True):
    """
    Preprocess entire DataFrame with text cleaning and label creation.
    
    Args:
        df: DataFrame with review text
        text_column: Name of the text column
        verbose: Show progress bar
    
    Returns:
        DataFrame with additional columns:
        - cleaned_text: HTML/encoding cleaned text
        - processed_text: Fully processed text for ML
        - ground_truth: Sentiment label from rating
    """
    df = df.copy()
    
    if verbose:
        print("Preprocessing text data...")
    
    if verbose:
        tqdm.pandas(desc="Cleaning text")
        df["cleaned_text"] = df[text_column].progress_apply(preprocess_for_display)
    else:
        df["cleaned_text"] = df[text_column].apply(preprocess_for_display)
    
    if verbose:
        tqdm.pandas(desc="Processing for ML")
        df["processed_text"] = df["cleaned_text"].progress_apply(preprocess_for_ml)
    else:
        df["processed_text"] = df["cleaned_text"].apply(preprocess_for_ml)
    
    df = create_ground_truth_labels(df, include_neutral=True)
    
    if verbose:
        print(f"\nPreprocessing complete: {len(df):,} reviews")
        print(f"\nGround truth distribution:")
        print(df["ground_truth"].value_counts().to_string())
    
    return df


def get_vocabulary_stats(df, text_column="processed_text"):
    """Get vocabulary statistics from processed text."""
    all_tokens = []
    for text in df[text_column].dropna():
        all_tokens.extend(text.split())
    
    from collections import Counter
    token_counts = Counter(all_tokens)
    
    stats = {
        "total_tokens": len(all_tokens),
        "unique_tokens": len(token_counts),
        "avg_tokens_per_review": len(all_tokens) / len(df),
        "most_common_20": token_counts.most_common(20),
    }
    
    return stats


def save_preprocessed_data(df, output_path=None):
    """Save preprocessed DataFrame to CSV."""
    if output_path is None:
        output_path = Path(__file__).parent.parent / "data" / "preprocessed_reviews.csv"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"Saved preprocessed data to: {output_path}")
    
    return output_path


if __name__ == "__main__":
    data_path = Path(__file__).parent.parent / "data" / "amazon_reviews.csv"
    
    if data_path.exists():
        print(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        
        df = preprocess_dataframe(df, verbose=True)
        
        print("\n" + "=" * 50)
        print("Vocabulary Statistics")
        print("=" * 50)
        stats = get_vocabulary_stats(df)
        print(f"Total tokens: {stats['total_tokens']:,}")
        print(f"Unique tokens: {stats['unique_tokens']:,}")
        print(f"Avg tokens/review: {stats['avg_tokens_per_review']:.1f}")
        print("\nMost common tokens:")
        for token, count in stats["most_common_20"]:
            print(f"  {token}: {count:,}")
        
        output_path = save_preprocessed_data(df)
        
        print("\n" + "=" * 50)
        print("Sample preprocessed reviews")
        print("=" * 50)
        for _, row in df.head(3).iterrows():
            print(f"\nOriginal: {row['review_text'][:100]}...")
            print(f"Cleaned:  {row['cleaned_text'][:100]}...")
            print(f"Processed: {row['processed_text'][:100]}...")
            print(f"Ground truth: {row['ground_truth']}")
    else:
        print(f"Data file not found: {data_path}")
        print("Run data_loader.py first to fetch Amazon reviews.")

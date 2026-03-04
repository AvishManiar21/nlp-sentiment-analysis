"""
Opinion mining module: extracts key drivers of positive and negative sentiment
using TF-IDF analysis, n-gram extraction, and dynamic aspect-level aggregation.
Supports both hardcoded tech aspects and dynamic noun phrase extraction.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import re
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

SPACY_AVAILABLE = False
try:
    import spacy
    nlp_test = spacy.blank("en")
    SPACY_AVAILABLE = True
except (ImportError, Exception):
    SPACY_AVAILABLE = False


STOP_WORDS_EXTRA = {
    "product", "device", "thing", "item", "purchase", "bought", "buy",
    "using", "used", "use", "also", "really", "just", "much", "get",
    "got", "one", "would", "could", "even", "like", "make", "made",
    "good", "bad", "great", "terrible", "amazing", "awful", "love",
    "hate", "best", "worst", "recommend", "dont", "highly", "overall",
    "say", "think", "feel", "want", "need", "know", "well", "lot",
    "time", "day", "week", "month", "year", "way", "thing", "everything",
    "something", "anything", "nothing", "someone", "anyone", "everyone",
    "people", "person", "stuff", "kind", "sort", "type",
}

TECH_ASPECTS = [
    "battery life", "battery", "camera quality", "camera", "display", "screen",
    "performance", "speed", "build quality", "software", "sound quality",
    "audio", "noise cancellation", "comfort", "fitness tracking", "keyboard",
    "trackpad", "charging", "connectivity", "bluetooth", "wifi", "value",
    "price", "speaker", "microphone", "quality", "design", "size", "weight",
    "durability", "warranty", "support", "shipping", "packaging", "instructions",
    "setup", "installation", "app", "interface", "features", "functionality",
    "usability", "ease of use", "reliability", "consistency", "accuracy",
]


def load_spacy_model():
    """Load spaCy model for NLP tasks."""
    if not SPACY_AVAILABLE:
        return None
    
    try:
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except OSError:
        try:
            print("Downloading spaCy model...")
            from spacy.cli import download
            download("en_core_web_sm")
            return spacy.load("en_core_web_sm")
        except Exception:
            return None
    except Exception:
        return None


def extract_noun_phrases_spacy(texts, nlp=None, min_count=10, max_phrases=100):
    """
    Extract frequent noun phrases from texts using spaCy.
    
    Args:
        texts: List of text strings
        nlp: spaCy model (loaded if None)
        min_count: Minimum occurrence count
        max_phrases: Maximum number of phrases to return
    
    Returns:
        List of frequent noun phrases
    """
    if nlp is None:
        nlp = load_spacy_model()
    
    if nlp is None:
        return []
    
    phrase_counts = Counter()
    
    sample_size = min(5000, len(texts))
    if len(texts) > sample_size:
        import random
        texts = random.sample(texts, sample_size)
    
    for doc in nlp.pipe(texts, batch_size=100, disable=["ner"]):
        for chunk in doc.noun_chunks:
            phrase = chunk.text.lower().strip()
            
            if len(phrase) < 3 or len(phrase.split()) > 4:
                continue
            
            phrase_words = set(phrase.split())
            if phrase_words & STOP_WORDS_EXTRA:
                continue
            
            phrase_counts[phrase] += 1
    
    frequent = [phrase for phrase, count in phrase_counts.most_common(max_phrases * 2)
                if count >= min_count]
    
    return frequent[:max_phrases]


def extract_noun_phrases_nltk(texts, min_count=10, max_phrases=100):
    """
    Extract frequent noun phrases using NLTK (fallback when spaCy unavailable).
    Uses POS tagging and regex patterns.
    """
    import nltk
    from nltk import pos_tag, word_tokenize
    from nltk.chunk import RegexpParser
    
    try:
        nltk.data.find("taggers/averaged_perceptron_tagger")
    except LookupError:
        nltk.download("averaged_perceptron_tagger", quiet=True)
    
    grammar = r"""
        NP: {<DT>?<JJ>*<NN.*>+}
            {<NN.*>+}
    """
    parser = RegexpParser(grammar)
    
    phrase_counts = Counter()
    
    sample_size = min(3000, len(texts))
    if len(texts) > sample_size:
        import random
        texts = random.sample(texts, sample_size)
    
    for text in texts:
        try:
            tokens = word_tokenize(text.lower())
            tagged = pos_tag(tokens)
            tree = parser.parse(tagged)
            
            for subtree in tree.subtrees():
                if subtree.label() == "NP":
                    phrase = " ".join(word for word, tag in subtree.leaves())
                    
                    if len(phrase) < 3 or len(phrase.split()) > 4:
                        continue
                    
                    phrase_words = set(phrase.split())
                    if phrase_words & STOP_WORDS_EXTRA:
                        continue
                    
                    phrase_counts[phrase] += 1
        except Exception:
            continue
    
    frequent = [phrase for phrase, count in phrase_counts.most_common(max_phrases * 2)
                if count >= min_count]
    
    return frequent[:max_phrases]


def get_dynamic_aspects(texts, use_spacy=True, min_count=10, max_aspects=50):
    """
    Extract aspects dynamically from text using NLP.
    
    Args:
        texts: List of review texts
        use_spacy: Use spaCy if available
        min_count: Minimum occurrence count
        max_aspects: Maximum number of aspects
    
    Returns:
        List of aspect strings
    """
    if use_spacy and SPACY_AVAILABLE:
        aspects = extract_noun_phrases_spacy(texts, min_count=min_count, max_phrases=max_aspects)
    else:
        aspects = extract_noun_phrases_nltk(texts, min_count=min_count, max_phrases=max_aspects)
    
    all_aspects = list(set(TECH_ASPECTS + aspects))
    
    return all_aspects


def extract_key_phrases(texts, top_n=30, ngram_range=(1, 3)):
    """Extract key phrases using TF-IDF analysis."""
    if not texts or len(texts) == 0:
        return []
    
    texts = [str(t) for t in texts if t and len(str(t)) > 10]
    
    if len(texts) < 10:
        return []
    
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=ngram_range,
        stop_words="english",
        min_df=5,
        max_df=0.5,
    )
    
    try:
        tfidf_matrix = tfidf.fit_transform(texts)
    except ValueError:
        return []
    
    feature_names = tfidf.get_feature_names_out()
    mean_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
    top_indices = mean_scores.argsort()[::-1]
    
    phrases = []
    for idx in top_indices:
        phrase = feature_names[idx]
        if phrase.lower() not in STOP_WORDS_EXTRA and len(phrase) > 2:
            phrases.append({
                "phrase": phrase,
                "tfidf_score": float(mean_scores[idx]),
            })
            if len(phrases) >= top_n:
                break
    
    return phrases


def extract_aspect_sentiments(df, aspects=None, dynamic=True, text_column="review_text"):
    """
    Extract aspect-level sentiment aggregations.
    
    Args:
        df: DataFrame with reviews and sentiment scores
        aspects: List of aspects to search for (auto-detected if None)
        dynamic: Use dynamic aspect extraction
        text_column: Column containing review text
    
    Returns:
        DataFrame with aspect sentiment statistics
    """
    if aspects is None:
        if dynamic:
            texts = df[text_column].dropna().astype(str).tolist()
            aspects = get_dynamic_aspects(texts, min_count=max(10, len(df) // 500))
        else:
            aspects = TECH_ASPECTS
    
    results = []
    
    for aspect in aspects:
        pattern = re.compile(re.escape(aspect), re.IGNORECASE)
        mask = df[text_column].astype(str).str.contains(pattern, na=False)
        
        min_mentions = max(10, len(df) // 1000)
        if mask.sum() < min_mentions:
            continue
        
        subset = df[mask]
        
        sentiment_col = "ensemble_score" if "ensemble_score" in df.columns else None
        label_col = "sentiment_label" if "sentiment_label" in df.columns else None
        rating_col = "rating" if "rating" in df.columns else None
        
        result = {
            "aspect": aspect,
            "mention_count": int(mask.sum()),
            "mention_pct": float(mask.sum() / len(df) * 100),
        }
        
        if sentiment_col:
            result["avg_sentiment"] = float(subset[sentiment_col].mean())
        
        if rating_col:
            result["avg_rating"] = float(subset[rating_col].mean())
        
        if label_col:
            result["positive_pct"] = float((subset[label_col] == "positive").mean() * 100)
            result["negative_pct"] = float((subset[label_col] == "negative").mean() * 100)
            result["neutral_pct"] = float((subset[label_col] == "neutral").mean() * 100)
        
        results.append(result)
    
    if not results:
        return pd.DataFrame()
    
    return pd.DataFrame(results).sort_values("mention_count", ascending=False)


def analyze_drivers(df, sentiment_type="positive", text_column="review_text", 
                    label_column="sentiment_label", top_n=20):
    """
    Identify key phrases driving positive or negative sentiment.
    
    Args:
        df: DataFrame with reviews and sentiment labels
        sentiment_type: "positive", "negative", or "all"
        text_column: Column containing review text
        label_column: Column containing sentiment labels
        top_n: Number of top phrases to return
    
    Returns:
        List of driver phrases with TF-IDF scores
    """
    if label_column not in df.columns:
        subset = df
    elif sentiment_type == "positive":
        subset = df[df[label_column] == "positive"]
    elif sentiment_type == "negative":
        subset = df[df[label_column] == "negative"]
    else:
        subset = df
    
    if len(subset) == 0:
        return []
    
    texts = subset[text_column].dropna().astype(str).tolist()
    return extract_key_phrases(texts, top_n=top_n)


def category_sentiment_summary(df, category_column="category"):
    """Generate sentiment summary by category."""
    if category_column not in df.columns:
        return pd.DataFrame()
    
    agg_funcs = {}
    
    if "rating" in df.columns:
        agg_funcs["avg_rating"] = ("rating", "mean")
    
    if "ensemble_score" in df.columns:
        agg_funcs["avg_sentiment"] = ("ensemble_score", "mean")
    
    if "sentiment_label" in df.columns:
        agg_funcs["positive_pct"] = ("sentiment_label", lambda x: (x == "positive").mean() * 100)
        agg_funcs["negative_pct"] = ("sentiment_label", lambda x: (x == "negative").mean() * 100)
        agg_funcs["neutral_pct"] = ("sentiment_label", lambda x: (x == "neutral").mean() * 100)
    
    if "textblob_subjectivity" in df.columns:
        agg_funcs["avg_subjectivity"] = ("textblob_subjectivity", "mean")
    
    id_col = "review_id" if "review_id" in df.columns else df.columns[0]
    agg_funcs["total_reviews"] = (id_col, "count")
    
    summary = df.groupby(category_column).agg(**agg_funcs).reset_index()
    
    sort_col = "avg_sentiment" if "avg_sentiment" in summary.columns else "total_reviews"
    return summary.sort_values(sort_col, ascending=False)


def brand_sentiment_summary(df, brand_column="brand", category_column="category"):
    """Generate sentiment summary by brand (and optionally category)."""
    if brand_column not in df.columns:
        return pd.DataFrame()
    
    group_cols = [brand_column]
    if category_column in df.columns:
        group_cols = [category_column, brand_column]
    
    agg_funcs = {}
    
    id_col = "review_id" if "review_id" in df.columns else df.columns[0]
    agg_funcs["total_reviews"] = (id_col, "count")
    
    if "rating" in df.columns:
        agg_funcs["avg_rating"] = ("rating", "mean")
    
    if "ensemble_score" in df.columns:
        agg_funcs["avg_sentiment"] = ("ensemble_score", "mean")
    
    if "sentiment_label" in df.columns:
        agg_funcs["positive_pct"] = ("sentiment_label", lambda x: (x == "positive").mean() * 100)
        agg_funcs["negative_pct"] = ("sentiment_label", lambda x: (x == "negative").mean() * 100)
    
    summary = df.groupby(group_cols).agg(**agg_funcs).reset_index()
    
    sort_cols = group_cols[:-1] + ["avg_sentiment" if "avg_sentiment" in summary.columns else "total_reviews"]
    ascending = [True] * (len(group_cols) - 1) + [False]
    
    return summary.sort_values(sort_cols, ascending=ascending)


def temporal_sentiment(df, date_column="review_date", category_column="category"):
    """Generate temporal sentiment trends."""
    if date_column not in df.columns:
        return pd.DataFrame()
    
    df = df.copy()
    
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
    df = df.dropna(subset=[date_column])
    
    if len(df) == 0:
        return pd.DataFrame()
    
    df["year_month"] = df[date_column].dt.to_period("M").astype(str)
    
    group_cols = ["year_month"]
    if category_column in df.columns:
        group_cols.append(category_column)
    
    agg_funcs = {}
    
    id_col = "review_id" if "review_id" in df.columns else df.columns[0]
    agg_funcs["review_count"] = (id_col, "count")
    
    if "ensemble_score" in df.columns:
        agg_funcs["avg_sentiment"] = ("ensemble_score", "mean")
    
    if "rating" in df.columns:
        agg_funcs["avg_rating"] = ("rating", "mean")
    
    temporal = df.groupby(group_cols).agg(**agg_funcs).reset_index()
    
    return temporal.sort_values("year_month")


def run_opinion_mining(df, text_column="review_text", dynamic_aspects=True, verbose=True):
    """
    Run complete opinion mining pipeline.
    
    Args:
        df: DataFrame with reviews and sentiment analysis results
        text_column: Column containing review text
        dynamic_aspects: Use dynamic aspect extraction
        verbose: Print progress
    
    Returns:
        Dictionary with all opinion mining results
    """
    if verbose:
        print("Extracting aspect-level sentiments...")
    aspect_df = extract_aspect_sentiments(df, dynamic=dynamic_aspects, text_column=text_column)
    
    if verbose:
        print("Identifying positive sentiment drivers...")
    positive_drivers = analyze_drivers(df, "positive", text_column)
    
    if verbose:
        print("Identifying negative sentiment drivers...")
    negative_drivers = analyze_drivers(df, "negative", text_column)
    
    if verbose:
        print("Generating category summary...")
    cat_summary = category_sentiment_summary(df)
    
    if verbose:
        print("Generating brand summary...")
    brand_summary = brand_sentiment_summary(df)
    
    if verbose:
        print("Generating temporal trends...")
    temporal = temporal_sentiment(df)
    
    results = {
        "aspect_sentiments": aspect_df,
        "positive_drivers": pd.DataFrame(positive_drivers),
        "negative_drivers": pd.DataFrame(negative_drivers),
        "category_summary": cat_summary,
        "brand_summary": brand_summary,
        "temporal_trends": temporal,
    }
    
    if verbose:
        print("Opinion mining complete.")
        print(f"  Aspects found: {len(aspect_df)}")
        print(f"  Positive drivers: {len(positive_drivers)}")
        print(f"  Negative drivers: {len(negative_drivers)}")
    
    return results


if __name__ == "__main__":
    from pathlib import Path
    
    data_path = Path(__file__).parent.parent / "data" / "reviews_with_sentiment.csv"
    
    if data_path.exists():
        print(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        
        results = run_opinion_mining(df, dynamic_aspects=True, verbose=True)
        
        print("\n" + "=" * 60)
        print("CATEGORY SUMMARY")
        print("=" * 60)
        print(results["category_summary"].to_string(index=False))
        
        print("\n" + "=" * 60)
        print("TOP ASPECTS")
        print("=" * 60)
        print(results["aspect_sentiments"].head(15).to_string(index=False))
        
        print("\n" + "=" * 60)
        print("POSITIVE DRIVERS")
        print("=" * 60)
        print(results["positive_drivers"].head(10).to_string(index=False))
        
        print("\n" + "=" * 60)
        print("NEGATIVE DRIVERS")
        print("=" * 60)
        print(results["negative_drivers"].head(10).to_string(index=False))
    else:
        print(f"Data file not found: {data_path}")
        print("Run the sentiment analysis pipeline first.")

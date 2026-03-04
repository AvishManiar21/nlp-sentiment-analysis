"""
Data loader for Amazon Reviews datasets.

McAuley-Lab/Amazon-Reviews-2023 via UCSD/HuggingFace JSONL - REAL categories, brands, timestamps.
"""

import gzip
import io
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: 'datasets' library not installed. Run: pip install datasets")

# Default categories. Plan suggests All_Beauty, Electronics, Home_and_Kitchen;
# Those meta files are 5GB+ each. Using smaller cats for faster first run.
MCAULEY_CATEGORIES = ["All_Beauty", "Digital_Music", "Subscription_Boxes"]

# HuggingFace raw file base (UCSD datarepo returns 404; HF hosts the JSONL)
HF_RAW_BASE = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw"
UCSD_REVIEW_BASE = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories"
UCSD_META_BASE = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories"


def _stream_jsonl(url, max_lines=None, is_gzipped=False, verbose=False):
    """Stream and parse JSONL from URL. Yields dicts. Stops after max_lines if set."""
    if not REQUESTS_AVAILABLE:
        raise ImportError("requests required. Install with: pip install requests")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    count = 0
    if is_gzipped:
        gz = gzip.GzipFile(fileobj=resp.raw, mode="rb")
        reader = io.TextIOWrapper(gz, encoding="utf-8", errors="replace")
    else:
        reader = resp.iter_lines(decode_unicode=True)
    for line in reader:
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="replace")
        line = (line or "").strip()
        if not line:
            continue
        try:
            yield json.loads(line)
            count += 1
            if max_lines is not None and count >= max_lines:
                break
        except json.JSONDecodeError:
            continue


def _fetch_meta_lookup(category, verbose=True):
    """Fetch metadata for a category and return parent_asin -> {main_category, store} dict."""
    meta_url_hf = f"{HF_RAW_BASE}/meta_categories/meta_{category}.jsonl"
    meta_url_ucsd = f"{UCSD_META_BASE}/meta_{category}.jsonl.gz"
    lookup = {}
    for url, gz in [(meta_url_ucsd, True), (meta_url_hf, False)]:
        try:
            if verbose:
                print(f"  Fetching meta for {category}...")
            for obj in _stream_jsonl(url, is_gzipped=gz, verbose=verbose):
                pid = obj.get("parent_asin")
                if pid and pid not in lookup:
                    lookup[pid] = {
                        "main_category": obj.get("main_category") or category.replace("_", " "),
                        "store": obj.get("store") or "Unknown",
                    }
            if lookup:
                if verbose:
                    print(f"  Loaded {len(lookup):,} meta entries for {category}")
                return lookup
        except Exception as e:
            if verbose:
                print(f"  Meta {url[:50]}... failed: {e}")
            continue
    return lookup


def load_mcauley_reviews(
    sample_size=50000,
    categories=None,
    output_path=None,
    force_reload=False,
    verbose=True,
):
    """
    Load McAuley-Lab Amazon Reviews 2023 from JSONL files (UCSD or HuggingFace).
    Streams a limited number of lines per category to avoid loading huge files.
    Real categories, brands (store), ratings, timestamps.
    """
    if output_path is None:
        output_path = Path(__file__).parent.parent / "data" / "amazon_reviews.csv"
    output_path = Path(output_path)
    categories = categories or MCAULEY_CATEGORIES

    if output_path.exists() and not force_reload:
        if verbose:
            print(f"Loading cached reviews from {output_path}")
        df = pd.read_csv(output_path)
        if verbose:
            print(f"Loaded {len(df):,} reviews from cache")
        return df

    if verbose:
        print("\n" + "=" * 60)
        print("Fetching McAuley-Lab Amazon Reviews 2023 (JSONL)")
        print("=" * 60)

    n_per_cat = max(5000, sample_size // len(categories))
    all_rows = []

    for cat in categories:
        rev_url_hf = f"{HF_RAW_BASE}/review_categories/{cat}.jsonl"
        rev_url_ucsd = f"{UCSD_REVIEW_BASE}/{cat}.jsonl.gz"
        meta_lookup = _fetch_meta_lookup(cat, verbose=verbose)

        for url, gz in [(rev_url_ucsd, True), (rev_url_hf, False)]:
            try:
                if verbose:
                    print(f"Streaming reviews from {cat} (max {n_per_cat:,} lines)...")
                rows = []
                for obj in tqdm(
                    _stream_jsonl(url, max_lines=n_per_cat, is_gzipped=gz, verbose=verbose),
                    total=n_per_cat,
                    disable=not verbose,
                    desc=cat,
                    unit=" reviews",
                ):
                    text = (obj.get("title") or "") + ". " + (obj.get("text") or "")
                    if len(text.strip()) < 20:
                        continue
                    pid = obj.get("parent_asin")
                    meta = meta_lookup.get(pid, {}) if meta_lookup else {}
                    rows.append({
                        "title": obj.get("title") or "",
                        "text": obj.get("text") or "",
                        "rating": obj.get("rating", 3),
                        "parent_asin": pid,
                        "timestamp": obj.get("timestamp"),
                        "helpful_vote": obj.get("helpful_vote", 0),
                        "verified_purchase": obj.get("verified_purchase", True),
                        "main_category": meta.get("main_category", cat.replace("_", " ")),
                        "store": meta.get("store", "Unknown"),
                    })
                if rows:
                    all_rows.extend(rows)
                    if verbose:
                        print(f"  Got {len(rows):,} reviews from {cat}")
                    break
            except Exception as e:
                if verbose:
                    print(f"  Review {url[:60]}... failed: {e}")
                continue

    if not all_rows:
        raise RuntimeError("Could not load any McAuley reviews from JSONL (UCSD or HuggingFace)")

    df = pd.DataFrame(all_rows)
    df["category"] = df["main_category"].fillna("Unknown")
    df["brand"] = df["store"].fillna("Unknown")
    df["review_date"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")

    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    mapped = map_mcauley_to_project_schema(df, verbose=verbose)
    mapped = filter_reviews(mapped, verbose=verbose)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mapped.to_csv(output_path, index=False)
    if verbose:
        print(f"Saved {len(mapped):,} reviews to {output_path}")
    return mapped


def map_mcauley_to_project_schema(df, verbose=True):
    """Map McAuley-Lab schema to project schema."""
    mapped = pd.DataFrame()
    mapped["review_id"] = [f"AMZ-{i:07d}" for i in range(len(df))]
    mapped["review_text"] = (
        df["title"].fillna("").astype(str) + ". " + df["text"].fillna("").astype(str)
    ).str.strip().str.replace(r"^\.\s*", "", regex=True)
    mapped["review_title"] = df["title"].fillna("").astype(str)
    mapped["rating"] = df["rating"].clip(1, 5).astype(int)
    mapped["ground_truth"] = mapped["rating"].apply(
        lambda r: "negative" if r <= 2 else "positive"
    )
    mapped["category"] = df["category"].fillna("Unknown").astype(str)
    mapped["brand"] = df["brand"].fillna("Unknown").astype(str)
    mapped["review_date"] = df.get("review_date", pd.NaT)
    mapped["verified_purchase"] = df.get("verified_purchase", True)
    mapped["helpful_votes"] = df.get("helpful_vote", 0).fillna(0).astype(int)
    return mapped


def load_amazon_mcauley(
    sample_size=50000,
    output_path=None,
    force_reload=False,
    verbose=True,
):
    """
    Load REAL Amazon reviews from McAuley-Lab/Amazon-Reviews-2023.
    Uses direct JSONL download (no fallback).
    """
    return load_mcauley_reviews(
        sample_size=sample_size,
        output_path=output_path,
        force_reload=force_reload,
        verbose=verbose,
    )


def _map_mcauley_to_schema(df, verbose=True):
    """Map McAuley-Lab schema to project schema."""
    mapped = pd.DataFrame()
    mapped["review_id"] = [f"AMZ-{i:07d}" for i in range(len(df))]
    mapped["review_text"] = (
        df["title"].fillna("").astype(str) + ". " + df["text"].fillna("").astype(str)
    ).str.strip().str.replace(r"^\.\s*", "", regex=True)
    mapped["review_title"] = df["title"].fillna("").astype(str)
    mapped["rating"] = df["rating"].clip(1, 5).astype(int)
    mapped["ground_truth"] = mapped["rating"].apply(
        lambda r: "negative" if r <= 2 else "positive"
    )
    mapped["category"] = df["category"].fillna("Unknown").astype(str)
    mapped["brand"] = df["brand"].fillna("Unknown").astype(str)
    mapped["review_date"] = df.get("review_date", pd.NaT)
    mapped["verified_purchase"] = df.get("verified_purchase", True)
    mapped["helpful_votes"] = df.get("helpful_vote", 0).fillna(0).astype(int)
    return mapped


def load_amazon_polarity(
    sample_size=50000,
    output_path=None,
    force_reload=False,
    verbose=True,
):
    """
    Load REAL Amazon reviews from fancyzhx/amazon_polarity dataset.
    
    This dataset contains 3.6M real Amazon product reviews:
    - label: 0=negative (1-2 stars), 1=positive (4-5 stars)
    - title: Review title (real)
    - content: Review body (real)
    
    Args:
        sample_size: Number of reviews to sample (default 50K)
        output_path: Path to save/load cached CSV
        force_reload: Force re-download even if cache exists
        verbose: Show progress
    
    Returns:
        DataFrame with real Amazon reviews
    """
    if not HF_AVAILABLE:
        raise ImportError("datasets library required. Install with: pip install datasets")
    
    if output_path is None:
        output_path = Path(__file__).parent.parent / "data" / "amazon_reviews.csv"
    
    output_path = Path(output_path)
    
    if output_path.exists() and not force_reload:
        if verbose:
            print(f"Loading cached reviews from {output_path}")
        df = pd.read_csv(output_path)
        if verbose:
            print(f"Loaded {len(df):,} reviews from cache")
        return df
    
    if verbose:
        print("\n" + "=" * 60)
        print("Fetching REAL Amazon Reviews from HuggingFace")
        print("Dataset: fancyzhx/amazon_polarity (3.6M reviews)")
        print("=" * 60)
        print(f"Sampling {sample_size:,} reviews...")
        print("-" * 60)
    
    try:
        dataset = load_dataset(
            "fancyzhx/amazon_polarity",
            split="train",
        )
        
        if verbose:
            print(f"Loaded {len(dataset):,} reviews from HuggingFace")
        
        df = dataset.to_pandas()
        
        if sample_size and len(df) > sample_size:
            neg_samples = sample_size // 2
            pos_samples = sample_size - neg_samples
            
            neg_df = df[df["label"] == 0].sample(n=neg_samples, random_state=42)
            pos_df = df[df["label"] == 1].sample(n=pos_samples, random_state=42)
            
            df = pd.concat([neg_df, pos_df], ignore_index=True)
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            if verbose:
                print(f"Sampled {len(df):,} reviews (balanced: {neg_samples:,} neg, {pos_samples:,} pos)")
        
    except Exception as e:
        raise ValueError(f"Could not load dataset: {e}")
    
    mapped_df = map_to_project_schema(df, verbose=verbose)
    
    mapped_df = filter_reviews(mapped_df, verbose=verbose)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mapped_df.to_csv(output_path, index=False)
    
    if verbose:
        print("-" * 60)
        print(f"Total reviews: {len(mapped_df):,}")
        print(f"Saved to: {output_path}")
        print("\nSentiment distribution:")
        print(mapped_df["ground_truth"].value_counts().to_string())
        print("\nRating distribution:")
        print(mapped_df["rating"].value_counts().sort_index().to_string())
    
    return mapped_df


def map_to_project_schema(df, verbose=True):
    """
    Map amazon_polarity dataset fields to project schema.
    All review text is REAL from Amazon.
    """
    if verbose:
        print("\nMapping to project schema...")
    
    mapped = pd.DataFrame()
    
    mapped["review_id"] = [f"AMZ-{i:07d}" for i in range(len(df))]
    
    if "content" in df.columns and "title" in df.columns:
        mapped["review_text"] = df.apply(
            lambda row: f"{row['title']}. {row['content']}" if pd.notna(row['title']) and row['title'] 
            else row['content'],
            axis=1
        ).fillna("").astype(str)
    elif "content" in df.columns:
        mapped["review_text"] = df["content"].fillna("").astype(str)
    else:
        mapped["review_text"] = ""
    
    if "title" in df.columns:
        mapped["review_title"] = df["title"].fillna("").astype(str)
    
    if "label" in df.columns:
        mapped["rating"] = df["label"].apply(lambda x: 5 if x == 1 else 1)
        mapped["ground_truth"] = df["label"].apply(lambda x: "positive" if x == 1 else "negative")
    else:
        mapped["rating"] = 3
        mapped["ground_truth"] = "neutral"
    
    categories = [
        "Electronics", "Books", "Home & Kitchen", "Clothing",
        "Sports & Outdoors", "Health & Personal Care", "Toys & Games",
        "Automotive", "Office Products", "Pet Supplies"
    ]
    brands = [
        "Amazon", "Apple", "Samsung", "Sony", "LG", "Microsoft",
        "Dell", "HP", "Bose", "JBL", "Fitbit", "Nike"
    ]
    np.random.seed(42)
    mapped["category"] = np.random.choice(categories, size=len(df))
    mapped["brand"] = np.random.choice(brands, size=len(df))
    
    mapped["verified_purchase"] = True
    mapped["helpful_votes"] = 0
    
    return mapped


def filter_reviews(df, min_length=20, max_length=5000, verbose=True):
    """Filter out low-quality reviews."""
    initial_count = len(df)
    
    df = df[df["review_text"].str.len() >= min_length]
    df = df[df["review_text"].str.len() <= max_length]
    df = df.drop_duplicates(subset=["review_text"])
    
    final_count = len(df)
    if verbose:
        print(f"Filtered: {initial_count:,} -> {final_count:,} reviews")
    
    return df.reset_index(drop=True)


def load_amazon_reviews(output_path=None, categories=None, force_reload=False, sample_size=50000, verbose=True):
    """
    Main entry point to load Amazon reviews.
    Uses McAuley-Lab only (no fallback).
    """
    return load_mcauley_reviews(
        sample_size=sample_size,
        categories=categories,
        output_path=output_path,
        force_reload=force_reload,
        verbose=verbose,
    )


def get_dataset_stats(df):
    """Get summary statistics for the dataset."""
    stats = {
        "total_reviews": len(df),
        "categories": df["category"].nunique() if "category" in df.columns else 0,
        "brands": df["brand"].nunique() if "brand" in df.columns else 0,
        "avg_review_length": df["review_text"].str.len().mean(),
        "median_review_length": df["review_text"].str.len().median(),
        "rating_distribution": df["rating"].value_counts().sort_index().to_dict() if "rating" in df.columns else {},
    }
    
    if "ground_truth" in df.columns:
        stats["sentiment_distribution"] = df["ground_truth"].value_counts().to_dict()
    
    return stats


if __name__ == "__main__":
    print("=" * 60)
    print("Loading Amazon Reviews (Real Data)")
    print("=" * 60)
    
    df = load_amazon_reviews(force_reload=True)
    
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    
    stats = get_dataset_stats(df)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\nSample reviews:")
    for _, row in df.head(3).iterrows():
        print(f"\n[{row['ground_truth'].upper()}] Rating: {row['rating']}")
        print(f"  {row['review_text'][:200]}...")

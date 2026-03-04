"""
Data loader for Amazon Reviews datasets from HuggingFace.

Primary: McAuley-Lab/Amazon-Reviews-2023 - REAL categories, brands, ratings, timestamps.
Fallback: fancyzhx/amazon_polarity - real text and labels, synthetic categories/brands.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: 'datasets' library not installed. Run: pip install datasets")

MCAULEY_CATEGORIES = [
    "All_Beauty",  # 701K reviews, 112K meta - manageable
    "Digital_Music",  # 130K reviews, 70K meta - small
]


def load_amazon_mcauley(
    sample_size=50000,
    output_path=None,
    force_reload=False,
    verbose=True,
):
    """
    Load REAL Amazon reviews from McAuley-Lab/Amazon-Reviews-2023.
    Real categories, real brands (store), real ratings, real timestamps.
    Falls back to amazon_polarity if loading fails.
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
        print("Fetching McAuley-Lab Amazon Reviews 2023 (REAL metadata)")
        print("=" * 60)

    all_reviews = []
    all_meta = []

    for cat in MCAULEY_CATEGORIES:
        try:
            n_per_cat = max(5000, sample_size // len(MCAULEY_CATEGORIES))
            rev_config = f"raw_review_{cat}"
            meta_config = f"raw_meta_{cat}"
            if verbose:
                print(f"Loading {cat}...")
            rev_ds = load_dataset(
                "McAuley-Lab/Amazon-Reviews-2023",
                rev_config,
                split=f"full[:{n_per_cat}]",
                trust_remote_code=True,
            )
            meta_ds = load_dataset(
                "McAuley-Lab/Amazon-Reviews-2023",
                meta_config,
                split="full",
                trust_remote_code=True,
            )
            rev_df = rev_ds.to_pandas()
            meta_df = meta_ds.to_pandas()
            meta_sub = meta_df[["parent_asin", "main_category", "store"]].drop_duplicates("parent_asin")
            rev_df = rev_df.merge(
                meta_sub, on="parent_asin", how="left", suffixes=("", "_meta")
            )
            rev_df["category"] = rev_df["main_category"].fillna(cat.replace("_", " "))
            rev_df["brand"] = rev_df["store"].fillna("Unknown")
            rev_df["review_date"] = pd.to_datetime(rev_df["timestamp"], unit="ms", errors="coerce")
            all_reviews.append(rev_df)
        except Exception as e:
            if verbose:
                print(f"  Skipping {cat}: {e}")
            continue

    if not all_reviews:
        if verbose:
            print("McAuley-Lab load failed, falling back to amazon_polarity...")
        return load_amazon_polarity(sample_size, output_path, force_reload, verbose)

    df = pd.concat(all_reviews, ignore_index=True)
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    mapped = _map_mcauley_to_schema(df, verbose)
    mapped = filter_reviews(mapped, verbose=verbose)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mapped.to_csv(output_path, index=False)
    if verbose:
        print(f"Saved {len(mapped):,} reviews to {output_path}")
    return mapped


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
    Tries McAuley-Lab (real categories, brands) first, falls back to amazon_polarity.
    """
    try:
        return load_amazon_mcauley(
            sample_size=sample_size,
            output_path=output_path,
            force_reload=force_reload,
            verbose=verbose,
        )
    except Exception as e:
        if verbose:
            print(f"McAuley-Lab unavailable ({e}), using amazon_polarity...")
        return load_amazon_polarity(
            sample_size=sample_size,
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

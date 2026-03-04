"""
Main pipeline for NLP Sentiment Analysis & Opinion Mining.
Loads real Amazon reviews, preprocesses text, trains ML models,
evaluates performance, and generates visualizations.
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from src.data_loader import load_amazon_reviews, get_dataset_stats
from src.preprocessor import preprocess_dataframe, save_preprocessed_data
from src.sentiment_analyzer import run_sentiment_analysis, get_sentiment_summary
from src.ml_models import run_ml_pipeline, save_model
from src.opinion_miner import run_opinion_mining
from src.visualizer import generate_all_visualizations
from src.model_evaluator import compare_all_models, save_evaluation_results, print_detailed_report


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="NLP Sentiment Analysis Pipeline with Real Amazon Reviews"
    )
    parser.add_argument(
        "--force-reload",
        action="store_true",
        help="Force reload data from HuggingFace (ignore cache)"
    )
    parser.add_argument(
        "--skip-ml",
        action="store_true",
        help="Skip ML model training"
    )
    parser.add_argument(
        "--train-transformer",
        action="store_true",
        help="Train DistilBERT model (requires GPU for reasonable speed)"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Limit dataset to N samples (for quick testing)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for visualizations"
    )
    return parser.parse_args()


def print_header():
    """Print pipeline header."""
    print("=" * 70)
    print("  NLP Sentiment Analysis & Opinion Mining Pipeline")
    print("  Dataset: Amazon Reviews 2023 (Real Product Reviews)")
    print("=" * 70)


def step_load_data(args, data_dir):
    """Step 1: Load Amazon reviews from HuggingFace or cache."""
    print("\n[1/7] Loading Amazon Reviews dataset...")
    
    reviews_path = data_dir / "amazon_reviews.csv"
    
    df = load_amazon_reviews(
        output_path=reviews_path,
        force_reload=args.force_reload,
    )
    
    if args.sample_size and len(df) > args.sample_size:
        print(f"  Sampling {args.sample_size:,} reviews for faster processing...")
        df = df.sample(n=args.sample_size, random_state=42).reset_index(drop=True)
    
    stats = get_dataset_stats(df)
    print(f"  Loaded {stats['total_reviews']:,} reviews")
    print(f"  Categories: {stats['categories']}")
    if "brands" in stats:
        print(f"  Brands: {stats['brands']}")
    print(f"  Avg review length: {stats['avg_review_length']:.0f} chars")
    
    return df


def step_preprocess(df, data_dir):
    """Step 2: Preprocess text data."""
    print("\n[2/7] Preprocessing text data...")
    
    preprocessed_path = data_dir / "preprocessed_reviews.csv"
    
    if preprocessed_path.exists():
        print(f"  Loading cached preprocessed data from {preprocessed_path}")
        df = pd.read_csv(preprocessed_path)
    else:
        df = preprocess_dataframe(df, verbose=True)
        save_preprocessed_data(df, preprocessed_path)
    
    print(f"  Ground truth distribution:")
    print(f"    {df['ground_truth'].value_counts().to_dict()}")
    
    return df


def step_sentiment_analysis(df, data_dir):
    """Step 3: Run VADER and TextBlob sentiment analysis."""
    print("\n[3/7] Running sentiment analysis (VADER + TextBlob)...")
    
    analyzed_path = data_dir / "reviews_with_sentiment.csv"
    
    df = run_sentiment_analysis(df, text_column="review_text", verbose=True)
    
    df.to_csv(analyzed_path, index=False)
    print(f"  Saved analyzed data -> {analyzed_path}")
    
    summary = get_sentiment_summary(df)
    print(f"\n  Sentiment summary:")
    print(f"    Avg VADER compound: {summary['avg_vader_compound']:.4f}")
    print(f"    Avg TextBlob polarity: {summary['avg_textblob_polarity']:.4f}")
    print(f"    Avg ensemble score: {summary['avg_ensemble_score']:.4f}")
    
    return df


def step_train_ml_models(df, skip_ml=False):
    """Step 4: Train classical ML models."""
    if skip_ml:
        print("\n[4/7] Skipping ML model training (--skip-ml flag)")
        return None
    
    print("\n[4/7] Training ML models (Logistic Regression, Naive Bayes)...")
    
    ml_results = run_ml_pipeline(
        df,
        text_column="processed_text",
        label_column="ground_truth",
        model_types=["logistic_regression", "naive_bayes"],
        save_models=True,
        verbose=True,
    )
    
    return ml_results


def step_train_transformer(df, train_transformer=False):
    """Step 5: Train DistilBERT model (optional)."""
    if not train_transformer:
        print("\n[5/7] Skipping transformer training (use --train-transformer to enable)")
        return None
    
    print("\n[5/7] Training DistilBERT model...")
    
    try:
        from src.transformer_model import (
            train_distilbert, 
            is_transformer_available,
            get_gpu_info,
        )
        
        if not is_transformer_available():
            print("  Transformer dependencies not available. Skipping.")
            print("  Install with: pip install torch transformers")
            return None
        
        gpu_info = get_gpu_info()
        if not gpu_info.get("available"):
            print(f"  Warning: No GPU detected. Training on CPU will be slow.")
            print(f"  GPU info: {gpu_info}")
        
        sample_size = min(20000, len(df))
        if len(df) > sample_size:
            print(f"  Using {sample_size:,} samples for transformer training...")
            train_df = df.sample(n=sample_size, random_state=42)
        else:
            train_df = df
        
        results = train_distilbert(
            train_df,
            text_column="cleaned_text",
            label_column="ground_truth",
            epochs=3,
            batch_size=16,
            verbose=True,
        )
        
        return results
    
    except Exception as e:
        print(f"  Transformer training failed: {e}")
        return None


def step_evaluate_models(df, ml_results, transformer_results):
    """Step 6: Evaluate and compare all models."""
    print("\n[6/7] Evaluating and comparing models...")
    
    ml_model_results = ml_results.get("models", {}) if ml_results else {}
    
    evaluation = compare_all_models(
        df,
        ml_results=ml_model_results,
        transformer_results=transformer_results,
        text_column="review_text",
        processed_column="processed_text",
        cleaned_column="cleaned_text" if "cleaned_text" in df.columns else "review_text",
        label_column="ground_truth",
        include_transformer=transformer_results is not None,
        verbose=True,
    )
    
    return evaluation


def step_opinion_mining(df):
    """Step 6.5: Run opinion mining."""
    print("\n[6.5/7] Running opinion mining...")
    
    mining_results = run_opinion_mining(df, dynamic_aspects=True, verbose=True)
    
    print("\n  Category Summary:")
    if not mining_results["category_summary"].empty:
        print(mining_results["category_summary"].to_string(index=False))
    
    print("\n  Top Aspects:")
    if not mining_results["aspect_sentiments"].empty:
        print(mining_results["aspect_sentiments"].head(10).to_string(index=False))
    
    return mining_results


def step_generate_visualizations(df, mining_results, evaluation_results, output_dir):
    """Step 7: Generate all visualizations."""
    print("\n[7/7] Generating visualizations...")
    
    paths = generate_all_visualizations(
        df,
        mining_results,
        output_dir,
        evaluation_results=evaluation_results,
    )
    
    print("\nGenerated files:")
    for name, path in paths.items():
        if isinstance(path, dict):
            for sub_name, sub_path in path.items():
                print(f"  {name}/{sub_name}: {sub_path}")
        else:
            print(f"  {name}: {path}")
    
    return paths


def save_results(evaluation_results, results_dir):
    """Save evaluation results to files."""
    if evaluation_results:
        save_evaluation_results(evaluation_results, results_dir)
        print(f"\nEvaluation results saved to: {results_dir}")


def main():
    """Run the complete pipeline."""
    args = parse_args()
    
    data_dir = Path(__file__).parent / "data"
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent / "outputs"
    results_dir = Path(__file__).parent / "results"
    
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print_header()
    
    df = step_load_data(args, data_dir)
    
    df = step_preprocess(df, data_dir)
    
    df = step_sentiment_analysis(df, data_dir)
    
    ml_results = step_train_ml_models(df, skip_ml=args.skip_ml)
    
    transformer_results = step_train_transformer(df, train_transformer=args.train_transformer)
    
    evaluation_results = step_evaluate_models(df, ml_results, transformer_results)
    
    mining_results = step_opinion_mining(df)
    
    paths = step_generate_visualizations(df, mining_results, evaluation_results, output_dir)
    
    save_results(evaluation_results, results_dir)
    
    if evaluation_results:
        print("\n")
        print_detailed_report(evaluation_results)
    
    print("\n" + "=" * 70)
    print("  Pipeline complete!")
    print(f"  Total reviews analyzed: {len(df):,}")
    print(f"  Visualizations saved to: {output_dir}/")
    print(f"  Results saved to: {results_dir}/")
    print(f"\n  Run the dashboard: python -m streamlit run app.py")
    print("=" * 70)


if __name__ == "__main__":
    main()

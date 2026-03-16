"""Data loading and caching utilities."""

import streamlit as st
import pandas as pd
import json
from pathlib import Path
import time
import os

from utils.logger import log_data_load

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
RESULTS_DIR = Path(__file__).parent.parent / "results"
MODELS_DIR = Path(__file__).parent.parent / "models"

CLOUD_SAMPLE_SIZE = 30000

# Optional Cloud-specific sampling to keep the dashboard responsive
IS_CLOUD_MODE = os.getenv("CLOUD_MODE", "").lower() == "true"
CLOUD_DISPLAY_SAMPLE_SIZE = int(os.getenv("CLOUD_DISPLAY_SAMPLE_SIZE", "20000"))


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
    start = time.perf_counter()
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
    
    # On Streamlit Cloud, optionally downsample for faster interactions
    if IS_CLOUD_MODE and len(df) > CLOUD_DISPLAY_SAMPLE_SIZE:
        df = df.sample(CLOUD_DISPLAY_SAMPLE_SIZE, random_state=42).reset_index(drop=True)
    
    duration_ms = (time.perf_counter() - start) * 1000
    log_data_load(source=str(path), count=len(df), duration_ms=duration_ms)
    
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


@st.cache_data
def check_dl_models_available():
    """Check which deep learning models are trained and available."""
    dl_models_dir = MODELS_DIR / "dl"

    if not dl_models_dir.exists():
        return []

    available_models = []

    # Check for TensorFlow models
    tf_models = [
        ("cnn_tensorflow", "CNN (TensorFlow)", ".keras"),
        ("cnn_tensorflow_pretrained", "CNN + GloVe (TensorFlow)", ".keras"),
    ]

    for model_file, display_name, ext in tf_models:
        if (dl_models_dir / f"{model_file}{ext}").exists():
            available_models.append({
                "file": model_file,
                "name": display_name,
                "framework": "TensorFlow",
                "type": "CNN"
            })

    # Check for PyTorch models
    pt_models = [
        ("cnn_pytorch", "CNN (PyTorch)", ".pt"),
        ("cnn_pytorch_pretrained", "CNN + GloVe (PyTorch)", ".pt"),
        ("lstm_pytorch", "BiLSTM (PyTorch)", ".pt"),
        ("lstm_pytorch_pretrained", "BiLSTM + GloVe (PyTorch)", ".pt"),
    ]

    for model_file, display_name, ext in pt_models:
        if (dl_models_dir / f"{model_file}{ext}").exists():
            model_type = "BiLSTM" if "lstm" in model_file else "CNN"
            available_models.append({
                "file": model_file,
                "name": display_name,
                "framework": "PyTorch",
                "type": model_type
            })

    return available_models

#!/usr/bin/env python3
"""
Re-evaluate existing deep learning models and generate metrics JSON files.

This script loads trained models, re-runs predictions on the test set,
calculates full metrics (F1, Precision, Recall), and saves them to JSON files.

Usage:
    python scripts/regenerate_dl_metrics.py
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
from datetime import datetime
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import deep learning modules
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available")

import torch

# Local imports
from src.dl_trainer import prepare_data_for_dl
from src.embedding_manager import EmbeddingManager
from utils.cache import load_data

MODELS_DIR = Path("models/dl")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def calculate_and_save_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    framework: str,
    n_samples: int,
    label_mapping: dict = None
):
    """Calculate and save full evaluation metrics."""
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)

    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred)

    # Compile results
    results = {
        "model_name": model_name,
        "framework": framework,
        "accuracy": float(accuracy),
        "test_loss": 0.0,  # Not available without re-running model
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "n_samples": int(n_samples),
        "confusion_matrix": cm.tolist(),
        "label_mapping": label_mapping,
        "timestamp": datetime.now().isoformat(),
        "note": "Regenerated from existing trained model"
    }

    # Save to JSON
    safe_name = model_name.lower().replace(" ", "_").replace("+", "").replace("(", "").replace(")", "")
    results_file = RESULTS_DIR / f"dl_results_{safe_name}.json"

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"✓ Saved metrics for {model_name}")
    logger.info(f"  Accuracy: {accuracy:.4f}, F1 (weighted): {f1_weighted:.4f}, F1 (macro): {f1_macro:.4f}")

    return results


def evaluate_tensorflow_model(model_path, model_name, data_dict):
    """Evaluate a TensorFlow model."""
    logger.info(f"Evaluating TensorFlow model: {model_name}")

    model = keras.models.load_model(model_path)

    # Get predictions
    y_pred_probs = model.predict(data_dict['X_test'], verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = data_dict['y_test']

    # Calculate and save metrics
    calculate_and_save_metrics(
        y_true=y_true,
        y_pred=y_pred,
        model_name=model_name,
        framework='tensorflow',
        n_samples=len(y_true),
        label_mapping=data_dict.get('label_mapping')
    )


def evaluate_pytorch_model(model_path, model_name, data_dict):
    """Evaluate a PyTorch model."""
    logger.info(f"Evaluating PyTorch model: {model_name}")

    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    model = checkpoint['model']
    model.eval()

    # Prepare test data
    X_test_tensor = torch.LongTensor(data_dict['X_test'])
    y_test = data_dict['y_test']

    # Get predictions
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, y_pred = torch.max(outputs, 1)
        y_pred = y_pred.numpy()

    # Calculate and save metrics
    calculate_and_save_metrics(
        y_true=y_test,
        y_pred=y_pred,
        model_name=model_name,
        framework='pytorch',
        n_samples=len(y_test),
        label_mapping=data_dict.get('label_mapping')
    )


def main():
    """Main function."""
    logger.info("=" * 60)
    logger.info("Regenerating Deep Learning Model Metrics")
    logger.info("=" * 60)

    # Load data
    logger.info("Loading dataset...")
    df = load_data()

    if len(df) == 0:
        logger.error("No data found. Run main.py first to generate data.")
        return

    # Prepare data with GloVe embeddings for pretrained models
    logger.info("Preparing data with GloVe embeddings...")
    embedding_manager = EmbeddingManager()
    embedding_manager.load_embedding('glove-wiki-gigaword-100')
    embedding_manager.build_vocab(df['processed_text'].tolist(), max_vocab_size=20000)

    data_dict_pretrained = prepare_data_for_dl(
        df,
        text_column='processed_text',
        label_column='ground_truth',
        embedding_manager=embedding_manager,
        max_seq_length=200
    )

    # Prepare data without embeddings for non-pretrained models
    logger.info("Preparing data without embeddings...")
    data_dict_scratch = prepare_data_for_dl(
        df,
        text_column='processed_text',
        label_column='ground_truth',
        embedding_manager=None,
        max_seq_length=200
    )

    # Models to evaluate
    models_to_evaluate = [
        # TensorFlow models
        ("cnn_tensorflow.keras", "CNN (TensorFlow)", "tensorflow", data_dict_scratch),
        ("cnn_tensorflow_pretrained.keras", "CNN + GloVe (TensorFlow)", "tensorflow", data_dict_pretrained),

        # PyTorch models
        ("cnn_pytorch.pt", "CNN (PyTorch)", "pytorch", data_dict_scratch),
        ("cnn_pytorch_pretrained.pt", "CNN + GloVe (PyTorch)", "pytorch", data_dict_pretrained),
        ("lstm_pytorch_pretrained.pt", "BiLSTM + GloVe (PyTorch)", "pytorch", data_dict_pretrained),
    ]

    success_count = 0
    total_count = 0

    for model_file, model_name, framework, data_dict in models_to_evaluate:
        model_path = MODELS_DIR / model_file

        if not model_path.exists():
            logger.warning(f"✗ Model not found: {model_path}")
            continue

        total_count += 1

        try:
            if framework == 'tensorflow':
                if not TF_AVAILABLE:
                    logger.warning(f"✗ TensorFlow not available, skipping {model_name}")
                    continue
                evaluate_tensorflow_model(model_path, model_name, data_dict)
            elif framework == 'pytorch':
                evaluate_pytorch_model(model_path, model_name, data_dict)

            success_count += 1

        except Exception as e:
            logger.error(f"✗ Failed to evaluate {model_name}: {e}")
            import traceback
            traceback.print_exc()

    logger.info("=" * 60)
    logger.info(f"✓ Successfully regenerated metrics for {success_count}/{total_count} models")
    logger.info(f"Results saved to: {RESULTS_DIR}")
    logger.info("")
    logger.info("Next step: Restart the Streamlit dashboard to see the metrics")
    logger.info("  streamlit run app.py")


if __name__ == "__main__":
    main()

"""
Model evaluation and comparison module.
Collects predictions from all models and generates comprehensive metrics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    cohen_kappa_score,
    matthews_corrcoef,
)
import json


RESULTS_DIR = Path(__file__).parent.parent / "results"


def compute_metrics(y_true, y_pred, model_name="Model"):
    """
    Compute comprehensive evaluation metrics for a model.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        model_name: Name of the model
    
    Returns:
        Dictionary with all metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    
    precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    
    try:
        kappa = cohen_kappa_score(y_true, y_pred)
    except ValueError:
        kappa = 0.0
    
    try:
        mcc = matthews_corrcoef(y_true, y_pred)
    except ValueError:
        mcc = 0.0
    
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    labels = sorted(set(y_true) | set(y_pred))
    
    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "precision_weighted": precision_w,
        "recall_weighted": recall_w,
        "f1_weighted": f1_w,
        "precision_macro": precision_m,
        "recall_macro": recall_m,
        "f1_macro": f1_m,
        "cohen_kappa": kappa,
        "matthews_corrcoef": mcc,
        "confusion_matrix": conf_matrix,
        "classification_report": report,
        "labels": labels,
        "predictions": y_pred,
        "n_samples": len(y_true),
    }


def evaluate_vader_predictions(df, text_column="review_text", 
                               label_column="ground_truth"):
    """Evaluate VADER sentiment predictions."""
    from .sentiment_analyzer import predict_sentiment_vader
    
    texts = df[text_column].tolist()
    y_true = df[label_column].tolist()
    
    y_pred = predict_sentiment_vader(texts)
    
    return compute_metrics(y_true, y_pred, "VADER")


def evaluate_textblob_predictions(df, text_column="review_text", 
                                  label_column="ground_truth"):
    """Evaluate TextBlob sentiment predictions."""
    from .sentiment_analyzer import predict_sentiment_textblob
    
    texts = df[text_column].tolist()
    y_true = df[label_column].tolist()
    
    y_pred = predict_sentiment_textblob(texts)
    
    return compute_metrics(y_true, y_pred, "TextBlob")


def evaluate_ensemble_predictions(df, label_column="ground_truth"):
    """Evaluate ensemble (VADER + TextBlob) predictions."""
    if "sentiment_label" not in df.columns:
        raise ValueError("Run sentiment analysis first to get ensemble predictions")
    
    y_true = df[label_column].tolist()
    y_pred = df["sentiment_label"].tolist()
    
    return compute_metrics(y_true, y_pred, "Ensemble (VADER+TextBlob)")


def evaluate_ml_model(pipeline, X_test, y_test, model_name="ML Model"):
    """Evaluate a trained ML model pipeline."""
    y_pred = pipeline.predict(X_test)
    return compute_metrics(y_test, y_pred, model_name)


def evaluate_transformer_model(df, text_column="cleaned_text",
                               label_column="ground_truth",
                               model_dir=None):
    """Evaluate a fine-tuned transformer model."""
    from .transformer_model import evaluate_distilbert, is_transformer_available

    if not is_transformer_available():
        return None

    return evaluate_distilbert(
        df, text_column, label_column, model_dir=model_dir, verbose=False
    )


def evaluate_dl_model(model, history, framework, model_name="DL Model"):
    """
    Evaluate a deep learning model using pre-computed results.

    Args:
        model: Trained TensorFlow or PyTorch model
        history: Training history containing test results
        framework: 'tensorflow' or 'pytorch'
        model_name: Display name for the model

    Returns:
        Dictionary with evaluation metrics
    """
    # Extract metrics from training history
    # Deep learning models have already been evaluated during training
    # We use those results here

    test_accuracy = history.get('test_accuracy', 0.0)
    test_loss = history.get('test_loss', 0.0)

    # If we don't have full evaluation metrics, create a simplified result
    # In practice, you'd want to re-run evaluation with full metrics
    return {
        "model_name": model_name,
        "accuracy": test_accuracy,
        "test_loss": test_loss,
        "framework": framework,
        "n_samples": "N/A",  # Would need to recalculate from test set
        # Note: For full metrics, we'd need to re-run predictions on test set
        # This is a simplified version for integration purposes
    }


def compare_all_models(
    df,
    ml_results=None,
    dl_results=None,
    transformer_results=None,
    text_column="review_text",
    processed_column="processed_text",
    cleaned_column="cleaned_text",
    label_column="ground_truth",
    include_transformer=False,
    verbose=True,
):
    """
    Compare all available models on the same test data.

    Args:
        df: DataFrame with text and labels
        ml_results: Pre-computed ML model results (from ml_models.py)
        dl_results: Pre-computed DL model results (from dl_trainer.py)
        transformer_results: Pre-computed transformer results
        text_column: Column with original text
        processed_column: Column with preprocessed text
        cleaned_column: Column with cleaned text
        label_column: Column with ground truth labels
        include_transformer: Whether to include transformer evaluation
        verbose: Show progress

    Returns:
        Dictionary with all model results and comparison
    """
    if verbose:
        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)
    
    all_results = {}
    
    if verbose:
        print("\nEvaluating VADER...")
    try:
        vader_results = evaluate_vader_predictions(df, text_column, label_column)
        all_results["vader"] = vader_results
        if verbose:
            print(f"  VADER Accuracy: {vader_results['accuracy']:.4f}")
    except Exception as e:
        if verbose:
            print(f"  VADER evaluation failed: {e}")
    
    if verbose:
        print("Evaluating TextBlob...")
    try:
        textblob_results = evaluate_textblob_predictions(df, text_column, label_column)
        all_results["textblob"] = textblob_results
        if verbose:
            print(f"  TextBlob Accuracy: {textblob_results['accuracy']:.4f}")
    except Exception as e:
        if verbose:
            print(f"  TextBlob evaluation failed: {e}")
    
    if "sentiment_label" in df.columns:
        if verbose:
            print("Evaluating Ensemble...")
        try:
            ensemble_results = evaluate_ensemble_predictions(df, label_column)
            all_results["ensemble"] = ensemble_results
            if verbose:
                print(f"  Ensemble Accuracy: {ensemble_results['accuracy']:.4f}")
        except Exception as e:
            if verbose:
                print(f"  Ensemble evaluation failed: {e}")
    
    if ml_results:
        if verbose:
            print("Adding ML model results...")
        for model_type, results in ml_results.items():
            if "pipeline" in results and "predictions" in results:
                all_results[model_type] = results
                if verbose:
                    print(f"  {results['model_name']} Accuracy: {results['accuracy']:.4f}")

    # Add deep learning model results
    if dl_results:
        if verbose:
            print("Adding deep learning model results...")
        for model_key, dl_data in dl_results.items():
            try:
                model = dl_data.get('model')
                history = dl_data.get('history')
                framework = dl_data.get('framework', 'unknown')

                dl_metrics = evaluate_dl_model(
                    model=model,
                    history=history,
                    framework=framework,
                    model_name=model_key
                )

                all_results[model_key] = dl_metrics
                if verbose:
                    print(f"  {model_key} Accuracy: {dl_metrics['accuracy']:.4f}")
            except Exception as e:
                if verbose:
                    print(f"  Error adding {model_key}: {e}")

    if include_transformer and transformer_results:
        all_results["distilbert"] = transformer_results
        if verbose:
            print(f"  DistilBERT Accuracy: {transformer_results['accuracy']:.4f}")
    
    comparison_df = create_comparison_dataframe(all_results)
    
    if verbose:
        print("\n" + "-" * 60)
        print("COMPARISON SUMMARY")
        print("-" * 60)
        print(comparison_df.to_string(index=False))
    
    best_models = identify_best_models(all_results)
    
    if verbose:
        print("\n" + "-" * 60)
        print("BEST MODELS")
        print("-" * 60)
        for metric, model_info in best_models.items():
            print(f"  {metric}: {model_info['model']} ({model_info['value']:.4f})")
    
    return {
        "model_results": all_results,
        "comparison_df": comparison_df,
        "best_models": best_models,
    }


def create_comparison_dataframe(all_results):
    """Create a DataFrame comparing all models."""
    rows = []
    
    for model_key, results in all_results.items():
        row = {
            "Model": results.get("model_name", model_key),
            "Accuracy": results.get("accuracy", 0),
            "F1 (weighted)": results.get("f1_weighted", 0),
            "F1 (macro)": results.get("f1_macro", 0),
            "Precision": results.get("precision_weighted", 0),
            "Recall": results.get("recall_weighted", 0),
            "Cohen's Kappa": results.get("cohen_kappa", 0),
            "MCC": results.get("matthews_corrcoef", 0),
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.sort_values("F1 (weighted)", ascending=False).reset_index(drop=True)
    
    return df


def identify_best_models(all_results):
    """Identify the best model for each metric."""
    metrics = [
        "accuracy", "f1_weighted", "f1_macro", 
        "precision_weighted", "recall_weighted",
        "cohen_kappa", "matthews_corrcoef",
    ]
    
    best_models = {}
    
    for metric in metrics:
        best_model = None
        best_value = -float("inf")
        
        for model_key, results in all_results.items():
            value = results.get(metric, 0)
            if value > best_value:
                best_value = value
                best_model = results.get("model_name", model_key)
        
        if best_model:
            best_models[metric] = {
                "model": best_model,
                "value": best_value,
            }
    
    return best_models


def get_per_class_metrics(all_results):
    """Extract per-class metrics for all models."""
    per_class = {}
    
    for model_key, results in all_results.items():
        model_name = results.get("model_name", model_key)
        report = results.get("classification_report", {})
        
        per_class[model_name] = {}
        for label in results.get("labels", []):
            if str(label) in report:
                per_class[model_name][label] = {
                    "precision": report[str(label)].get("precision", 0),
                    "recall": report[str(label)].get("recall", 0),
                    "f1": report[str(label)].get("f1-score", 0),
                    "support": report[str(label)].get("support", 0),
                }
    
    return per_class


def save_evaluation_results(results, output_dir=None, prefix="evaluation"):
    """Save evaluation results to files."""
    if output_dir is None:
        output_dir = RESULTS_DIR
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    comparison_df = results.get("comparison_df")
    if comparison_df is not None:
        comparison_path = output_dir / f"{prefix}_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        print(f"Saved comparison to: {comparison_path}")
    
    summary = {
        "best_models": results.get("best_models", {}),
        "models_evaluated": list(results.get("model_results", {}).keys()),
    }
    
    for model_key, model_results in results.get("model_results", {}).items():
        summary[model_key] = {
            "model_name": model_results.get("model_name"),
            "accuracy": model_results.get("accuracy"),
            "f1_weighted": model_results.get("f1_weighted"),
            "f1_macro": model_results.get("f1_macro"),
            "n_samples": model_results.get("n_samples"),
        }
        
        conf_matrix = model_results.get("confusion_matrix")
        if conf_matrix is not None:
            cm_path = output_dir / f"{prefix}_confusion_matrix_{model_key}.csv"
            labels = model_results.get("labels", [])
            cm_df = pd.DataFrame(conf_matrix, index=labels, columns=labels)
            cm_df.to_csv(cm_path)
    
    summary_path = output_dir / f"{prefix}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved summary to: {summary_path}")
    
    return output_dir


def load_evaluation_results(results_dir=None, prefix="evaluation"):
    """Load saved evaluation results."""
    if results_dir is None:
        results_dir = RESULTS_DIR
    
    results_dir = Path(results_dir)
    
    results = {}
    
    comparison_path = results_dir / f"{prefix}_comparison.csv"
    if comparison_path.exists():
        results["comparison_df"] = pd.read_csv(comparison_path)
    
    summary_path = results_dir / f"{prefix}_summary.json"
    if summary_path.exists():
        with open(summary_path, "r") as f:
            results["summary"] = json.load(f)
    
    return results


def print_detailed_report(results):
    """Print a detailed evaluation report."""
    print("\n" + "=" * 70)
    print("DETAILED MODEL EVALUATION REPORT")
    print("=" * 70)
    
    for model_key, model_results in results.get("model_results", {}).items():
        model_name = model_results.get("model_name", model_key)
        
        print(f"\n{'-' * 70}")
        print(f"MODEL: {model_name}")
        print(f"{'-' * 70}")
        
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:          {model_results.get('accuracy', 0):.4f}")
        print(f"  F1 (weighted):     {model_results.get('f1_weighted', 0):.4f}")
        print(f"  F1 (macro):        {model_results.get('f1_macro', 0):.4f}")
        print(f"  Precision:         {model_results.get('precision_weighted', 0):.4f}")
        print(f"  Recall:            {model_results.get('recall_weighted', 0):.4f}")
        print(f"  Cohen's Kappa:     {model_results.get('cohen_kappa', 0):.4f}")
        print(f"  MCC:               {model_results.get('matthews_corrcoef', 0):.4f}")
        
        report = model_results.get("classification_report", {})
        labels = model_results.get("labels", [])
        
        if labels and report:
            print(f"\nPer-Class Metrics:")
            print(f"  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
            print(f"  {'-' * 52}")
            
            for label in labels:
                if str(label) in report:
                    metrics = report[str(label)]
                    print(f"  {label:<12} {metrics['precision']:>10.4f} "
                          f"{metrics['recall']:>10.4f} {metrics['f1-score']:>10.4f} "
                          f"{metrics['support']:>10}")
        
        conf_matrix = model_results.get("confusion_matrix")
        if conf_matrix is not None:
            print(f"\nConfusion Matrix:")
            print(f"  Predicted:  {' '.join([f'{l:>10}' for l in labels])}")
            print(f"  Actual")
            for i, label in enumerate(labels):
                row = conf_matrix[i]
                print(f"  {label:<10} {' '.join([f'{v:>10}' for v in row])}")
    
    best_models = results.get("best_models", {})
    if best_models:
        print(f"\n{'=' * 70}")
        print("BEST MODEL PER METRIC")
        print(f"{'=' * 70}")
        for metric, info in best_models.items():
            print(f"  {metric:<20}: {info['model']} ({info['value']:.4f})")


if __name__ == "__main__":
    data_path = Path(__file__).parent.parent / "data" / "reviews_with_sentiment.csv"
    
    if data_path.exists():
        print(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        
        if "ground_truth" not in df.columns:
            from .preprocessor import rating_to_sentiment
            df["ground_truth"] = df["rating"].apply(rating_to_sentiment)
        
        results = compare_all_models(
            df,
            text_column="review_text",
            label_column="ground_truth",
            verbose=True,
        )
        
        print_detailed_report(results)
        
        save_evaluation_results(results)
    else:
        print(f"Data file not found: {data_path}")
        print("Run the main pipeline first to generate analyzed data.")

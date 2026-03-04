"""
Classical ML models for sentiment classification.
Implements TF-IDF vectorization with Logistic Regression and Naive Bayes.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from tqdm import tqdm


MODELS_DIR = Path(__file__).parent.parent / "models"

TFIDF_CONFIG = {
    "max_features": 10000,
    "ngram_range": (1, 2),
    "min_df": 5,
    "max_df": 0.95,
    "sublinear_tf": True,
}


def create_tfidf_vectorizer(**kwargs):
    """Create TF-IDF vectorizer with optimal settings for sentiment analysis."""
    config = {**TFIDF_CONFIG, **kwargs}
    return TfidfVectorizer(**config)


def create_logistic_regression():
    """Create Logistic Regression classifier optimized for text classification."""
    return LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
        n_jobs=-1,
    )


def create_naive_bayes():
    """Create Multinomial Naive Bayes classifier."""
    return MultinomialNB(alpha=0.1)


def create_svm():
    """Create Linear SVM classifier."""
    return LinearSVC(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )


def create_random_forest():
    """Create Random Forest classifier."""
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=50,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


def get_model_config():
    """Get all available model configurations."""
    return {
        "logistic_regression": {
            "name": "Logistic Regression",
            "model": create_logistic_regression,
            "description": "Linear model with L2 regularization",
        },
        "naive_bayes": {
            "name": "Naive Bayes",
            "model": create_naive_bayes,
            "description": "Probabilistic classifier based on Bayes theorem",
        },
        "svm": {
            "name": "Linear SVM",
            "model": create_svm,
            "description": "Support Vector Machine with linear kernel",
        },
        "random_forest": {
            "name": "Random Forest",
            "model": create_random_forest,
            "description": "Ensemble of decision trees",
        },
    }


def prepare_data(df, text_column="processed_text", label_column="ground_truth", 
                 test_size=0.2, random_state=42):
    """
    Prepare data for training with train/test split.
    
    Args:
        df: DataFrame with text and labels
        text_column: Column containing preprocessed text
        label_column: Column containing labels
        test_size: Fraction of data for testing
        random_state: Random seed for reproducibility
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    df = df.dropna(subset=[text_column, label_column])
    
    X = df[text_column].values
    y = df[label_column].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set: {len(X_train):,} samples")
    print(f"Test set: {len(X_test):,} samples")
    print(f"Label distribution (train): {pd.Series(y_train).value_counts().to_dict()}")
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, model_type="logistic_regression", verbose=True):
    """
    Train a single ML model with TF-IDF features.
    
    Args:
        X_train: Training text data
        y_train: Training labels
        model_type: Type of model to train
        verbose: Show progress
    
    Returns:
        Trained pipeline (TF-IDF + Classifier)
    """
    model_configs = get_model_config()
    
    if model_type not in model_configs:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(model_configs.keys())}")
    
    config = model_configs[model_type]
    
    if verbose:
        print(f"\nTraining {config['name']}...")
    
    pipeline = Pipeline([
        ("tfidf", create_tfidf_vectorizer()),
        ("classifier", config["model"]()),
    ])
    
    pipeline.fit(X_train, y_train)
    
    if verbose:
        print(f"  {config['name']} training complete")
    
    return pipeline


def evaluate_model(pipeline, X_test, y_test, model_name="Model"):
    """
    Evaluate a trained model on test data.
    
    Args:
        pipeline: Trained sklearn pipeline
        X_test: Test text data
        y_test: True labels
        model_name: Name for display
    
    Returns:
        Dictionary with evaluation metrics
    """
    y_pred = pipeline.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average="weighted"
    )
    
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro"
    )
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    report = classification_report(y_test, y_pred, output_dict=True)
    
    results = {
        "model_name": model_name,
        "accuracy": accuracy,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "confusion_matrix": conf_matrix,
        "classification_report": report,
        "predictions": y_pred,
        "labels": sorted(set(y_test)),
    }
    
    return results


def cross_validate_model(pipeline, X, y, cv=5, verbose=True):
    """
    Perform cross-validation on a model.
    
    Args:
        pipeline: sklearn pipeline
        X: Features (text)
        y: Labels
        cv: Number of folds
        verbose: Show progress
    
    Returns:
        Dictionary with CV scores
    """
    if verbose:
        print(f"Running {cv}-fold cross-validation...")
    
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    
    cv_results = {
        "cv_accuracy_mean": scores.mean(),
        "cv_accuracy_std": scores.std(),
        "cv_scores": scores,
    }
    
    if verbose:
        print(f"  CV Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    return cv_results


def train_all_models(X_train, X_test, y_train, y_test, 
                     model_types=None, verbose=True):
    """
    Train and evaluate all specified models.
    
    Args:
        X_train, X_test: Train/test text data
        y_train, y_test: Train/test labels
        model_types: List of model types to train (default: all)
        verbose: Show progress
    
    Returns:
        Dictionary with trained models and their results
    """
    if model_types is None:
        model_types = ["logistic_regression", "naive_bayes"]
    
    model_configs = get_model_config()
    all_results = {}
    
    for model_type in model_types:
        if model_type not in model_configs:
            print(f"Warning: Unknown model type '{model_type}', skipping")
            continue
        
        config = model_configs[model_type]
        
        pipeline = train_model(X_train, y_train, model_type, verbose)
        
        results = evaluate_model(pipeline, X_test, y_test, config["name"])
        
        results["pipeline"] = pipeline
        results["model_type"] = model_type
        
        all_results[model_type] = results
        
        if verbose:
            print(f"\n  {config['name']} Results:")
            print(f"    Accuracy: {results['accuracy']:.4f}")
            print(f"    F1 (weighted): {results['f1_weighted']:.4f}")
            print(f"    F1 (macro): {results['f1_macro']:.4f}")
    
    return all_results


def save_model(pipeline, model_name, output_dir=None):
    """Save trained model to disk."""
    if output_dir is None:
        output_dir = MODELS_DIR
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / f"{model_name}.joblib"
    joblib.dump(pipeline, model_path)
    
    print(f"Saved model to: {model_path}")
    return model_path


def load_model(model_name, models_dir=None):
    """Load trained model from disk."""
    if models_dir is None:
        models_dir = MODELS_DIR
    
    model_path = Path(models_dir) / f"{model_name}.joblib"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    return joblib.load(model_path)


def predict(pipeline, texts):
    """Make predictions on new texts."""
    if isinstance(texts, str):
        texts = [texts]
    
    predictions = pipeline.predict(texts)
    
    if hasattr(pipeline, "predict_proba"):
        try:
            probabilities = pipeline.predict_proba(texts)
            return predictions, probabilities
        except AttributeError:
            pass
    
    return predictions, None


def get_feature_importance(pipeline, top_n=20):
    """Extract most important features from the model."""
    tfidf = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]
    
    feature_names = tfidf.get_feature_names_out()
    
    if hasattr(classifier, "coef_"):
        coefficients = classifier.coef_
        
        if coefficients.ndim == 1:
            importance_df = pd.DataFrame({
                "feature": feature_names,
                "importance": np.abs(coefficients),
            }).sort_values("importance", ascending=False).head(top_n)
        else:
            importance_dfs = {}
            classes = classifier.classes_ if hasattr(classifier, "classes_") else range(coefficients.shape[0])
            
            for i, cls in enumerate(classes):
                top_pos = pd.DataFrame({
                    "feature": feature_names,
                    "importance": coefficients[i],
                }).nlargest(top_n, "importance")
                
                top_neg = pd.DataFrame({
                    "feature": feature_names,
                    "importance": coefficients[i],
                }).nsmallest(top_n, "importance")
                
                importance_dfs[f"{cls}_positive"] = top_pos
                importance_dfs[f"{cls}_negative"] = top_neg
            
            return importance_dfs
    
    elif hasattr(classifier, "feature_importances_"):
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": classifier.feature_importances_,
        }).sort_values("importance", ascending=False).head(top_n)
    
    else:
        return None
    
    return importance_df


def print_classification_report(results):
    """Print formatted classification report."""
    report = results["classification_report"]
    
    print(f"\n{'='*60}")
    print(f"Classification Report: {results['model_name']}")
    print(f"{'='*60}")
    
    for label in results["labels"]:
        if str(label) in report:
            metrics = report[str(label)]
            print(f"\n{label}:")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1-score']:.4f}")
            print(f"  Support:   {metrics['support']}")
    
    print(f"\nOverall Accuracy: {results['accuracy']:.4f}")
    print(f"Weighted F1:      {results['f1_weighted']:.4f}")
    print(f"Macro F1:         {results['f1_macro']:.4f}")


def run_ml_pipeline(df, text_column="processed_text", label_column="ground_truth",
                    model_types=None, save_models=True, verbose=True):
    """
    Run the complete ML training pipeline.
    
    Args:
        df: DataFrame with text and labels
        text_column: Column containing preprocessed text
        label_column: Column containing sentiment labels
        model_types: List of model types to train
        save_models: Whether to save trained models
        verbose: Show progress
    
    Returns:
        Dictionary with all results and trained models
    """
    if model_types is None:
        model_types = ["logistic_regression", "naive_bayes"]
    
    if verbose:
        print("\n" + "=" * 60)
        print("ML SENTIMENT CLASSIFICATION PIPELINE")
        print("=" * 60)
    
    X_train, X_test, y_train, y_test = prepare_data(
        df, text_column, label_column
    )
    
    all_results = train_all_models(
        X_train, X_test, y_train, y_test, model_types, verbose
    )
    
    if save_models:
        for model_type, results in all_results.items():
            save_model(results["pipeline"], model_type)
    
    if verbose:
        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)
        
        comparison_df = pd.DataFrame([
            {
                "Model": r["model_name"],
                "Accuracy": r["accuracy"],
                "F1 (weighted)": r["f1_weighted"],
                "F1 (macro)": r["f1_macro"],
                "Precision": r["precision_weighted"],
                "Recall": r["recall_weighted"],
            }
            for r in all_results.values()
        ]).sort_values("F1 (weighted)", ascending=False)
        
        print(comparison_df.to_string(index=False))
    
    results_output = {
        "models": all_results,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }
    
    return results_output


if __name__ == "__main__":
    data_path = Path(__file__).parent.parent / "data" / "preprocessed_reviews.csv"
    
    if data_path.exists():
        print(f"Loading preprocessed data from {data_path}")
        df = pd.read_csv(data_path)
        
        results = run_ml_pipeline(
            df,
            text_column="processed_text",
            label_column="ground_truth",
            model_types=["logistic_regression", "naive_bayes"],
            save_models=True,
            verbose=True,
        )
        
        for model_type, model_results in results["models"].items():
            print_classification_report(model_results)
        
        print("\n" + "=" * 60)
        print("TOP FEATURES (Logistic Regression)")
        print("=" * 60)
        
        lr_results = results["models"].get("logistic_regression")
        if lr_results:
            features = get_feature_importance(lr_results["pipeline"], top_n=15)
            if isinstance(features, dict):
                for key, feat_df in features.items():
                    print(f"\n{key}:")
                    for _, row in feat_df.head(10).iterrows():
                        print(f"  {row['feature']}: {row['importance']:.4f}")
            elif features is not None:
                print(features.to_string(index=False))
    else:
        print(f"Preprocessed data not found: {data_path}")
        print("Run preprocessor.py first to prepare the data.")

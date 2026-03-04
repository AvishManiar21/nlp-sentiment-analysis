"""
Transformer-based sentiment classification using DistilBERT.
Fine-tunes a pre-trained model for sentiment analysis.
Includes GPU detection and optional CPU fallback.
"""

import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

TRANSFORMERS_AVAILABLE = False
TORCH_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    pass

try:
    from transformers import (
        DistilBertTokenizer,
        DistilBertForSequenceClassification,
        Trainer,
        TrainingArguments,
        EarlyStoppingCallback,
    )
    from datasets import Dataset
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass


MODEL_NAME = "distilbert-base-uncased"
MODELS_DIR = Path(__file__).parent.parent / "models" / "distilbert-sentiment"
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5

LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}


def check_dependencies():
    """Check if required dependencies are available."""
    missing = []
    
    if not TORCH_AVAILABLE:
        missing.append("torch")
    if not TRANSFORMERS_AVAILABLE:
        missing.append("transformers")
    
    if missing:
        raise ImportError(
            f"Missing required packages: {', '.join(missing)}. "
            f"Install with: pip install {' '.join(missing)}"
        )
    
    return True


def get_device():
    """Detect and return the best available device."""
    if not TORCH_AVAILABLE:
        return None
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return device
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
        return device
    else:
        device = torch.device("cpu")
        print("Using CPU (training will be slower)")
        return device


def create_dataset(texts, labels, tokenizer, max_length=MAX_LENGTH):
    """Create a HuggingFace Dataset from texts and labels."""
    encoded = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors=None,
    )
    
    label_ids = [LABEL_MAP.get(label, 1) for label in labels]
    
    dataset = Dataset.from_dict({
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": label_ids,
    })
    
    return dataset


def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted"
    )
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def train_distilbert(
    df,
    text_column="cleaned_text",
    label_column="ground_truth",
    output_dir=None,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    max_length=MAX_LENGTH,
    use_early_stopping=True,
    verbose=True,
):
    """
    Fine-tune DistilBERT for sentiment classification.
    
    Args:
        df: DataFrame with text and labels
        text_column: Column containing text
        label_column: Column containing labels
        output_dir: Directory to save the model
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        max_length: Maximum sequence length
        use_early_stopping: Use early stopping
        verbose: Show training progress
    
    Returns:
        Dictionary with trained model, tokenizer, and evaluation results
    """
    check_dependencies()
    
    if output_dir is None:
        output_dir = MODELS_DIR
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = get_device()
    
    if verbose:
        print("\n" + "=" * 60)
        print("DISTILBERT FINE-TUNING")
        print("=" * 60)
        print(f"Model: {MODEL_NAME}")
        print(f"Max length: {max_length}")
        print(f"Batch size: {batch_size}")
        print(f"Epochs: {epochs}")
        print(f"Learning rate: {learning_rate}")
    
    df = df.dropna(subset=[text_column, label_column])
    texts = df[text_column].tolist()
    labels = df[label_column].tolist()
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    if verbose:
        print(f"\nTraining samples: {len(train_texts):,}")
        print(f"Validation samples: {len(val_texts):,}")
    
    if verbose:
        print("\nLoading tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    
    if verbose:
        print("Tokenizing datasets...")
    train_dataset = create_dataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = create_dataset(val_texts, val_labels, tokenizer, max_length)
    
    if verbose:
        print("Loading pre-trained model...")
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_MAP),
        id2label=ID_TO_LABEL,
        label2id=LABEL_MAP,
    )
    
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        warmup_steps=500,
        weight_decay=0.01,
        learning_rate=learning_rate,
        logging_dir=str(output_dir / "logs"),
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
    )
    
    callbacks = []
    if use_early_stopping:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=2))
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    
    if verbose:
        print("\nStarting training...")
    
    trainer.train()
    
    if verbose:
        print("\nEvaluating model...")
    
    eval_results = trainer.evaluate()
    
    if verbose:
        print("\nSaving model...")
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    if verbose:
        print(f"\nModel saved to: {output_dir}")
        print("\nEvaluation Results:")
        for key, value in eval_results.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
    
    return {
        "model": model,
        "tokenizer": tokenizer,
        "trainer": trainer,
        "eval_results": eval_results,
        "output_dir": output_dir,
    }


def load_distilbert(model_dir=None):
    """Load a fine-tuned DistilBERT model."""
    check_dependencies()
    
    if model_dir is None:
        model_dir = MODELS_DIR
    
    model_dir = Path(model_dir)
    
    if not model_dir.exists():
        raise FileNotFoundError(f"Model not found: {model_dir}")
    
    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    
    device = get_device()
    model = model.to(device)
    model.eval()
    
    return model, tokenizer


def predict_distilbert(texts, model=None, tokenizer=None, model_dir=None, 
                       batch_size=32, verbose=True):
    """
    Make predictions using fine-tuned DistilBERT.
    
    Args:
        texts: List of texts to classify
        model: Pre-loaded model (optional)
        tokenizer: Pre-loaded tokenizer (optional)
        model_dir: Directory containing the model
        batch_size: Batch size for inference
        verbose: Show progress
    
    Returns:
        Dictionary with predictions and probabilities
    """
    check_dependencies()
    
    if model is None or tokenizer is None:
        model, tokenizer = load_distilbert(model_dir)
    
    device = next(model.parameters()).device
    
    if isinstance(texts, str):
        texts = [texts]
    
    all_predictions = []
    all_probabilities = []
    
    from tqdm import tqdm
    
    num_batches = (len(texts) + batch_size - 1) // batch_size
    iterator = range(0, len(texts), batch_size)
    if verbose:
        iterator = tqdm(iterator, total=num_batches, desc="DistilBERT inference")
    
    with torch.no_grad():
        for i in iterator:
            batch_texts = texts[i:i + batch_size]
            
            encoded = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            
            encoded = {k: v.to(device) for k, v in encoded.items()}
            
            outputs = model(**encoded)
            logits = outputs.logits
            
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            
            all_predictions.extend([ID_TO_LABEL[p.item()] for p in preds])
            all_probabilities.extend(probs.cpu().numpy())
    
    return {
        "predictions": all_predictions,
        "probabilities": np.array(all_probabilities),
        "label_map": LABEL_MAP,
    }


def evaluate_distilbert(df, text_column="cleaned_text", label_column="ground_truth",
                        model=None, tokenizer=None, model_dir=None, verbose=True):
    """
    Evaluate DistilBERT on a test set.
    
    Args:
        df: DataFrame with text and labels
        text_column: Column containing text
        label_column: Column containing ground truth labels
        model: Pre-loaded model (optional)
        tokenizer: Pre-loaded tokenizer (optional)
        model_dir: Directory containing the model
        verbose: Show progress
    
    Returns:
        Dictionary with evaluation metrics
    """
    check_dependencies()
    
    df = df.dropna(subset=[text_column, label_column])
    texts = df[text_column].tolist()
    y_true = df[label_column].tolist()
    
    results = predict_distilbert(
        texts, model, tokenizer, model_dir, verbose=verbose
    )
    y_pred = results["predictions"]
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro"
    )
    
    conf_matrix = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    
    evaluation = {
        "model_name": "DistilBERT",
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
        "probabilities": results["probabilities"],
        "labels": sorted(set(y_true)),
    }
    
    if verbose:
        print("\n" + "=" * 60)
        print("DISTILBERT EVALUATION")
        print("=" * 60)
        print(f"Accuracy:     {accuracy:.4f}")
        print(f"F1 (weighted): {f1:.4f}")
        print(f"F1 (macro):    {f1_macro:.4f}")
        print(f"Precision:    {precision:.4f}")
        print(f"Recall:       {recall:.4f}")
    
    return evaluation


def is_transformer_available():
    """Check if transformer training is available."""
    return TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE


def get_gpu_info():
    """Get information about available GPU."""
    if not TORCH_AVAILABLE:
        return {"available": False, "reason": "PyTorch not installed"}
    
    if torch.cuda.is_available():
        return {
            "available": True,
            "type": "CUDA",
            "device_name": torch.cuda.get_device_name(0),
            "memory_total": torch.cuda.get_device_properties(0).total_memory / 1e9,
        }
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return {
            "available": True,
            "type": "MPS (Apple Silicon)",
            "device_name": "Apple GPU",
        }
    else:
        return {
            "available": False,
            "type": "CPU",
            "reason": "No GPU detected",
        }


if __name__ == "__main__":
    print("\nChecking transformer dependencies...")
    print(f"PyTorch available: {TORCH_AVAILABLE}")
    print(f"Transformers available: {TRANSFORMERS_AVAILABLE}")
    
    gpu_info = get_gpu_info()
    print(f"\nGPU Info: {gpu_info}")
    
    if not is_transformer_available():
        print("\nTransformer training not available. Install dependencies:")
        print("  pip install torch transformers datasets")
        exit(0)
    
    data_path = Path(__file__).parent.parent / "data" / "preprocessed_reviews.csv"
    
    if data_path.exists():
        print(f"\nLoading data from {data_path}")
        df = pd.read_csv(data_path)
        
        if len(df) > 10000:
            print(f"Sampling 10,000 reviews for faster training...")
            df = df.sample(n=10000, random_state=42)
        
        results = train_distilbert(
            df,
            text_column="cleaned_text",
            label_column="ground_truth",
            epochs=2,
            batch_size=16,
            verbose=True,
        )
        
        print("\n" + "=" * 60)
        print("Training complete!")
        print(f"Model saved to: {results['output_dir']}")
    else:
        print(f"\nData file not found: {data_path}")
        print("Run preprocessor.py first to prepare the data.")

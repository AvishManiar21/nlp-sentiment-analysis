"""
Deep Learning Training Pipeline

Unified training interface for TensorFlow and PyTorch models.
Handles data preparation, training loops, validation, and model persistence.
"""

import os
import logging
import json
from pathlib import Path
from typing import Tuple, Dict, Optional, Union
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

# TensorFlow imports (optional)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.callbacks import (
        EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
    )
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None
    keras = None

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Local imports
from src.dl_models import (
    TextCNNConfig,
    build_tensorflow_cnn,
    build_tensorflow_cnn_hybrid,
    TextCNNPyTorch,
    BiLSTMPyTorch,
    SentimentDataset,
    get_device,
    save_tensorflow_model,
    save_pytorch_model
)
from src.embedding_manager import EmbeddingManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_data_for_dl(
    df: pd.DataFrame,
    text_column: str = 'processed_text',
    label_column: str = 'ground_truth',
    embedding_manager: Optional[EmbeddingManager] = None,
    max_seq_length: int = 200,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict:
    """
    Prepare data for deep learning models.

    Args:
        df: DataFrame with text and labels
        text_column: Column containing text
        label_column: Column containing labels
        embedding_manager: EmbeddingManager instance (optional)
        max_seq_length: Maximum sequence length
        test_size: Fraction of data for testing
        random_state: Random seed

    Returns:
        Dictionary containing train/test splits and metadata
    """
    from sklearn.model_selection import train_test_split

    logger.info("Preparing data for deep learning...")

    # Extract texts and labels
    texts = df[text_column].values
    labels = df[label_column].values

    # Convert labels to integers if they're strings
    if labels.dtype == object:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        labels = le.fit_transform(labels)
        label_mapping = {i: label for i, label in enumerate(le.classes_)}
    else:
        label_mapping = None

    logger.info(f"Dataset size: {len(texts)}")
    logger.info(f"Number of classes: {len(np.unique(labels))}")

    # Convert texts to sequences
    if embedding_manager is not None:
        # Use embedding manager's vocabulary
        sequences = embedding_manager.texts_to_sequences(
            texts.tolist(),
            max_length=max_seq_length,
            padding='post',
            truncating='post'
        )
        vocab_size = len(embedding_manager.word2idx)
        embedding_matrix = embedding_manager.create_embedding_matrix()
        embedding_dim = embedding_manager.embedding_dim
    else:
        # Build vocabulary from scratch
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        tokenizer = Tokenizer(num_words=10000, oov_token='<UNK>')
        tokenizer.fit_on_texts(texts)

        sequences = tokenizer.texts_to_sequences(texts)
        sequences = pad_sequences(
            sequences,
            maxlen=max_seq_length,
            padding='post',
            truncating='post'
        )

        vocab_size = min(len(tokenizer.word_index) + 1, 10000)
        embedding_matrix = None
        embedding_dim = 128  # Default if not using pre-trained

    logger.info(f"Vocabulary size: {vocab_size}")
    logger.info(f"Max sequence length: {max_seq_length}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        sequences,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )

    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Test samples: {len(X_test)}")

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'vocab_size': vocab_size,
        'embedding_dim': embedding_dim,
        'embedding_matrix': embedding_matrix,
        'max_seq_length': max_seq_length,
        'num_classes': len(np.unique(labels)),
        'label_mapping': label_mapping
    }


# ============================================================================
# METRICS CALCULATION AND SAVING
# ============================================================================

def calculate_and_save_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    framework: str,
    test_loss: float,
    n_samples: int,
    label_mapping: Optional[Dict] = None,
    save_dir: str = 'results'
) -> Dict:
    """
    Calculate full evaluation metrics and save to JSON file.

    Args:
        y_true: True labels
        y_pred: Predicted labels (class indices)
        model_name: Name of the model
        framework: Framework used ('tensorflow' or 'pytorch')
        test_loss: Test loss value
        n_samples: Number of test samples
        label_mapping: Optional mapping of indices to label names
        save_dir: Directory to save results

    Returns:
        Dictionary containing all metrics
    """
    # Create results directory if it doesn't exist
    results_path = Path(save_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)

    # Calculate precision, recall, f1 for each class
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    # Calculate weighted and macro averages
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Generate classification report
    target_names = None
    if label_mapping:
        target_names = [label_mapping[i] for i in range(len(label_mapping))]

    report = classification_report(
        y_true, y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )

    # Compile results
    results = {
        "model_name": model_name,
        "framework": framework,
        "accuracy": float(accuracy),
        "test_loss": float(test_loss),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "n_samples": int(n_samples),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "per_class_metrics": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "f1": f1.tolist(),
            "support": support.tolist()
        },
        "label_mapping": label_mapping,
        "timestamp": datetime.now().isoformat()
    }

    # Save to JSON file
    # Create safe filename from model name
    safe_name = model_name.lower().replace(" ", "_").replace("+", "").replace("(", "").replace(")", "")
    results_file = results_path / f"dl_results_{safe_name}.json"

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"✓ Saved evaluation metrics to {results_file}")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  F1 (weighted): {f1_weighted:.4f}")
    logger.info(f"  F1 (macro): {f1_macro:.4f}")

    return results


# ============================================================================
# TENSORFLOW TRAINING
# ============================================================================

def train_tensorflow_model(
    data_dict: Dict,
    model_name: str = 'cnn',
    use_pretrained_embeddings: bool = False,
    epochs: int = 20,
    batch_size: int = 32,
    save_dir: str = 'models/dl',
    tensorboard_dir: str = 'logs/tensorboard'
):
    """
    Train a TensorFlow/Keras model.

    Note: TensorFlow is not available for Python 3.14+. Use PyTorch instead or downgrade to Python 3.10-3.12.

    Args:
        data_dict: Dictionary from prepare_data_for_dl()
        model_name: Model identifier
        use_pretrained_embeddings: Whether to use pre-trained embeddings
        epochs: Number of training epochs
        batch_size: Batch size
        save_dir: Directory to save models
        tensorboard_dir: Directory for TensorBoard logs

    Returns:
        Tuple of (trained_model, training_history)
    """
    if not TF_AVAILABLE:
        raise ImportError(
            "TensorFlow is not available. "
            "TensorFlow doesn't support Python 3.14 yet. "
            "Please use Python 3.10-3.12 for TensorFlow models, "
            "or use --dl-framework pytorch to train PyTorch models instead."
        )

    logger.info(f"Training TensorFlow {model_name} model...")

    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Create model configuration
    config = TextCNNConfig(
        vocab_size=data_dict['vocab_size'],
        embedding_dim=data_dict['embedding_dim'],
        num_classes=data_dict['num_classes'],
        max_seq_length=data_dict['max_seq_length'],
        embedding_matrix=data_dict['embedding_matrix'],
        embedding_trainable=not use_pretrained_embeddings
    )

    # Build model
    if use_pretrained_embeddings and data_dict['embedding_matrix'] is not None:
        model = build_tensorflow_cnn_hybrid(config)
        model_filename = f"{model_name}_pretrained"
    else:
        model = build_tensorflow_cnn(config)
        model_filename = f"{model_name}_scratch"

    # Callbacks
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=str(save_path / f"{model_filename}_best.keras"),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        TensorBoard(
            log_dir=f"{tensorboard_dir}/{model_filename}_{timestamp}",
            histogram_freq=1
        )
    ]

    # Train model
    logger.info("Starting training...")
    history = model.fit(
        data_dict['X_train'],
        data_dict['y_train'],
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Save final model
    save_tensorflow_model(model, str(save_path / model_filename))

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_loss, test_accuracy = model.evaluate(
        data_dict['X_test'],
        data_dict['y_test'],
        verbose=0
    )

    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_accuracy:.4f}")

    # Get predictions for full metrics calculation
    logger.info("Calculating full evaluation metrics...")
    y_pred_probs = model.predict(data_dict['X_test'], verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = data_dict['y_test']

    # Calculate and save full metrics
    metrics = calculate_and_save_metrics(
        y_true=y_true,
        y_pred=y_pred,
        model_name=model_name,
        framework='tensorflow',
        test_loss=test_loss,
        n_samples=len(y_true),
        label_mapping=data_dict.get('label_mapping')
    )

    # Return model and history
    history_dict = {
        'loss': history.history['loss'],
        'accuracy': history.history['accuracy'],
        'val_loss': history.history['val_loss'],
        'val_accuracy': history.history['val_accuracy'],
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'metrics': metrics  # Add full metrics to history
    }

    return model, history_dict


# ============================================================================
# PYTORCH TRAINING
# ============================================================================

def train_pytorch_model(
    data_dict: Dict,
    model_type: str = 'cnn',
    use_pretrained_embeddings: bool = False,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    save_dir: str = 'models/dl',
    tensorboard_dir: str = 'logs/tensorboard'
) -> Tuple[nn.Module, Dict]:
    """
    Train a PyTorch model.

    Args:
        data_dict: Dictionary from prepare_data_for_dl()
        model_type: 'cnn' or 'lstm'
        use_pretrained_embeddings: Whether to use pre-trained embeddings
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        save_dir: Directory to save models
        tensorboard_dir: Directory for TensorBoard logs

    Returns:
        Tuple of (trained_model, training_history)
    """
    logger.info(f"Training PyTorch {model_type} model...")

    # Get device
    device = get_device()

    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Create model configuration
    config = TextCNNConfig(
        vocab_size=data_dict['vocab_size'],
        embedding_dim=data_dict['embedding_dim'],
        num_classes=data_dict['num_classes'],
        max_seq_length=data_dict['max_seq_length'],
        embedding_matrix=data_dict['embedding_matrix'],
        embedding_trainable=not use_pretrained_embeddings
    )

    # Build model
    if model_type.lower() == 'cnn':
        model = TextCNNPyTorch(config)
        model_name = 'cnn_pytorch'
    elif model_type.lower() == 'lstm':
        model = BiLSTMPyTorch(config)
        model_name = 'lstm_pytorch'
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    if use_pretrained_embeddings:
        model_name += '_pretrained'
    else:
        model_name += '_scratch'

    model = model.to(device)

    # Create data loaders
    train_dataset = SentimentDataset(data_dict['X_train'], data_dict['y_train'])
    test_dataset = SentimentDataset(data_dict['X_test'], data_dict['y_test'])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3
    )

    # TensorBoard
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(f"{tensorboard_dir}/{model_name}_{timestamp}")

    # Training loop
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }

    best_val_accuracy = 0.0
    patience_counter = 0
    max_patience = 5

    logger.info("Starting training...")

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (sequences, labels) in enumerate(train_loader):
            sequences = sequences.to(device)
            labels = labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # Calculate training metrics
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for sequences, labels in test_loader:
                sequences = sequences.to(device)
                labels = labels.to(device)

                outputs = model(sequences)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(test_loader)
        val_accuracy = val_correct / val_total

        # Update learning rate
        scheduler.step(avg_val_loss)

        # Log metrics
        history['train_loss'].append(avg_train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_accuracy)

        # TensorBoard logging
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)

        # Print progress
        logger.info(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f}"
        )

        # Early stopping and model checkpoint
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0
            # Save best model
            save_pytorch_model(model, str(save_path / f"{model_name}_best"), config)
            logger.info(f"Best model saved (val_acc: {val_accuracy:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

    writer.close()

    # Save final model
    save_pytorch_model(model, str(save_path / model_name), config)

    # Final evaluation on test set
    logger.info("Final evaluation on test set...")
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    # Collect all predictions and labels for full metrics
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            outputs = model(sequences)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

            # Store predictions and labels
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_accuracy = test_correct / test_total
    avg_test_loss = test_loss / len(test_loader)

    history['test_loss'] = avg_test_loss
    history['test_accuracy'] = test_accuracy

    logger.info(f"Test Loss: {avg_test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_accuracy:.4f}")

    # Calculate and save full metrics
    logger.info("Calculating full evaluation metrics...")
    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)

    metrics = calculate_and_save_metrics(
        y_true=y_true,
        y_pred=y_pred,
        model_name=model_name,
        framework='pytorch',
        test_loss=avg_test_loss,
        n_samples=len(y_true),
        label_mapping=data_dict.get('label_mapping')
    )

    history['metrics'] = metrics  # Add full metrics to history

    return model, history


# ============================================================================
# UNIFIED TRAINING INTERFACE
# ============================================================================

def train_model(
    df: pd.DataFrame,
    framework: str = 'tensorflow',
    model_type: str = 'cnn',
    use_embeddings: bool = False,
    embedding_name: str = 'glove-wiki-gigaword-100',
    text_column: str = 'processed_text',
    label_column: str = 'ground_truth',
    **kwargs
) -> Tuple[Union[keras.Model, nn.Module], Dict]:
    """
    Unified interface for training deep learning models.

    Args:
        df: DataFrame with text and labels
        framework: 'tensorflow' or 'pytorch'
        model_type: 'cnn' or 'lstm' (pytorch only)
        use_embeddings: Whether to use pre-trained embeddings
        embedding_name: Name of embedding to use
        text_column: Column with text data
        label_column: Column with labels
        **kwargs: Additional arguments passed to training function

    Returns:
        Tuple of (model, history)
    """
    # Extract data preparation parameters from kwargs
    max_seq_length = kwargs.pop('max_seq_length', 200)
    max_vocab_size = kwargs.pop('max_vocab_size', 20000)

    # Initialize embedding manager if needed
    embedding_manager = None
    if use_embeddings:
        logger.info(f"Loading pre-trained embeddings: {embedding_name}")
        embedding_manager = EmbeddingManager()
        embedding_manager.load_embedding(embedding_name)
        embedding_manager.build_vocab(
            df[text_column].tolist(),
            max_vocab_size=max_vocab_size
        )

    # Prepare data
    data_dict = prepare_data_for_dl(
        df,
        text_column=text_column,
        label_column=label_column,
        embedding_manager=embedding_manager,
        max_seq_length=max_seq_length
    )

    # Train model
    if framework.lower() == 'tensorflow':
        if not TF_AVAILABLE:
            raise ImportError(
                "TensorFlow is not available. TensorFlow doesn't support Python 3.14 yet.\n"
                "Options:\n"
                "  1. Use --dl-framework pytorch instead\n"
                "  2. Downgrade to Python 3.10-3.12 to use TensorFlow"
            )
        model, history = train_tensorflow_model(
            data_dict,
            model_name=model_type,
            use_pretrained_embeddings=use_embeddings,
            **kwargs
        )
    elif framework.lower() == 'pytorch':
        model, history = train_pytorch_model(
            data_dict,
            model_type=model_type,
            use_pretrained_embeddings=use_embeddings,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported framework: {framework}")

    return model, history


if __name__ == "__main__":
    # Demo usage
    print("Deep Learning Training Pipeline")
    print("This module provides unified training for TensorFlow and PyTorch models")
    print("\nExample usage:")
    print("  model, history = train_model(")
    print("      df=your_dataframe,")
    print("      framework='tensorflow',")
    print("      model_type='cnn',")
    print("      use_embeddings=True,")
    print("      embedding_name='glove-wiki-gigaword-100',")
    print("      epochs=20,")
    print("      batch_size=32")
    print("  )")

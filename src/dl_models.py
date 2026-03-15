"""
Deep Learning Models for Sentiment Analysis

This module implements CNN and hybrid architectures using both
TensorFlow/Keras and PyTorch frameworks.
"""

import logging
from typing import Optional, List, Tuple, Dict
import numpy as np

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# TENSORFLOW / KERAS MODELS
# ============================================================================

class TextCNNConfig:
    """Configuration for CNN text classification models."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        num_classes: int = 3,
        max_seq_length: int = 200,
        filter_sizes: List[int] = None,
        num_filters: int = 128,
        dropout_rate: float = 0.5,
        l2_reg: float = 0.001,
        embedding_trainable: bool = True,
        embedding_matrix: Optional[np.ndarray] = None
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.max_seq_length = max_seq_length
        self.filter_sizes = filter_sizes or [3, 4, 5]  # Different n-gram sizes
        self.num_filters = num_filters
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.embedding_trainable = embedding_trainable
        self.embedding_matrix = embedding_matrix


def build_tensorflow_cnn(config: TextCNNConfig) -> keras.Model:
    """
    Build CNN model for text classification using TensorFlow/Keras.

    Architecture:
    - Embedding layer (trainable or pre-trained)
    - Multiple parallel 1D Conv layers with different kernel sizes (3,4,5-grams)
    - Global max pooling for each conv layer
    - Concatenate pooled features
    - Dropout
    - Dense layer with softmax activation

    Args:
        config: TextCNNConfig object

    Returns:
        Compiled Keras model
    """
    logger.info("Building TensorFlow CNN model...")

    # Input layer
    input_text = layers.Input(shape=(config.max_seq_length,), name='input_text')

    # Embedding layer
    if config.embedding_matrix is not None:
        # Use pre-trained embeddings
        logger.info("Using pre-trained embeddings")
        embedding = layers.Embedding(
            input_dim=config.vocab_size,
            output_dim=config.embedding_dim,
            weights=[config.embedding_matrix],
            trainable=config.embedding_trainable,
            mask_zero=True,
            name='embedding'
        )(input_text)
    else:
        # Learn embeddings from scratch
        logger.info("Learning embeddings from scratch")
        embedding = layers.Embedding(
            input_dim=config.vocab_size,
            output_dim=config.embedding_dim,
            embeddings_regularizer=regularizers.l2(config.l2_reg),
            trainable=True,
            mask_zero=True,
            name='embedding'
        )(input_text)

    # Parallel convolutional layers with different filter sizes
    conv_blocks = []
    for filter_size in config.filter_sizes:
        conv = layers.Conv1D(
            filters=config.num_filters,
            kernel_size=filter_size,
            activation='relu',
            padding='valid',
            kernel_regularizer=regularizers.l2(config.l2_reg),
            name=f'conv_{filter_size}gram'
        )(embedding)

        # Global max pooling
        pool = layers.GlobalMaxPooling1D(name=f'pool_{filter_size}gram')(conv)
        conv_blocks.append(pool)

    # Concatenate all pooled features
    concatenated = layers.Concatenate(name='concatenate')(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

    # Dropout for regularization
    dropout = layers.Dropout(config.dropout_rate, name='dropout')(concatenated)

    # Dense layer
    dense = layers.Dense(
        128,
        activation='relu',
        kernel_regularizer=regularizers.l2(config.l2_reg),
        name='dense'
    )(dropout)

    # Dropout again
    dropout2 = layers.Dropout(config.dropout_rate / 2, name='dropout2')(dense)

    # Output layer
    output = layers.Dense(
        config.num_classes,
        activation='softmax',
        name='output'
    )(dropout2)

    # Create model
    model = models.Model(inputs=input_text, outputs=output, name='TextCNN_TF')

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Print model summary
    logger.info(f"Model architecture:\n{model.summary()}")

    return model


def build_tensorflow_cnn_hybrid(config: TextCNNConfig) -> keras.Model:
    """
    Build hybrid CNN model with pre-trained embeddings (Word2Vec/GloVe).

    This is optimized for using frozen pre-trained embeddings.

    Args:
        config: TextCNNConfig object (must include embedding_matrix)

    Returns:
        Compiled Keras model
    """
    if config.embedding_matrix is None:
        raise ValueError("embedding_matrix required for hybrid model")

    # Use the same architecture but with frozen embeddings
    config.embedding_trainable = False
    return build_tensorflow_cnn(config)


# ============================================================================
# PYTORCH MODELS
# ============================================================================

class SentimentDataset(Dataset):
    """PyTorch Dataset for sentiment analysis."""

    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        """
        Args:
            sequences: Array of tokenized sequences (N, max_length)
            labels: Array of labels (N,)
        """
        self.sequences = torch.LongTensor(sequences)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class TextCNNPyTorch(nn.Module):
    """
    CNN model for text classification using PyTorch.

    Architecture matches TensorFlow version for fair comparison.
    """

    def __init__(self, config: TextCNNConfig):
        """
        Args:
            config: TextCNNConfig object
        """
        super(TextCNNPyTorch, self).__init__()

        self.config = config

        # Embedding layer
        if config.embedding_matrix is not None:
            # Use pre-trained embeddings
            self.embedding = nn.Embedding(
                num_embeddings=config.vocab_size,
                embedding_dim=config.embedding_dim,
                padding_idx=0
            )
            # Load pre-trained weights
            self.embedding.weight = nn.Parameter(
                torch.FloatTensor(config.embedding_matrix)
            )
            self.embedding.weight.requires_grad = config.embedding_trainable
        else:
            # Learn embeddings from scratch
            self.embedding = nn.Embedding(
                num_embeddings=config.vocab_size,
                embedding_dim=config.embedding_dim,
                padding_idx=0
            )

        # Convolutional layers with different kernel sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=config.embedding_dim,
                out_channels=config.num_filters,
                kernel_size=k,
                padding=0
            )
            for k in config.filter_sizes
        ])

        # Dropout
        self.dropout = nn.Dropout(config.dropout_rate)

        # Fully connected layers
        total_filters = config.num_filters * len(config.filter_sizes)
        self.fc1 = nn.Linear(total_filters, 128)
        self.fc2 = nn.Linear(128, config.num_classes)

        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(total_filters)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, max_seq_length)

        Returns:
            logits: Output tensor of shape (batch_size, num_classes)
        """
        # Embedding: (batch, seq_len) -> (batch, seq_len, embed_dim)
        embedded = self.embedding(x)

        # Transpose for Conv1d: (batch, seq_len, embed_dim) -> (batch, embed_dim, seq_len)
        embedded = embedded.transpose(1, 2)

        # Apply convolution + ReLU + max pooling for each filter size
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded))  # (batch, num_filters, seq_len - k + 1)
            pooled = F.max_pool1d(conv_out, conv_out.size(2))  # (batch, num_filters, 1)
            conv_outputs.append(pooled.squeeze(2))  # (batch, num_filters)

        # Concatenate all pooled features
        concatenated = torch.cat(conv_outputs, dim=1)  # (batch, total_filters)

        # Batch normalization
        normalized = self.batch_norm(concatenated)

        # Dropout
        dropped = self.dropout(normalized)

        # Fully connected layers
        fc1_out = F.relu(self.fc1(dropped))
        fc1_out = self.dropout(fc1_out)
        logits = self.fc2(fc1_out)

        return logits

    def predict_proba(self, x):
        """Get probability predictions."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


class BiLSTMPyTorch(nn.Module):
    """
    Bidirectional LSTM model for text classification.

    Alternative to CNN for capturing sequential dependencies.
    """

    def __init__(self, config: TextCNNConfig, hidden_size: int = 128, num_layers: int = 2):
        """
        Args:
            config: TextCNNConfig object
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
        """
        super(BiLSTMPyTorch, self).__init__()

        self.config = config
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Embedding layer
        if config.embedding_matrix is not None:
            self.embedding = nn.Embedding(
                num_embeddings=config.vocab_size,
                embedding_dim=config.embedding_dim,
                padding_idx=0
            )
            self.embedding.weight = nn.Parameter(
                torch.FloatTensor(config.embedding_matrix)
            )
            self.embedding.weight.requires_grad = config.embedding_trainable
        else:
            self.embedding = nn.Embedding(
                num_embeddings=config.vocab_size,
                embedding_dim=config.embedding_dim,
                padding_idx=0
            )

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=config.dropout_rate if num_layers > 1 else 0,
            bidirectional=True
        )

        # Dropout
        self.dropout = nn.Dropout(config.dropout_rate)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size * 2, config.num_classes)  # *2 for bidirectional

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, max_seq_length)

        Returns:
            logits: Output tensor of shape (batch_size, num_classes)
        """
        # Embedding
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)

        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)

        # Use the last hidden states from both directions
        # hidden shape: (num_layers * 2, batch, hidden_size)
        forward_hidden = hidden[-2, :, :]  # Last layer, forward direction
        backward_hidden = hidden[-1, :, :]  # Last layer, backward direction

        # Concatenate forward and backward
        combined = torch.cat([forward_hidden, backward_hidden], dim=1)

        # Dropout
        dropped = self.dropout(combined)

        # Fully connected layer
        logits = self.fc(dropped)

        return logits

    def predict_proba(self, x):
        """Get probability predictions."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_device() -> torch.device:
    """
    Get the best available device (CUDA > MPS > CPU).

    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")

    return device


def count_parameters(model) -> int:
    """
    Count trainable parameters in a model.

    Args:
        model: TensorFlow or PyTorch model

    Returns:
        Number of trainable parameters
    """
    if isinstance(model, keras.Model):
        # TensorFlow model
        return model.count_params()
    elif isinstance(model, nn.Module):
        # PyTorch model
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        raise ValueError("Unsupported model type")


def save_tensorflow_model(model: keras.Model, filepath: str) -> None:
    """
    Save TensorFlow model.

    Args:
        model: Keras model
        filepath: Path to save (without extension)
    """
    # Save in Keras format
    model.save(f"{filepath}.keras")
    logger.info(f"TensorFlow model saved to {filepath}.keras")


def load_tensorflow_model(filepath: str) -> keras.Model:
    """
    Load TensorFlow model.

    Args:
        filepath: Path to model file

    Returns:
        Loaded Keras model
    """
    model = keras.models.load_model(filepath)
    logger.info(f"TensorFlow model loaded from {filepath}")
    return model


def save_pytorch_model(model: nn.Module, filepath: str, config: TextCNNConfig) -> None:
    """
    Save PyTorch model.

    Args:
        model: PyTorch model
        filepath: Path to save (without extension)
        config: Model configuration
    """
    # Save state dict and config
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.__dict__
    }, f"{filepath}.pt")

    logger.info(f"PyTorch model saved to {filepath}.pt")


def load_pytorch_model(filepath: str, model_class: nn.Module = TextCNNPyTorch) -> Tuple[nn.Module, TextCNNConfig]:
    """
    Load PyTorch model.

    Args:
        filepath: Path to model file
        model_class: Model class to instantiate

    Returns:
        Tuple of (model, config)
    """
    checkpoint = torch.load(filepath, map_location='cpu')

    # Reconstruct config
    config = TextCNNConfig(**checkpoint['config'])

    # Instantiate model
    model = model_class(config)
    model.load_state_dict(checkpoint['model_state_dict'])

    logger.info(f"PyTorch model loaded from {filepath}")

    return model, config


# ============================================================================
# MODEL COMPARISON
# ============================================================================

def compare_models():
    """Print comparison of TensorFlow and PyTorch implementations."""
    print("\n=== Model Architecture Comparison ===\n")

    # Sample configuration
    config = TextCNNConfig(
        vocab_size=10000,
        embedding_dim=100,
        num_classes=3,
        max_seq_length=200
    )

    # Build TensorFlow model
    print("TensorFlow CNN:")
    tf_model = build_tensorflow_cnn(config)
    tf_params = count_parameters(tf_model)
    print(f"Trainable parameters: {tf_params:,}\n")

    # Build PyTorch model
    print("PyTorch CNN:")
    pt_model = TextCNNPyTorch(config)
    pt_params = count_parameters(pt_model)
    print(f"Trainable parameters: {pt_params:,}\n")

    print(f"Parameter difference: {abs(tf_params - pt_params):,}")


if __name__ == "__main__":
    # Demo: Compare model architectures
    compare_models()

    # Demo: Test PyTorch model forward pass
    print("\n=== Testing PyTorch Model ===")
    config = TextCNNConfig(vocab_size=1000, embedding_dim=100)
    model = TextCNNPyTorch(config)

    # Create dummy input
    batch_size = 4
    seq_length = 200
    dummy_input = torch.randint(0, 1000, (batch_size, seq_length))

    # Forward pass
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output (logits):\n{output}")

    # Probabilities
    probs = model.predict_proba(dummy_input)
    print(f"\nProbabilities:\n{probs}")

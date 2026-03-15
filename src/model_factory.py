"""
Model Factory

Unified interface for creating, loading, and managing all model types:
- Classical ML (scikit-learn)
- Deep Learning (TensorFlow/Keras)
- Deep Learning (PyTorch)
- Transformers (HuggingFace)
"""

import os
import logging
from pathlib import Path
from typing import Union, Tuple, Optional, Dict, Any
import joblib
import numpy as np

# TensorFlow imports
try:
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    keras = None

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    from src.dl_models import TextCNNPyTorch, BiLSTMPyTorch, load_pytorch_model, get_device
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    torch = None
    nn = None

# Transformers imports
try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# MODEL TYPES AND METADATA
# ============================================================================

class ModelType:
    """Enum for model types."""
    SKLEARN = 'sklearn'
    TENSORFLOW = 'tensorflow'
    PYTORCH = 'pytorch'
    TRANSFORMER = 'transformer'


class ModelMetadata:
    """Metadata for a model."""

    def __init__(
        self,
        model_type: str,
        framework: str,
        architecture: str,
        embedding_type: Optional[str] = None,
        vocab_size: Optional[int] = None,
        num_classes: int = 3
    ):
        self.model_type = model_type
        self.framework = framework
        self.architecture = architecture
        self.embedding_type = embedding_type
        self.vocab_size = vocab_size
        self.num_classes = num_classes


# ============================================================================
# MODEL FACTORY
# ============================================================================

class ModelFactory:
    """
    Factory for creating and loading models of different types.
    """

    # Model registry
    AVAILABLE_MODELS = {
        'logistic_regression': {
            'type': ModelType.SKLEARN,
            'description': 'Logistic Regression with TF-IDF',
            'framework': 'scikit-learn'
        },
        'naive_bayes': {
            'type': ModelType.SKLEARN,
            'description': 'Multinomial Naive Bayes with TF-IDF',
            'framework': 'scikit-learn'
        },
        'svm': {
            'type': ModelType.SKLEARN,
            'description': 'Linear SVM with TF-IDF',
            'framework': 'scikit-learn'
        },
        'random_forest': {
            'type': ModelType.SKLEARN,
            'description': 'Random Forest with TF-IDF',
            'framework': 'scikit-learn'
        },
        'cnn_tensorflow': {
            'type': ModelType.TENSORFLOW,
            'description': 'CNN for text classification (TensorFlow)',
            'framework': 'tensorflow'
        },
        'cnn_tensorflow_pretrained': {
            'type': ModelType.TENSORFLOW,
            'description': 'CNN with pre-trained embeddings (TensorFlow)',
            'framework': 'tensorflow'
        },
        'cnn_pytorch': {
            'type': ModelType.PYTORCH,
            'description': 'CNN for text classification (PyTorch)',
            'framework': 'pytorch'
        },
        'cnn_pytorch_pretrained': {
            'type': ModelType.PYTORCH,
            'description': 'CNN with pre-trained embeddings (PyTorch)',
            'framework': 'pytorch'
        },
        'lstm_pytorch': {
            'type': ModelType.PYTORCH,
            'description': 'BiLSTM for text classification (PyTorch)',
            'framework': 'pytorch'
        },
        'distilbert': {
            'type': ModelType.TRANSFORMER,
            'description': 'DistilBERT transformer model',
            'framework': 'transformers'
        }
    }

    def __init__(self, models_dir: str = 'models'):
        """
        Initialize Model Factory.

        Args:
            models_dir: Base directory for saved models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True, parents=True)

        # Create subdirectories
        (self.models_dir / 'dl').mkdir(exist_ok=True)
        (self.models_dir / 'sklearn').mkdir(exist_ok=True)
        (self.models_dir / 'transformer').mkdir(exist_ok=True)

        logger.info(f"ModelFactory initialized. Models directory: {self.models_dir}")

    def list_available_models(self) -> Dict[str, dict]:
        """List all available model types."""
        return self.AVAILABLE_MODELS

    def get_model_path(self, model_name: str) -> Path:
        """
        Get the file path for a model.

        Args:
            model_name: Name of the model

        Returns:
            Path to model file
        """
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {model_name}")

        model_info = self.AVAILABLE_MODELS[model_name]
        model_type = model_info['type']

        if model_type == ModelType.SKLEARN:
            return self.models_dir / f"{model_name}.joblib"
        elif model_type == ModelType.TENSORFLOW:
            return self.models_dir / 'dl' / f"{model_name}.keras"
        elif model_type == ModelType.PYTORCH:
            return self.models_dir / 'dl' / f"{model_name}.pt"
        elif model_type == ModelType.TRANSFORMER:
            return self.models_dir / 'transformer' / model_name
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def model_exists(self, model_name: str) -> bool:
        """Check if a model file exists."""
        try:
            path = self.get_model_path(model_name)
            return path.exists()
        except ValueError:
            return False

    def load_model(
        self,
        model_name: str,
        device: Optional[str] = None
    ) -> Tuple[Any, ModelMetadata]:
        """
        Load a model from disk.

        Args:
            model_name: Name of the model
            device: Device for PyTorch models ('cuda', 'mps', 'cpu')

        Returns:
            Tuple of (model, metadata)
        """
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {model_name}")

        model_info = self.AVAILABLE_MODELS[model_name]
        model_type = model_info['type']
        model_path = self.get_model_path(model_name)

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info(f"Loading {model_name} from {model_path}")

        # Load based on model type
        if model_type == ModelType.SKLEARN:
            model = joblib.load(model_path)
            metadata = ModelMetadata(
                model_type=ModelType.SKLEARN,
                framework='scikit-learn',
                architecture=model_name
            )

        elif model_type == ModelType.TENSORFLOW:
            if not TF_AVAILABLE:
                raise ImportError("TensorFlow not available")

            model = keras.models.load_model(model_path)
            metadata = ModelMetadata(
                model_type=ModelType.TENSORFLOW,
                framework='tensorflow',
                architecture='cnn',
                embedding_type='pretrained' if 'pretrained' in model_name else 'learned'
            )

        elif model_type == ModelType.PYTORCH:
            if not PYTORCH_AVAILABLE:
                raise ImportError("PyTorch not available")

            # Determine model class
            if 'lstm' in model_name.lower():
                model_class = BiLSTMPyTorch
            else:
                model_class = TextCNNPyTorch

            model, config = load_pytorch_model(str(model_path), model_class)

            # Move to device
            if device is None:
                device = get_device()
            else:
                device = torch.device(device)

            model = model.to(device)
            model.eval()

            metadata = ModelMetadata(
                model_type=ModelType.PYTORCH,
                framework='pytorch',
                architecture=model_class.__name__,
                embedding_type='pretrained' if 'pretrained' in model_name else 'learned',
                vocab_size=config.vocab_size,
                num_classes=config.num_classes
            )

        elif model_type == ModelType.TRANSFORMER:
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("Transformers library not available")

            model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))

            # Return both model and tokenizer
            model = (model, tokenizer)

            metadata = ModelMetadata(
                model_type=ModelType.TRANSFORMER,
                framework='transformers',
                architecture='distilbert'
            )

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        logger.info(f"Successfully loaded {model_name}")

        return model, metadata

    def predict(
        self,
        model: Any,
        metadata: ModelMetadata,
        texts: Union[str, list],
        preprocessor: Optional[callable] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with any model type.

        Args:
            model: Loaded model
            metadata: Model metadata
            texts: Single text or list of texts
            preprocessor: Optional preprocessing function

        Returns:
            Tuple of (predictions, probabilities)
        """
        # Convert single text to list
        if isinstance(texts, str):
            texts = [texts]

        # Preprocess if function provided
        if preprocessor is not None:
            texts = [preprocessor(text) for text in texts]

        # Predict based on model type
        if metadata.model_type == ModelType.SKLEARN:
            predictions = model.predict(texts)
            probabilities = model.predict_proba(texts) if hasattr(model, 'predict_proba') else None

        elif metadata.model_type == ModelType.TENSORFLOW:
            # Convert texts to sequences (requires tokenizer/embedding manager)
            # This is simplified - in practice, you'd need the original preprocessor
            logger.warning("TensorFlow prediction requires proper text preprocessing")
            raise NotImplementedError("Use dl_trainer.py for TensorFlow predictions")

        elif metadata.model_type == ModelType.PYTORCH:
            # Convert texts to sequences (requires tokenizer/embedding manager)
            logger.warning("PyTorch prediction requires proper text preprocessing")
            raise NotImplementedError("Use dl_trainer.py for PyTorch predictions")

        elif metadata.model_type == ModelType.TRANSFORMER:
            model_obj, tokenizer = model
            inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

            with torch.no_grad():
                outputs = model_obj(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1).numpy()
                predictions = np.argmax(probabilities, axis=1)

        else:
            raise ValueError(f"Unsupported model type: {metadata.model_type}")

        return predictions, probabilities

    def get_model_info(self, model_name: str) -> dict:
        """Get information about a model."""
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {model_name}")

        info = self.AVAILABLE_MODELS[model_name].copy()
        info['exists'] = self.model_exists(model_name)
        info['path'] = str(self.get_model_path(model_name))

        return info

    def list_trained_models(self) -> list:
        """List all trained models that exist on disk."""
        trained = []

        for model_name in self.AVAILABLE_MODELS.keys():
            if self.model_exists(model_name):
                info = self.get_model_info(model_name)
                trained.append({
                    'name': model_name,
                    'type': info['type'],
                    'framework': info['framework'],
                    'description': info['description'],
                    'path': info['path']
                })

        return trained


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_model_factory(models_dir: str = 'models') -> ModelFactory:
    """Get a ModelFactory instance."""
    return ModelFactory(models_dir)


def quick_load(model_name: str, models_dir: str = 'models') -> Tuple[Any, ModelMetadata]:
    """Quick load a model."""
    factory = ModelFactory(models_dir)
    return factory.load_model(model_name)


def list_all_models(models_dir: str = 'models') -> None:
    """Print all available models."""
    factory = ModelFactory(models_dir)

    print("\n=== Available Models ===\n")

    for model_name, info in factory.list_available_models().items():
        exists = factory.model_exists(model_name)
        status = "✓ TRAINED" if exists else "✗ NOT TRAINED"

        print(f"{model_name}")
        print(f"  Framework: {info['framework']}")
        print(f"  Description: {info['description']}")
        print(f"  Status: {status}")
        print()


if __name__ == "__main__":
    # Demo usage
    print("Model Factory - Unified Model Management\n")

    factory = ModelFactory()

    print("=== Available Model Types ===")
    for name, info in factory.list_available_models().items():
        print(f"- {name}: {info['description']}")

    print("\n=== Trained Models ===")
    trained = factory.list_trained_models()

    if trained:
        for model in trained:
            print(f"- {model['name']} ({model['framework']})")
    else:
        print("No trained models found. Train some models first!")

    print("\n=== Model Paths ===")
    print("sklearn models:", factory.models_dir)
    print("DL models:", factory.models_dir / 'dl')
    print("Transformer models:", factory.models_dir / 'transformer')

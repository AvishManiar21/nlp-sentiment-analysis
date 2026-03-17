#!/usr/bin/env python3
"""
Upload trained deep learning models to HuggingFace Hub.

This script uploads all trained models from models/dl/ to a HuggingFace Hub repository
for deployment on Streamlit Cloud or other platforms.

Usage:
    python scripts/upload_models_to_hub.py <repo_id>

Example:
    python scripts/upload_models_to_hub.py username/nlp-sentiment-models

Requirements:
    - huggingface_hub library installed (pip install huggingface-hub)
    - HuggingFace account and API token
    - Logged in via: huggingface-cli login
"""

import sys
import argparse
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model files to upload
MODEL_FILES = [
    "cnn_tensorflow.keras",
    "cnn_tensorflow_pretrained.keras",
    "cnn_pytorch.pt",
    "cnn_pytorch_pretrained.pt",
    "lstm_pytorch_pretrained.pt",
]

MODELS_DIR = Path("models/dl")


def check_prerequisites():
    """Check if prerequisites are met."""
    # Check if models directory exists
    if not MODELS_DIR.exists():
        logger.error(f"Models directory not found: {MODELS_DIR}")
        logger.error("Train models first with: python main.py --train-dl")
        return False

    # Check if any model files exist
    existing_models = []
    for model_file in MODEL_FILES:
        if (MODELS_DIR / model_file).exists():
            existing_models.append(model_file)

    if not existing_models:
        logger.error("No trained models found in models/dl/")
        logger.error("Train models first with: python main.py --train-dl")
        return False

    logger.info(f"Found {len(existing_models)} model(s) to upload:")
    for model in existing_models:
        size_mb = (MODELS_DIR / model).stat().st_size / (1024 * 1024)
        logger.info(f"  - {model} ({size_mb:.1f} MB)")

    # Check if huggingface_hub is installed
    try:
        import huggingface_hub
        logger.info(f"✓ huggingface_hub version: {huggingface_hub.__version__}")
    except ImportError:
        logger.error("huggingface_hub not installed")
        logger.error("Install with: pip install huggingface-hub")
        return False

    return True


def upload_models(repo_id: str, create_repo: bool = True, private: bool = False):
    """
    Upload models to HuggingFace Hub.

    Args:
        repo_id: Repository ID (e.g., "username/nlp-sentiment-models")
        create_repo: Whether to create the repository if it doesn't exist
        private: Whether to make the repository private
    """
    try:
        from huggingface_hub import HfApi, create_repo as hf_create_repo
    except ImportError:
        logger.error("huggingface_hub not installed")
        return False

    logger.info(f"Uploading models to HuggingFace Hub: {repo_id}")

    # Initialize HuggingFace API
    api = HfApi()

    # Create repository if needed
    if create_repo:
        try:
            logger.info(f"Creating repository: {repo_id} (private={private})")
            hf_create_repo(
                repo_id=repo_id,
                repo_type="model",
                private=private,
                exist_ok=True
            )
            logger.info("✓ Repository created/verified")
        except Exception as e:
            logger.error(f"Failed to create repository: {e}")
            logger.error("Make sure you're logged in: huggingface-cli login")
            return False

    # Upload each model file
    existing_models = []
    for model_file in MODEL_FILES:
        model_path = MODELS_DIR / model_file
        if model_path.exists():
            existing_models.append((model_file, model_path))

    total_files = len(existing_models)
    success_count = 0

    for idx, (model_file, model_path) in enumerate(existing_models, 1):
        try:
            logger.info(f"Uploading {model_file} ({idx}/{total_files})...")

            api.upload_file(
                path_or_fileobj=str(model_path),
                path_in_repo=model_file,
                repo_id=repo_id,
                repo_type="model",
            )

            logger.info(f"✓ Uploaded {model_file}")
            success_count += 1

        except Exception as e:
            logger.error(f"✗ Failed to upload {model_file}: {e}")

    # Summary
    logger.info("=" * 60)
    if success_count == total_files:
        logger.info(f"✓ Successfully uploaded all {total_files} models!")
    elif success_count > 0:
        logger.warning(
            f"Uploaded {success_count}/{total_files} models. "
            f"Some uploads failed."
        )
    else:
        logger.error("Failed to upload any models")
        return False

    # Create README if repository was just created
    try:
        logger.info("Creating README.md...")
        readme_content = f"""---
license: mit
tags:
- sentiment-analysis
- nlp
- pytorch
- tensorflow
- amazon-reviews
---

# NLP Sentiment Analysis Models

Pre-trained deep learning models for sentiment analysis on Amazon product reviews.

## Models Included

- **CNN (TensorFlow)**: Convolutional Neural Network with learned embeddings
- **CNN + GloVe (TensorFlow)**: CNN with pre-trained GloVe embeddings
- **CNN (PyTorch)**: Convolutional Neural Network with learned embeddings
- **CNN + GloVe (PyTorch)**: CNN with pre-trained GloVe embeddings
- **BiLSTM + GloVe (PyTorch)**: Bidirectional LSTM with pre-trained GloVe embeddings

## Dataset

Trained on Amazon Reviews 2023 dataset with ~47K product reviews across multiple categories.

## Performance

- CNN models: ~85-87% accuracy
- BiLSTM models: ~86-88% accuracy

## Usage

These models are automatically downloaded by the Streamlit dashboard when deployed on Streamlit Cloud.

### Manual Download

```python
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="{repo_id}",
    filename="cnn_pytorch_pretrained.pt"
)
```

## Repository

GitHub: https://github.com/YOUR_USERNAME/nlp-sentiment-analysis
"""

        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
        )
        logger.info("✓ Created README.md")

    except Exception as e:
        logger.warning(f"Could not create README: {e}")

    logger.info("=" * 60)
    logger.info(f"🎉 Models uploaded successfully!")
    logger.info(f"Repository URL: https://huggingface.co/{repo_id}")
    logger.info("")
    logger.info("Next steps:")
    logger.info(f"1. Set environment variable on Streamlit Cloud:")
    logger.info(f"   HF_MODEL_REPO={repo_id}")
    logger.info("2. Deploy your app to Streamlit Cloud")
    logger.info("3. Models will be automatically downloaded on first run")

    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Upload deep learning models to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload to public repository
  python scripts/upload_models_to_hub.py username/nlp-sentiment-models

  # Upload to private repository
  python scripts/upload_models_to_hub.py username/nlp-sentiment-models --private

  # Don't create repository (must exist)
  python scripts/upload_models_to_hub.py username/nlp-sentiment-models --no-create

Before running:
  1. Install: pip install huggingface-hub
  2. Login: huggingface-cli login
  3. Train models: python main.py --train-dl
        """
    )

    parser.add_argument(
        "repo_id",
        help="HuggingFace repository ID (e.g., username/nlp-sentiment-models)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private"
    )
    parser.add_argument(
        "--no-create",
        action="store_true",
        help="Don't create repository (must already exist)"
    )

    args = parser.parse_args()

    # Check prerequisites
    logger.info("Checking prerequisites...")
    if not check_prerequisites():
        sys.exit(1)

    # Upload models
    success = upload_models(
        repo_id=args.repo_id,
        create_repo=not args.no_create,
        private=args.private
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

"""
Model storage utilities for downloading and managing deep learning models.

Supports both local development (models stored locally) and cloud deployment
(models downloaded from HuggingFace Hub).
"""

import os
from pathlib import Path
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

# Model directory
MODELS_DIR = Path("models/dl")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# HuggingFace Hub configuration
DEFAULT_HF_REPO = os.getenv("HF_MODEL_REPO", "")  # User sets this in Streamlit Cloud
HF_CACHE_DIR = Path(".hf_models_cache")

# Model files to download
MODEL_FILES = [
    "cnn_tensorflow.keras",
    "cnn_tensorflow_pretrained.keras",
    "cnn_pytorch.pt",
    "cnn_pytorch_pretrained.pt",
    "lstm_pytorch_pretrained.pt",
]


def check_models_exist_locally() -> bool:
    """Check if any deep learning models exist in the local models directory."""
    if not MODELS_DIR.exists():
        return False

    # Check if at least one model file exists
    for model_file in MODEL_FILES:
        if (MODELS_DIR / model_file).exists():
            return True

    return False


def get_missing_models() -> List[str]:
    """Get list of model files that don't exist locally."""
    missing = []
    for model_file in MODEL_FILES:
        if not (MODELS_DIR / model_file).exists():
            missing.append(model_file)
    return missing


def download_models_from_hf(
    repo_id: Optional[str] = None,
    progress_callback: Optional[callable] = None
) -> bool:
    """
    Download deep learning models from HuggingFace Hub.

    Args:
        repo_id: HuggingFace repository ID (e.g., "username/nlp-sentiment-models")
                 If None, uses HF_MODEL_REPO environment variable
        progress_callback: Optional callback function for progress updates
                          Called with (current_file, total_files, filename)

    Returns:
        True if successful, False otherwise
    """
    # Check if HuggingFace Hub is available
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.error(
            "huggingface_hub not installed. "
            "Install with: pip install huggingface-hub"
        )
        return False

    # Get repository ID
    repo_id = repo_id or DEFAULT_HF_REPO
    if not repo_id:
        logger.error(
            "No HuggingFace repository specified. "
            "Set HF_MODEL_REPO environment variable or pass repo_id parameter."
        )
        return False

    logger.info(f"Downloading models from HuggingFace Hub: {repo_id}")

    # Get list of missing models
    missing_models = get_missing_models()
    if not missing_models:
        logger.info("All models already exist locally. Skipping download.")
        return True

    logger.info(f"Missing models: {missing_models}")

    # Download each model
    total_files = len(missing_models)
    success_count = 0

    for idx, model_file in enumerate(missing_models, 1):
        try:
            if progress_callback:
                progress_callback(idx, total_files, model_file)

            logger.info(f"Downloading {model_file} ({idx}/{total_files})...")

            # Download from HuggingFace Hub
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=model_file,
                cache_dir=str(HF_CACHE_DIR),
                resume_download=True
            )

            # Copy to models directory
            import shutil
            dest_path = MODELS_DIR / model_file
            shutil.copy2(downloaded_path, dest_path)

            logger.info(f"✓ Downloaded {model_file}")
            success_count += 1

        except Exception as e:
            logger.error(f"✗ Failed to download {model_file}: {e}")
            # Continue with other models

    if success_count == 0:
        logger.error("Failed to download any models")
        return False

    if success_count < total_files:
        logger.warning(
            f"Only downloaded {success_count}/{total_files} models. "
            "Some models may be unavailable."
        )
    else:
        logger.info(f"✓ Successfully downloaded all {total_files} models")

    return True


def ensure_models_available(
    repo_id: Optional[str] = None,
    progress_callback: Optional[callable] = None
) -> bool:
    """
    Ensure deep learning models are available.

    If models exist locally, does nothing.
    If models don't exist, attempts to download from HuggingFace Hub.

    Args:
        repo_id: HuggingFace repository ID
        progress_callback: Optional callback for progress updates

    Returns:
        True if models are available, False otherwise
    """
    # Check if models exist locally
    if check_models_exist_locally():
        logger.info("Models found locally. Skipping download.")
        return True

    # Models don't exist, try to download
    logger.info("No local models found. Attempting to download from HuggingFace Hub...")

    # Skip download if no repo configured (local development mode)
    repo_id = repo_id or DEFAULT_HF_REPO
    if not repo_id:
        logger.warning(
            "No HuggingFace repository configured. "
            "Running in local development mode. "
            "Train models locally with: python main.py --train-dl"
        )
        return False

    # Attempt download
    success = download_models_from_hf(repo_id, progress_callback)

    if success:
        logger.info("Models successfully downloaded from HuggingFace Hub")
    else:
        logger.error(
            "Failed to download models from HuggingFace Hub. "
            "Please check your configuration and try again."
        )

    return success


def get_model_info() -> dict:
    """Get information about model availability and storage."""
    models_exist = check_models_exist_locally()
    missing_models = get_missing_models()
    hf_repo = DEFAULT_HF_REPO

    return {
        "models_exist_locally": models_exist,
        "total_models": len(MODEL_FILES),
        "available_models": len(MODEL_FILES) - len(missing_models),
        "missing_models": missing_models,
        "hf_repo_configured": bool(hf_repo),
        "hf_repo": hf_repo or "Not configured",
        "models_dir": str(MODELS_DIR.absolute()),
    }

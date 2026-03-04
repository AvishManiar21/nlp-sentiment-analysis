"""Shared test fixtures for NLP Sentiment Analysis tests."""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_reviews():
    """Sample DataFrame with review data for testing."""
    return pd.DataFrame({
        "review_id": ["AMZ-001", "AMZ-002", "AMZ-003", "AMZ-004", "AMZ-005"],
        "review_text": [
            "This product is absolutely amazing! Best purchase ever.",
            "Terrible quality. Broke after one day. Complete waste of money.",
            "It's okay, nothing special. Does the job.",
            "Love it! Exceeded my expectations. Highly recommend!",
            "Disappointed. Not worth the price at all."
        ],
        "rating": [5, 1, 3, 5, 2],
        "category": ["Electronics", "Home", "Books", "Electronics", "Clothing"],
        "ground_truth": ["positive", "negative", "neutral", "positive", "negative"]
    })


@pytest.fixture
def sample_texts():
    """Sample text list for sentiment analysis testing."""
    return [
        "I love this product! Amazing quality and fast shipping.",
        "Worst purchase ever. Complete waste of money. Never buying again.",
        "It's decent, nothing special but gets the job done."
    ]


@pytest.fixture
def positive_text():
    """Clearly positive text for testing."""
    return "This is absolutely wonderful! I love it so much. Best product ever!"


@pytest.fixture
def negative_text():
    """Clearly negative text for testing."""
    return "This is terrible. Worst product ever. Complete garbage. Hate it."


@pytest.fixture
def neutral_text():
    """Neutral text for testing."""
    return "The product arrived. It exists. It is a thing."


@pytest.fixture
def html_text():
    """Text with HTML tags for testing HTML cleaning."""
    return "<p>This is a <strong>great</strong> product!</p>&amp; I love it &lt;3"


@pytest.fixture
def dirty_text():
    """Text with URLs, emails, and special characters for testing cleaning."""
    return "Check out https://example.com! Email me at test@email.com!!! OMG soooo goood"


@pytest.fixture
def sample_predictions():
    """Sample predictions and ground truth for metric testing."""
    return {
        "y_true": ["positive", "negative", "positive", "negative", "positive"],
        "y_pred": ["positive", "negative", "negative", "negative", "positive"]
    }


@pytest.fixture
def binary_labels():
    """Binary labels for ML testing."""
    return ["positive", "negative"]


@pytest.fixture
def sample_sentiment_df():
    """DataFrame with sentiment analysis results."""
    return pd.DataFrame({
        "review_text": [
            "Great product!",
            "Terrible quality.",
            "It was okay.",
            "Love it!",
            "Disappointed."
        ],
        "rating": [5, 1, 3, 5, 2],
        "ground_truth": ["positive", "negative", "neutral", "positive", "negative"],
        "vader_compound": [0.6, -0.5, 0.1, 0.7, -0.3],
        "textblob_polarity": [0.5, -0.4, 0.0, 0.6, -0.2],
        "ensemble_score": [0.55, -0.45, 0.05, 0.65, -0.25],
        "sentiment_label": ["positive", "negative", "neutral", "positive", "negative"],
        "category": ["Electronics", "Home", "Books", "Electronics", "Clothing"]
    })


@pytest.fixture
def model_results():
    """Sample model evaluation results for comparison testing."""
    return {
        "VADER": {
            "accuracy": 0.70,
            "f1_weighted": 0.68,
            "f1_macro": 0.65,
            "precision_weighted": 0.72,
            "recall_weighted": 0.70
        },
        "TextBlob": {
            "accuracy": 0.60,
            "f1_weighted": 0.58,
            "f1_macro": 0.55,
            "precision_weighted": 0.62,
            "recall_weighted": 0.60
        },
        "Logistic Regression": {
            "accuracy": 0.85,
            "f1_weighted": 0.84,
            "f1_macro": 0.83,
            "precision_weighted": 0.86,
            "recall_weighted": 0.85
        }
    }

"""Unit tests for src/sentiment_analyzer.py"""

import pytest
import pandas as pd
import numpy as np
from src.sentiment_analyzer import (
    sanitize_text,
    analyze_vader,
    analyze_textblob,
    classify_sentiment,
    sentiment_strength,
    compute_ensemble_score,
    predict_sentiment_vader,
    predict_sentiment_textblob,
)
from nltk.sentiment.vader import SentimentIntensityAnalyzer


@pytest.fixture
def vader_analyzer():
    """Create VADER analyzer for tests."""
    return SentimentIntensityAnalyzer()


class TestSanitizeText:
    """Tests for sanitize_text function."""

    def test_handles_none(self):
        assert sanitize_text(None) == ""

    def test_handles_nan(self):
        assert sanitize_text(float("nan")) == ""

    def test_handles_normal_text(self):
        result = sanitize_text("Hello World")
        assert result == "Hello World"

    def test_truncates_long_text(self):
        long_text = "a" * 20000
        result = sanitize_text(long_text)
        assert len(result) <= 10000

    def test_strips_whitespace(self):
        result = sanitize_text("  Hello  ")
        assert result == "Hello"

    def test_handles_unicode(self):
        result = sanitize_text("Hello 世界")
        assert isinstance(result, str)


class TestAnalyzeVader:
    """Tests for analyze_vader function."""

    def test_returns_four_values(self, vader_analyzer):
        result = analyze_vader("I love this!", vader_analyzer)
        assert len(result) == 4

    def test_compound_in_range(self, vader_analyzer, positive_text):
        compound, _, _, _ = analyze_vader(positive_text, vader_analyzer)
        assert -1.0 <= compound <= 1.0

    def test_positive_text_has_positive_compound(self, vader_analyzer, positive_text):
        compound, _, _, _ = analyze_vader(positive_text, vader_analyzer)
        assert compound > 0

    def test_negative_text_has_negative_compound(self, vader_analyzer, negative_text):
        compound, _, _, _ = analyze_vader(negative_text, vader_analyzer)
        assert compound < 0

    def test_empty_text_returns_neutral(self, vader_analyzer):
        compound, pos, neg, neu = analyze_vader("", vader_analyzer)
        assert compound == 0.0
        assert neu == 1.0


class TestAnalyzeTextblob:
    """Tests for analyze_textblob function."""

    def test_returns_two_values(self):
        result = analyze_textblob("I love this!")
        assert len(result) == 2

    def test_polarity_in_range(self, positive_text):
        polarity, _ = analyze_textblob(positive_text)
        assert -1.0 <= polarity <= 1.0

    def test_subjectivity_in_range(self, positive_text):
        _, subjectivity = analyze_textblob(positive_text)
        assert 0.0 <= subjectivity <= 1.0

    def test_positive_text_has_positive_polarity(self, positive_text):
        polarity, _ = analyze_textblob(positive_text)
        assert polarity > 0

    def test_negative_text_has_negative_polarity(self, negative_text):
        polarity, _ = analyze_textblob(negative_text)
        assert polarity < 0

    def test_empty_text_returns_neutral(self):
        polarity, subjectivity = analyze_textblob("")
        assert polarity == 0.0
        assert subjectivity == 0.5


class TestClassifySentiment:
    """Tests for classify_sentiment function."""

    def test_positive_above_threshold(self):
        assert classify_sentiment(0.1) == "positive"
        assert classify_sentiment(0.5) == "positive"
        assert classify_sentiment(1.0) == "positive"

    def test_negative_below_threshold(self):
        assert classify_sentiment(-0.1) == "negative"
        assert classify_sentiment(-0.5) == "negative"
        assert classify_sentiment(-1.0) == "negative"

    def test_neutral_at_threshold(self):
        assert classify_sentiment(0.0) == "neutral"
        assert classify_sentiment(0.04) == "neutral"
        assert classify_sentiment(-0.04) == "neutral"

    def test_boundary_positive(self):
        assert classify_sentiment(0.05) == "positive"

    def test_boundary_negative(self):
        assert classify_sentiment(-0.05) == "negative"


class TestSentimentStrength:
    """Tests for sentiment_strength function."""

    def test_strong_positive(self):
        assert sentiment_strength(0.8) == "strong"
        assert sentiment_strength(0.6) == "strong"

    def test_strong_negative(self):
        assert sentiment_strength(-0.8) == "strong"
        assert sentiment_strength(-0.6) == "strong"

    def test_moderate(self):
        assert sentiment_strength(0.4) == "moderate"
        assert sentiment_strength(-0.4) == "moderate"
        assert sentiment_strength(0.3) == "moderate"

    def test_weak(self):
        assert sentiment_strength(0.1) == "weak"
        assert sentiment_strength(-0.1) == "weak"
        assert sentiment_strength(0.0) == "weak"


class TestComputeEnsembleScore:
    """Tests for compute_ensemble_score function."""

    def test_returns_float(self):
        result = compute_ensemble_score(0.5, 0.5)
        assert isinstance(result, float)

    def test_weighted_combination(self):
        result = compute_ensemble_score(1.0, 0.0, vader_weight=0.65)
        assert result == pytest.approx(0.65, rel=0.01)

    def test_clamped_to_range(self):
        result = compute_ensemble_score(1.0, 1.0)
        assert -1.0 <= result <= 1.0

    def test_negative_clamped(self):
        result = compute_ensemble_score(-1.0, -1.0)
        assert result >= -1.0

    def test_equal_weights(self):
        result = compute_ensemble_score(0.8, 0.4, vader_weight=0.5)
        assert result == pytest.approx(0.6, rel=0.01)


class TestPredictSentimentVader:
    """Tests for predict_sentiment_vader function."""

    def test_returns_list(self, sample_texts):
        result = predict_sentiment_vader(sample_texts)
        assert isinstance(result, list)

    def test_returns_correct_length(self, sample_texts):
        result = predict_sentiment_vader(sample_texts)
        assert len(result) == len(sample_texts)

    def test_labels_are_valid(self, sample_texts):
        result = predict_sentiment_vader(sample_texts)
        valid_labels = {"positive", "negative", "neutral"}
        assert all(label in valid_labels for label in result)

    def test_positive_text_labeled_positive(self, positive_text):
        result = predict_sentiment_vader([positive_text])
        assert result[0] == "positive"

    def test_negative_text_labeled_negative(self, negative_text):
        result = predict_sentiment_vader([negative_text])
        assert result[0] == "negative"


class TestPredictSentimentTextblob:
    """Tests for predict_sentiment_textblob function."""

    def test_returns_list(self, sample_texts):
        result = predict_sentiment_textblob(sample_texts)
        assert isinstance(result, list)

    def test_returns_correct_length(self, sample_texts):
        result = predict_sentiment_textblob(sample_texts)
        assert len(result) == len(sample_texts)

    def test_labels_are_valid(self, sample_texts):
        result = predict_sentiment_textblob(sample_texts)
        valid_labels = {"positive", "negative", "neutral"}
        assert all(label in valid_labels for label in result)

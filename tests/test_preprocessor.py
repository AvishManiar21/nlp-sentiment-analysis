"""Unit tests for src/preprocessor.py"""

import pytest
import pandas as pd
from src.preprocessor import (
    clean_html,
    clean_text,
    normalize_text,
    tokenize,
    lemmatize,
    remove_stopwords,
    rating_to_sentiment,
    rating_to_sentiment_binary,
    preprocess_for_ml,
    create_ground_truth_labels,
)


class TestCleanHtml:
    """Tests for clean_html function."""

    def test_removes_html_tags(self, html_text):
        result = clean_html(html_text)
        assert "<p>" not in result
        assert "<strong>" not in result
        assert "</p>" not in result

    def test_decodes_html_entities(self, html_text):
        result = clean_html(html_text)
        assert "&amp;" not in result
        assert "&lt;" not in result
        assert "&" in result or "<" in result

    def test_handles_none(self):
        assert clean_html(None) == ""

    def test_handles_nan(self):
        assert clean_html(float("nan")) == ""

    def test_handles_empty_string(self):
        assert clean_html("") == ""


class TestCleanText:
    """Tests for clean_text function."""

    def test_removes_urls(self, dirty_text):
        result = clean_text(dirty_text)
        assert "https://example.com" not in result
        assert "http" not in result

    def test_removes_emails(self, dirty_text):
        result = clean_text(dirty_text)
        assert "test@email.com" not in result
        assert "@" not in result

    def test_reduces_repeated_chars(self, dirty_text):
        result = clean_text(dirty_text)
        assert "soooo" not in result
        assert "!!!" not in result

    def test_handles_none(self):
        assert clean_text(None) == ""

    def test_preserves_sentiment_words(self):
        text = "I love this! It's great!"
        result = clean_text(text)
        assert "love" in result.lower()
        assert "great" in result.lower()


class TestNormalizeText:
    """Tests for normalize_text function."""

    def test_lowercases_text(self):
        result = normalize_text("HELLO WORLD", lowercase=True)
        assert result == "hello world"

    def test_removes_punctuation(self):
        result = normalize_text("Hello, world!", remove_punctuation=True)
        assert "," not in result
        assert "!" not in result

    def test_preserves_case_when_disabled(self):
        result = normalize_text("HELLO", lowercase=False, remove_punctuation=False)
        assert "HELLO" in result

    def test_handles_empty_string(self):
        assert normalize_text("") == ""


class TestTokenize:
    """Tests for tokenize function."""

    def test_tokenizes_simple_sentence(self):
        tokens = tokenize("Hello world")
        assert "hello" in tokens
        assert "world" in tokens

    def test_returns_list(self):
        result = tokenize("Test sentence")
        assert isinstance(result, list)

    def test_handles_empty_string(self):
        assert tokenize("") == []

    def test_lowercases_tokens(self):
        tokens = tokenize("UPPERCASE TEXT")
        assert all(t.islower() or not t.isalpha() for t in tokens)


class TestLemmatize:
    """Tests for lemmatize function."""

    def test_lemmatizes_plural(self):
        result = lemmatize(["running", "cats", "better"])
        assert "running" in result or "run" in result
        assert "cat" in result
        assert "better" in result

    def test_returns_list(self):
        result = lemmatize(["word", "words"])
        assert isinstance(result, list)

    def test_handles_empty_list(self):
        assert lemmatize([]) == []


class TestRatingToSentiment:
    """Tests for rating_to_sentiment function."""

    def test_rating_1_is_negative(self):
        assert rating_to_sentiment(1) == "negative"

    def test_rating_2_is_negative(self):
        assert rating_to_sentiment(2) == "negative"

    def test_rating_3_is_neutral(self):
        assert rating_to_sentiment(3) == "neutral"

    def test_rating_4_is_positive(self):
        assert rating_to_sentiment(4) == "positive"

    def test_rating_5_is_positive(self):
        assert rating_to_sentiment(5) == "positive"


class TestRatingToSentimentBinary:
    """Tests for rating_to_sentiment_binary function."""

    def test_rating_1_is_negative(self):
        assert rating_to_sentiment_binary(1) == "negative"

    def test_rating_2_is_negative(self):
        assert rating_to_sentiment_binary(2) == "negative"

    def test_rating_3_is_negative(self):
        assert rating_to_sentiment_binary(3) == "negative"

    def test_rating_4_is_positive(self):
        assert rating_to_sentiment_binary(4) == "positive"

    def test_rating_5_is_positive(self):
        assert rating_to_sentiment_binary(5) == "positive"


class TestPreprocessForMl:
    """Tests for preprocess_for_ml function."""

    def test_returns_string(self):
        result = preprocess_for_ml("This is a test sentence.")
        assert isinstance(result, str)

    def test_removes_stopwords(self):
        result = preprocess_for_ml("This is the best product")
        assert "the" not in result.split()
        assert "is" not in result.split()

    def test_lowercases_output(self):
        result = preprocess_for_ml("UPPERCASE TEXT HERE")
        assert result == result.lower()

    def test_handles_empty_string(self):
        result = preprocess_for_ml("")
        assert result == ""

    def test_preserves_meaningful_words(self):
        result = preprocess_for_ml("I love this amazing product")
        assert "love" in result or "amazing" in result or "product" in result


class TestCreateGroundTruthLabels:
    """Tests for create_ground_truth_labels function."""

    def test_adds_ground_truth_column(self, sample_reviews):
        result = create_ground_truth_labels(sample_reviews)
        assert "ground_truth" in result.columns

    def test_maps_ratings_correctly(self):
        df = pd.DataFrame({"rating": [1, 2, 3, 4, 5]})
        result = create_ground_truth_labels(df)
        assert result["ground_truth"].tolist() == [
            "negative", "negative", "neutral", "positive", "positive"
        ]

    def test_excludes_neutral_when_disabled(self):
        df = pd.DataFrame({"rating": [1, 3, 5]})
        result = create_ground_truth_labels(df, include_neutral=False)
        assert "neutral" not in result["ground_truth"].values
        assert len(result) == 2

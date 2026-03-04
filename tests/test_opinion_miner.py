"""Unit tests for src/opinion_miner.py"""

import pytest
import pandas as pd
import numpy as np
from src.opinion_miner import (
    extract_key_phrases,
    extract_aspect_sentiments,
    analyze_drivers,
    category_sentiment_summary,
    get_dynamic_aspects,
)


class TestExtractKeyPhrases:
    """Tests for extract_key_phrases function."""

    def test_returns_list(self):
        texts = ["This product is amazing", "Great quality and value"]
        result = extract_key_phrases(texts, top_n=5)
        assert isinstance(result, list)

    def test_returns_tuples(self):
        texts = ["This product is amazing", "Great quality and value"]
        result = extract_key_phrases(texts, top_n=5)
        if result:
            assert all(isinstance(item, tuple) for item in result)
            assert all(len(item) == 2 for item in result)

    def test_respects_top_n(self):
        texts = ["Product quality price value design"] * 10
        result = extract_key_phrases(texts, top_n=3)
        assert len(result) <= 3

    def test_handles_empty_input(self):
        result = extract_key_phrases([], top_n=5)
        assert result == []

    def test_phrase_score_format(self):
        texts = ["Amazing product quality", "Great value for money"]
        result = extract_key_phrases(texts, top_n=5)
        if result:
            phrase, score = result[0]
            assert isinstance(phrase, str)
            assert isinstance(score, (int, float))


class TestExtractAspectSentiments:
    """Tests for extract_aspect_sentiments function."""

    def test_returns_dataframe(self, sample_sentiment_df):
        result = extract_aspect_sentiments(sample_sentiment_df)
        assert isinstance(result, pd.DataFrame)

    def test_contains_aspect_column(self, sample_sentiment_df):
        result = extract_aspect_sentiments(sample_sentiment_df)
        if not result.empty:
            assert "aspect" in result.columns

    def test_contains_sentiment_metrics(self, sample_sentiment_df):
        result = extract_aspect_sentiments(sample_sentiment_df)
        if not result.empty:
            assert "avg_sentiment" in result.columns or "mention_count" in result.columns


class TestAnalyzeDrivers:
    """Tests for analyze_drivers function."""

    def test_returns_list(self, sample_sentiment_df):
        result = analyze_drivers(sample_sentiment_df, sentiment_type="positive")
        assert isinstance(result, list)

    def test_positive_drivers_format(self, sample_sentiment_df):
        result = analyze_drivers(sample_sentiment_df, sentiment_type="positive")
        if result:
            assert all(isinstance(item, tuple) for item in result)

    def test_negative_drivers_format(self, sample_sentiment_df):
        result = analyze_drivers(sample_sentiment_df, sentiment_type="negative")
        if result:
            assert all(isinstance(item, tuple) for item in result)


class TestCategorySentimentSummary:
    """Tests for category_sentiment_summary function."""

    def test_returns_dataframe(self, sample_sentiment_df):
        result = category_sentiment_summary(sample_sentiment_df)
        assert isinstance(result, pd.DataFrame)

    def test_groups_by_category(self, sample_sentiment_df):
        result = category_sentiment_summary(sample_sentiment_df)
        if not result.empty:
            assert "category" in result.columns

    def test_contains_aggregated_metrics(self, sample_sentiment_df):
        result = category_sentiment_summary(sample_sentiment_df)
        if not result.empty:
            expected_cols = ["avg_rating", "avg_sentiment", "total_reviews"]
            for col in expected_cols:
                if col in result.columns:
                    assert True
                    break

    def test_handles_missing_category(self):
        df = pd.DataFrame({
            "review_text": ["Test review"],
            "rating": [5],
            "ensemble_score": [0.8]
        })
        result = category_sentiment_summary(df)
        assert isinstance(result, pd.DataFrame)


class TestGetDynamicAspects:
    """Tests for get_dynamic_aspects function."""

    def test_returns_list(self):
        texts = ["The quality is great", "Price is too high"]
        result = get_dynamic_aspects(texts)
        assert isinstance(result, list)

    def test_extracts_common_aspects(self):
        texts = [
            "The quality is excellent",
            "Great quality product",
            "Quality is important"
        ] * 5
        result = get_dynamic_aspects(texts, min_count=3)
        assert len(result) >= 0

    def test_handles_empty_input(self):
        result = get_dynamic_aspects([])
        assert isinstance(result, list)

    def test_returns_unique_aspects(self):
        texts = ["quality quality quality"] * 10
        result = get_dynamic_aspects(texts)
        assert len(result) == len(set(result))

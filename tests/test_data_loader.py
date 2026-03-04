"""Unit tests for src/data_loader.py"""

import pytest
import pandas as pd
import numpy as np
from src.data_loader import (
    map_to_project_schema,
    map_mcauley_to_project_schema,
    filter_reviews,
    get_dataset_stats,
)


class TestMapToProjectSchema:
    """Tests for map_to_project_schema function."""

    def test_returns_dataframe(self):
        df = pd.DataFrame({
            "content": ["Great product!", "Terrible."],
            "title": ["Love it", "Hate it"],
            "label": [1, 0]
        })
        result = map_to_project_schema(df, verbose=False)
        assert isinstance(result, pd.DataFrame)

    def test_creates_review_text_column(self):
        df = pd.DataFrame({
            "content": ["Great product!"],
            "title": ["Love it"],
            "label": [1]
        })
        result = map_to_project_schema(df, verbose=False)
        assert "review_text" in result.columns

    def test_creates_rating_column(self):
        df = pd.DataFrame({
            "content": ["Great product!"],
            "title": ["Love it"],
            "label": [1]
        })
        result = map_to_project_schema(df, verbose=False)
        assert "rating" in result.columns

    def test_creates_ground_truth_column(self):
        df = pd.DataFrame({
            "content": ["Great product!"],
            "title": ["Love it"],
            "label": [1]
        })
        result = map_to_project_schema(df, verbose=False)
        assert "ground_truth" in result.columns

    def test_maps_label_to_sentiment(self):
        df = pd.DataFrame({
            "content": ["Great!", "Terrible!"],
            "title": ["A", "B"],
            "label": [1, 0]
        })
        result = map_to_project_schema(df, verbose=False)
        assert result["ground_truth"].iloc[0] == "positive"
        assert result["ground_truth"].iloc[1] == "negative"

    def test_combines_title_and_content(self):
        df = pd.DataFrame({
            "content": ["Great product!"],
            "title": ["Love it"],
            "label": [1]
        })
        result = map_to_project_schema(df, verbose=False)
        assert "Love it" in result["review_text"].iloc[0]
        assert "Great product" in result["review_text"].iloc[0]


class TestMapMcauleyToProjectSchema:
    """Tests for map_mcauley_to_project_schema function."""

    def test_returns_dataframe(self):
        df = pd.DataFrame({
            "title": ["Love it"],
            "text": ["Great product!"],
            "rating": [5],
            "category": ["Electronics"],
            "brand": ["Acme"],
            "helpful_vote": [0],
            "verified_purchase": [True],
        })
        df["review_date"] = pd.NaT
        result = map_mcauley_to_project_schema(df, verbose=False)
        assert isinstance(result, pd.DataFrame)

    def test_creates_review_date_column(self):
        df = pd.DataFrame({
            "title": ["A"],
            "text": ["B" * 30],
            "rating": [5],
            "category": ["X"],
            "brand": ["Y"],
            "helpful_vote": [0],
            "verified_purchase": [True],
        })
        df["review_date"] = pd.to_datetime(["2020-01-15"])
        result = map_mcauley_to_project_schema(df, verbose=False)
        assert "review_date" in result.columns
        assert result["review_date"].notna().all()

    def test_maps_rating_to_ground_truth(self):
        df = pd.DataFrame({
            "title": ["A", "B", "C"],
            "text": ["x" * 30] * 3,
            "rating": [1, 3, 5],
            "category": ["X"] * 3,
            "brand": ["Y"] * 3,
            "helpful_vote": [0] * 3,
            "verified_purchase": [True] * 3,
        })
        df["review_date"] = pd.NaT
        result = map_mcauley_to_project_schema(df, verbose=False)
        assert result["ground_truth"].iloc[0] == "negative"
        assert result["ground_truth"].iloc[1] == "positive"
        assert result["ground_truth"].iloc[2] == "positive"


class TestFilterReviews:
    """Tests for filter_reviews function."""

    def test_returns_dataframe(self):
        df = pd.DataFrame({
            "review_text": ["This is a valid review with enough length."] * 3
        })
        result = filter_reviews(df, verbose=False)
        assert isinstance(result, pd.DataFrame)

    def test_filters_short_reviews(self):
        df = pd.DataFrame({
            "review_text": ["Short", "This is a much longer review text."]
        })
        result = filter_reviews(df, min_length=20, verbose=False)
        assert len(result) == 1

    def test_filters_long_reviews(self):
        df = pd.DataFrame({
            "review_text": ["Normal review.", "x" * 6000]
        })
        result = filter_reviews(df, min_length=5, max_length=5000, verbose=False)
        assert len(result) == 1

    def test_removes_duplicates(self):
        df = pd.DataFrame({
            "review_text": [
                "This is a duplicate review.",
                "This is a duplicate review.",
                "This is a unique review here."
            ]
        })
        result = filter_reviews(df, min_length=10, verbose=False)
        assert len(result) == 2

    def test_resets_index(self):
        df = pd.DataFrame({
            "review_text": ["Valid review one here.", "Valid review two here."]
        })
        result = filter_reviews(df, min_length=10, verbose=False)
        assert result.index.tolist() == [0, 1]


class TestGetDatasetStats:
    """Tests for get_dataset_stats function."""

    def test_returns_dict(self, sample_reviews):
        result = get_dataset_stats(sample_reviews)
        assert isinstance(result, dict)

    def test_contains_total_reviews(self, sample_reviews):
        result = get_dataset_stats(sample_reviews)
        assert "total_reviews" in result
        assert result["total_reviews"] == len(sample_reviews)

    def test_contains_categories_count(self, sample_reviews):
        result = get_dataset_stats(sample_reviews)
        assert "categories" in result

    def test_contains_avg_review_length(self, sample_reviews):
        result = get_dataset_stats(sample_reviews)
        assert "avg_review_length" in result
        assert result["avg_review_length"] > 0

    def test_contains_rating_distribution(self, sample_reviews):
        result = get_dataset_stats(sample_reviews)
        assert "rating_distribution" in result
        assert isinstance(result["rating_distribution"], dict)

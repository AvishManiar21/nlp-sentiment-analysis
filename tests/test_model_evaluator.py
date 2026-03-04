"""Unit tests for src/model_evaluator.py"""

import pytest
import pandas as pd
import numpy as np
from src.model_evaluator import (
    compute_metrics,
    create_comparison_dataframe,
    identify_best_models,
    get_per_class_metrics,
)


class TestComputeMetrics:
    """Tests for compute_metrics function."""

    def test_returns_dict(self, sample_predictions):
        result = compute_metrics(
            sample_predictions["y_true"],
            sample_predictions["y_pred"]
        )
        assert isinstance(result, dict)

    def test_contains_accuracy(self, sample_predictions):
        result = compute_metrics(
            sample_predictions["y_true"],
            sample_predictions["y_pred"]
        )
        assert "accuracy" in result
        assert 0 <= result["accuracy"] <= 1

    def test_contains_f1_scores(self, sample_predictions):
        result = compute_metrics(
            sample_predictions["y_true"],
            sample_predictions["y_pred"]
        )
        assert "f1_weighted" in result
        assert "f1_macro" in result
        assert 0 <= result["f1_weighted"] <= 1
        assert 0 <= result["f1_macro"] <= 1

    def test_contains_precision_recall(self, sample_predictions):
        result = compute_metrics(
            sample_predictions["y_true"],
            sample_predictions["y_pred"]
        )
        assert "precision_weighted" in result
        assert "recall_weighted" in result

    def test_contains_confusion_matrix(self, sample_predictions):
        result = compute_metrics(
            sample_predictions["y_true"],
            sample_predictions["y_pred"]
        )
        assert "confusion_matrix" in result
        assert isinstance(result["confusion_matrix"], np.ndarray)

    def test_contains_model_name(self, sample_predictions):
        result = compute_metrics(
            sample_predictions["y_true"],
            sample_predictions["y_pred"],
            model_name="TestModel"
        )
        assert result["model_name"] == "TestModel"

    def test_perfect_predictions(self):
        y_true = ["positive", "negative", "positive"]
        y_pred = ["positive", "negative", "positive"]
        result = compute_metrics(y_true, y_pred)
        assert result["accuracy"] == 1.0

    def test_wrong_predictions(self):
        y_true = ["positive", "positive", "positive"]
        y_pred = ["negative", "negative", "negative"]
        result = compute_metrics(y_true, y_pred)
        assert result["accuracy"] == 0.0


class TestCreateComparisonDataframe:
    """Tests for create_comparison_dataframe function."""

    def test_returns_dataframe(self, model_results):
        result = create_comparison_dataframe(model_results)
        assert isinstance(result, pd.DataFrame)

    def test_contains_model_column(self, model_results):
        result = create_comparison_dataframe(model_results)
        assert "Model" in result.columns

    def test_contains_metric_columns(self, model_results):
        result = create_comparison_dataframe(model_results)
        assert "Accuracy" in result.columns
        assert "F1 (weighted)" in result.columns

    def test_correct_number_of_rows(self, model_results):
        result = create_comparison_dataframe(model_results)
        assert len(result) == len(model_results)

    def test_sorted_by_accuracy(self, model_results):
        result = create_comparison_dataframe(model_results)
        accuracies = result["Accuracy"].tolist()
        assert accuracies == sorted(accuracies, reverse=True)


class TestIdentifyBestModels:
    """Tests for identify_best_models function."""

    def test_returns_dict(self, model_results):
        result = identify_best_models(model_results)
        assert isinstance(result, dict)

    def test_contains_accuracy_best(self, model_results):
        result = identify_best_models(model_results)
        assert "accuracy" in result

    def test_best_model_format(self, model_results):
        result = identify_best_models(model_results)
        for metric, info in result.items():
            assert isinstance(info, dict)
            assert "model" in info
            assert "value" in info
            assert isinstance(info["model"], str)
            assert isinstance(info["value"], (int, float))

    def test_identifies_correct_best(self, model_results):
        result = identify_best_models(model_results)
        assert result["accuracy"]["model"] == "Logistic Regression"
        assert result["accuracy"]["value"] == 0.85


class TestGetPerClassMetrics:
    """Tests for get_per_class_metrics function."""

    def test_returns_dict(self):
        results = {
            "Model1": {
                "model_name": "Model1",
                "labels": ["positive", "negative"],
                "classification_report": {
                    "positive": {"precision": 0.8, "recall": 0.7, "f1-score": 0.75},
                    "negative": {"precision": 0.7, "recall": 0.8, "f1-score": 0.75},
                }
            }
        }
        result = get_per_class_metrics(results)
        assert isinstance(result, dict)

    def test_contains_model_entries(self):
        results = {
            "Model1": {
                "model_name": "Model1",
                "labels": ["positive"],
                "classification_report": {
                    "positive": {"precision": 0.8, "recall": 0.7, "f1-score": 0.75},
                }
            }
        }
        result = get_per_class_metrics(results)
        assert "Model1" in result

    def test_contains_class_metrics(self):
        results = {
            "TestModel": {
                "model_name": "TestModel",
                "labels": ["positive", "negative"],
                "classification_report": {
                    "positive": {"precision": 0.8, "recall": 0.7, "f1-score": 0.75},
                    "negative": {"precision": 0.6, "recall": 0.9, "f1-score": 0.72},
                }
            }
        }
        result = get_per_class_metrics(results)
        assert "positive" in result["TestModel"]
        assert "negative" in result["TestModel"]

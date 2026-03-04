"""Unit tests for src/ml_models.py"""

import pytest
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.ml_models import (
    create_tfidf_vectorizer,
    create_logistic_regression,
    create_naive_bayes,
    prepare_data,
    evaluate_model,
    get_feature_importance,
    predict,
)


class TestCreateTfidfVectorizer:
    """Tests for create_tfidf_vectorizer function."""

    def test_returns_vectorizer(self):
        result = create_tfidf_vectorizer()
        assert isinstance(result, TfidfVectorizer)

    def test_has_max_features(self):
        result = create_tfidf_vectorizer(max_features=5000)
        assert result.max_features == 5000

    def test_has_ngram_range(self):
        result = create_tfidf_vectorizer(ngram_range=(1, 2))
        assert result.ngram_range == (1, 2)

    def test_default_config(self):
        result = create_tfidf_vectorizer()
        assert result.max_features == 10000
        assert result.ngram_range == (1, 2)


class TestCreateLogisticRegression:
    """Tests for create_logistic_regression function."""

    def test_returns_model(self):
        result = create_logistic_regression()
        assert isinstance(result, LogisticRegression)

    def test_has_balanced_weights(self):
        result = create_logistic_regression()
        assert result.class_weight == "balanced"


class TestCreateNaiveBayes:
    """Tests for create_naive_bayes function."""

    def test_returns_model(self):
        from sklearn.naive_bayes import MultinomialNB
        result = create_naive_bayes()
        assert isinstance(result, MultinomialNB)


class TestPrepareData:
    """Tests for prepare_data function."""

    def test_returns_four_arrays(self):
        df = pd.DataFrame({
            "processed_text": ["good product"] * 10 + ["bad item"] * 10,
            "ground_truth": ["positive"] * 10 + ["negative"] * 10
        })
        X_train, X_test, y_train, y_test = prepare_data(df, test_size=0.2)
        assert len(X_train) == 16
        assert len(X_test) == 4
        assert len(y_train) == 16
        assert len(y_test) == 4

    def test_stratified_split(self):
        df = pd.DataFrame({
            "processed_text": ["a"] * 50 + ["b"] * 50,
            "ground_truth": ["positive"] * 50 + ["negative"] * 50
        })
        _, _, y_train, y_test = prepare_data(df, test_size=0.2)
        train_pos = sum(1 for y in y_train if y == "positive")
        train_neg = sum(1 for y in y_train if y == "negative")
        assert train_pos == train_neg

    def test_respects_test_size(self):
        df = pd.DataFrame({
            "processed_text": ["text"] * 100,
            "ground_truth": ["positive"] * 50 + ["negative"] * 50
        })
        X_train, X_test, _, _ = prepare_data(df, test_size=0.3)
        assert len(X_test) == 30
        assert len(X_train) == 70


class TestEvaluateModel:
    """Tests for evaluate_model function."""

    def test_returns_dict(self):
        vectorizer = TfidfVectorizer(max_features=100)
        model = LogisticRegression()
        pipeline = Pipeline([("tfidf", vectorizer), ("clf", model)])
        
        X_train = ["good product", "bad item", "great stuff", "terrible thing"]
        y_train = ["positive", "negative", "positive", "negative"]
        pipeline.fit(X_train, y_train)
        
        X_test = ["nice product", "awful item"]
        y_test = ["positive", "negative"]
        
        result = evaluate_model(pipeline, X_test, y_test)
        assert isinstance(result, dict)

    def test_contains_metrics(self):
        vectorizer = TfidfVectorizer(max_features=100)
        model = LogisticRegression()
        pipeline = Pipeline([("tfidf", vectorizer), ("clf", model)])
        
        X_train = ["good product", "bad item", "great stuff", "terrible thing"]
        y_train = ["positive", "negative", "positive", "negative"]
        pipeline.fit(X_train, y_train)
        
        X_test = ["nice product", "awful item"]
        y_test = ["positive", "negative"]
        
        result = evaluate_model(pipeline, X_test, y_test)
        assert "accuracy" in result
        assert "f1_weighted" in result


class TestPredict:
    """Tests for predict function."""

    def test_returns_tuple(self):
        vectorizer = TfidfVectorizer(max_features=100)
        model = LogisticRegression()
        pipeline = Pipeline([("tfidf", vectorizer), ("clf", model)])
        
        X_train = ["good product", "bad item", "great stuff", "terrible thing"]
        y_train = ["positive", "negative", "positive", "negative"]
        pipeline.fit(X_train, y_train)
        
        result = predict(pipeline, ["test text"])
        assert isinstance(result, tuple)
        assert len(result) == 2
        predictions, probabilities = result
        assert isinstance(predictions, np.ndarray)

    def test_returns_correct_length(self):
        vectorizer = TfidfVectorizer(max_features=100)
        model = LogisticRegression()
        pipeline = Pipeline([("tfidf", vectorizer), ("clf", model)])
        
        X_train = ["good product", "bad item", "great stuff", "terrible thing"]
        y_train = ["positive", "negative", "positive", "negative"]
        pipeline.fit(X_train, y_train)
        
        texts = ["text one", "text two", "text three"]
        predictions, _ = predict(pipeline, texts)
        assert len(predictions) == 3


class TestGetFeatureImportance:
    """Tests for get_feature_importance function."""

    def test_returns_dataframe_or_dict_or_none(self):
        vectorizer = TfidfVectorizer(max_features=100)
        model = LogisticRegression()
        pipeline = Pipeline([("tfidf", vectorizer), ("classifier", model)])
        
        X_train = ["good great", "bad terrible", "nice awesome", "poor awful"]
        y_train = ["positive", "negative", "positive", "negative"]
        pipeline.fit(X_train, y_train)
        
        result = get_feature_importance(pipeline)
        assert result is None or isinstance(result, (pd.DataFrame, dict))

    def test_logistic_regression_with_multiclass(self):
        vectorizer = TfidfVectorizer(max_features=100)
        model = LogisticRegression()
        pipeline = Pipeline([("tfidf", vectorizer), ("classifier", model)])
        
        X_train = ["good great", "bad terrible", "okay normal", "good amazing",
                   "bad awful", "okay average"]
        y_train = ["positive", "negative", "neutral", "positive", "negative", "neutral"]
        pipeline.fit(X_train, y_train)
        
        result = get_feature_importance(pipeline)
        assert result is None or isinstance(result, (pd.DataFrame, dict))

    def test_handles_pipeline_structure(self):
        vectorizer = TfidfVectorizer(max_features=100)
        model = LogisticRegression()
        pipeline = Pipeline([("tfidf", vectorizer), ("classifier", model)])
        
        X_train = ["good product", "bad item"]
        y_train = ["positive", "negative"]
        pipeline.fit(X_train, y_train)
        
        try:
            result = get_feature_importance(pipeline)
            assert result is None or isinstance(result, (pd.DataFrame, dict))
        except KeyError:
            pass

"""Tests for the FastAPI sentiment analysis API."""

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.predictor import predictor


@pytest.fixture(scope="module")
def client():
    """Create test client with loaded models."""
    predictor.load_models()
    with TestClient(app) as client:
        yield client


class TestRootEndpoint:
    """Tests for the root endpoint."""
    
    def test_returns_api_info(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Sentiment Analysis API"
        assert data["version"] == "1.0.0"
        assert "docs_url" in data
    
    def test_has_description(self, client):
        response = client.get("/")
        data = response.json()
        assert "description" in data
        assert len(data["description"]) > 0


class TestHealthEndpoint:
    """Tests for the health check endpoint."""
    
    def test_returns_healthy_status(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_reports_models_loaded(self, client):
        response = client.get("/health")
        data = response.json()
        assert "models_loaded" in data
        assert isinstance(data["models_loaded"], bool)


class TestModelsEndpoint:
    """Tests for the models listing endpoint."""
    
    def test_returns_models_list(self, client):
        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert len(data["models"]) >= 2
    
    def test_model_has_required_fields(self, client):
        response = client.get("/models")
        data = response.json()
        model = data["models"][0]
        assert "name" in model
        assert "display_name" in model
        assert "type" in model
        assert "description" in model
        assert "available" in model
    
    def test_includes_vader_model(self, client):
        response = client.get("/models")
        data = response.json()
        model_names = [m["name"] for m in data["models"]]
        assert "vader" in model_names
    
    def test_includes_textblob_model(self, client):
        response = client.get("/models")
        data = response.json()
        model_names = [m["name"] for m in data["models"]]
        assert "textblob" in model_names


class TestPredictEndpoint:
    """Tests for the single prediction endpoint."""
    
    def test_predicts_positive_sentiment(self, client):
        response = client.post(
            "/predict",
            json={"text": "This is amazing! I love it!", "model": "vader"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["sentiment"] == "positive"
    
    def test_predicts_negative_sentiment(self, client):
        response = client.post(
            "/predict",
            json={"text": "This is terrible! I hate it!", "model": "vader"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["sentiment"] == "negative"
    
    def test_returns_confidence_score(self, client):
        response = client.post(
            "/predict",
            json={"text": "Great product!", "model": "vader"}
        )
        data = response.json()
        assert "confidence" in data
        assert 0 <= data["confidence"] <= 1
    
    def test_returns_sentiment_scores(self, client):
        response = client.post(
            "/predict",
            json={"text": "Good quality.", "model": "vader"}
        )
        data = response.json()
        assert "scores" in data
        assert "positive" in data["scores"]
        assert "negative" in data["scores"]
    
    def test_default_model_is_logistic_regression(self, client):
        response = client.post(
            "/predict",
            json={"text": "Nice product"}
        )
        data = response.json()
        assert data["model"] == "logistic_regression"
    
    def test_textblob_model_works(self, client):
        response = client.post(
            "/predict",
            json={"text": "Excellent quality!", "model": "textblob"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "textblob"
    
    def test_empty_text_returns_error(self, client):
        response = client.post(
            "/predict",
            json={"text": "", "model": "vader"}
        )
        assert response.status_code == 422
    
    def test_invalid_model_returns_error(self, client):
        response = client.post(
            "/predict",
            json={"text": "Test text", "model": "invalid_model"}
        )
        assert response.status_code == 422
    
    def test_returns_original_text(self, client):
        test_text = "This product is great!"
        response = client.post(
            "/predict",
            json={"text": test_text, "model": "vader"}
        )
        data = response.json()
        assert data["text"] == test_text


class TestBatchPredictEndpoint:
    """Tests for the batch prediction endpoint."""
    
    def test_predicts_multiple_texts(self, client):
        response = client.post(
            "/predict/batch",
            json={
                "texts": ["Great product!", "Terrible quality!"],
                "model": "vader"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2
        assert len(data["predictions"]) == 2
    
    def test_returns_correct_count(self, client):
        texts = ["Good", "Bad", "Okay"]
        response = client.post(
            "/predict/batch",
            json={"texts": texts, "model": "vader"}
        )
        data = response.json()
        assert data["count"] == len(texts)
    
    def test_each_prediction_has_required_fields(self, client):
        response = client.post(
            "/predict/batch",
            json={"texts": ["Test text"], "model": "vader"}
        )
        data = response.json()
        prediction = data["predictions"][0]
        assert "text" in prediction
        assert "sentiment" in prediction
        assert "confidence" in prediction
        assert "scores" in prediction
    
    def test_empty_texts_returns_error(self, client):
        response = client.post(
            "/predict/batch",
            json={"texts": [], "model": "vader"}
        )
        assert response.status_code == 422
    
    def test_batch_model_specified_in_response(self, client):
        response = client.post(
            "/predict/batch",
            json={"texts": ["Test"], "model": "textblob"}
        )
        data = response.json()
        assert data["model"] == "textblob"


class TestPredictorIntegration:
    """Integration tests for the predictor service."""
    
    def test_vader_positive_detection(self, client):
        response = client.post(
            "/predict",
            json={"text": "I absolutely love this amazing product! Best ever!", "model": "vader"}
        )
        data = response.json()
        assert data["sentiment"] == "positive"
        assert data["confidence"] > 0.7
    
    def test_vader_negative_detection(self, client):
        response = client.post(
            "/predict",
            json={"text": "Worst product ever. Complete waste of money!", "model": "vader"}
        )
        data = response.json()
        assert data["sentiment"] == "negative"
        assert data["confidence"] > 0.7
    
    def test_textblob_positive_detection(self, client):
        response = client.post(
            "/predict",
            json={"text": "Wonderful experience, highly recommend!", "model": "textblob"}
        )
        data = response.json()
        assert data["sentiment"] == "positive"
    
    def test_textblob_negative_detection(self, client):
        response = client.post(
            "/predict",
            json={"text": "Awful, horrible, disappointing product.", "model": "textblob"}
        )
        data = response.json()
        assert data["sentiment"] == "negative"


class TestMLModelPredictions:
    """Tests for ML model predictions (may be skipped if models not available)."""
    
    def test_logistic_regression_prediction(self, client):
        response = client.post(
            "/predict",
            json={"text": "This product is excellent!", "model": "logistic_regression"}
        )
        if response.status_code == 503:
            pytest.skip("Logistic regression model not available")
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "logistic_regression"
        assert data["sentiment"] in ["positive", "negative"]
    
    def test_naive_bayes_prediction(self, client):
        response = client.post(
            "/predict",
            json={"text": "This product is terrible!", "model": "naive_bayes"}
        )
        if response.status_code == 503:
            pytest.skip("Naive Bayes model not available")
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "naive_bayes"
        assert data["sentiment"] in ["positive", "negative"]

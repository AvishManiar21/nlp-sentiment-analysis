"""Prediction service for sentiment analysis API."""

import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import nltk

from src.ml_models import load_model, MODELS_DIR
from src.sentiment_analyzer import sanitize_text, ensure_nltk_data


class SentimentPredictor:
    """Unified prediction service for all sentiment models."""
    
    def __init__(self):
        self._vader_analyzer: Optional[SentimentIntensityAnalyzer] = None
        self._ml_models: dict = {}
        self._models_loaded = False
    
    def load_models(self) -> bool:
        """Load all available models at startup."""
        ensure_nltk_data()
        self._vader_analyzer = SentimentIntensityAnalyzer()
        
        ml_model_types = ["logistic_regression", "naive_bayes"]
        for model_type in ml_model_types:
            try:
                self._ml_models[model_type] = load_model(model_type)
            except FileNotFoundError:
                pass
        
        self._models_loaded = True
        return True
    
    @property
    def models_loaded(self) -> bool:
        """Check if models are loaded."""
        return self._models_loaded
    
    def get_available_models(self) -> list[dict]:
        """Get list of available models with metadata."""
        models = [
            {
                "name": "vader",
                "display_name": "VADER",
                "type": "rule-based",
                "description": "Valence Aware Dictionary and sEntiment Reasoner - optimized for social media",
                "available": self._vader_analyzer is not None,
            },
            {
                "name": "textblob",
                "display_name": "TextBlob",
                "type": "rule-based",
                "description": "Pattern-based sentiment analysis using linguistic patterns",
                "available": True,
            },
            {
                "name": "logistic_regression",
                "display_name": "Logistic Regression",
                "type": "ml",
                "description": "TF-IDF features with logistic regression classifier",
                "available": "logistic_regression" in self._ml_models,
            },
            {
                "name": "naive_bayes",
                "display_name": "Naive Bayes",
                "type": "ml",
                "description": "TF-IDF features with multinomial Naive Bayes classifier",
                "available": "naive_bayes" in self._ml_models,
            },
        ]
        return models
    
    def predict_vader(self, text: str) -> dict:
        """Predict sentiment using VADER."""
        text = sanitize_text(text)
        
        if self._vader_analyzer is None:
            raise RuntimeError("VADER analyzer not initialized")
        
        scores = self._vader_analyzer.polarity_scores(text)
        compound = scores["compound"]
        
        if compound >= 0.05:
            sentiment = "positive"
            confidence = min(1.0, (compound + 1) / 2 + 0.3)
        elif compound <= -0.05:
            sentiment = "negative"
            confidence = min(1.0, (1 - compound) / 2 + 0.3)
        else:
            sentiment = "positive" if compound >= 0 else "negative"
            confidence = 0.5 + abs(compound) * 2
        
        pos_score = (compound + 1) / 2
        neg_score = 1 - pos_score
        
        return {
            "sentiment": sentiment,
            "confidence": round(confidence, 4),
            "scores": {
                "positive": round(pos_score, 4),
                "negative": round(neg_score, 4),
            },
        }
    
    def predict_textblob(self, text: str) -> dict:
        """Predict sentiment using TextBlob."""
        text = sanitize_text(text)
        
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity >= 0.05:
            sentiment = "positive"
            confidence = min(1.0, 0.5 + polarity * 0.5)
        elif polarity <= -0.05:
            sentiment = "negative"
            confidence = min(1.0, 0.5 + abs(polarity) * 0.5)
        else:
            sentiment = "positive" if polarity >= 0 else "negative"
            confidence = 0.5 + abs(polarity) * 2
        
        pos_score = (polarity + 1) / 2
        neg_score = 1 - pos_score
        
        return {
            "sentiment": sentiment,
            "confidence": round(confidence, 4),
            "scores": {
                "positive": round(pos_score, 4),
                "negative": round(neg_score, 4),
            },
        }
    
    def predict_ml(self, text: str, model_name: str) -> dict:
        """Predict sentiment using an ML model."""
        if model_name not in self._ml_models:
            raise ValueError(f"Model '{model_name}' not available")
        
        pipeline = self._ml_models[model_name]
        text = sanitize_text(text)
        
        prediction = pipeline.predict([text])[0]
        
        if hasattr(pipeline, "predict_proba"):
            try:
                proba = pipeline.predict_proba([text])[0]
                classes = pipeline.classes_
                
                class_proba = dict(zip(classes, proba))
                
                pos_score = class_proba.get("positive", 0.0)
                neg_score = class_proba.get("negative", 0.0)
                
                confidence = max(pos_score, neg_score)
            except AttributeError:
                pos_score = 1.0 if prediction == "positive" else 0.0
                neg_score = 1.0 - pos_score
                confidence = 0.9
        else:
            pos_score = 1.0 if prediction == "positive" else 0.0
            neg_score = 1.0 - pos_score
            confidence = 0.9
        
        return {
            "sentiment": prediction,
            "confidence": round(float(confidence), 4),
            "scores": {
                "positive": round(float(pos_score), 4),
                "negative": round(float(neg_score), 4),
            },
        }
    
    def predict(self, text: str, model: str = "logistic_regression") -> dict:
        """
        Unified prediction interface for all models.
        
        Args:
            text: Text to analyze
            model: Model to use (vader, textblob, logistic_regression, naive_bayes)
        
        Returns:
            Dictionary with sentiment, confidence, and scores
        """
        if model == "vader":
            return self.predict_vader(text)
        elif model == "textblob":
            return self.predict_textblob(text)
        elif model in ["logistic_regression", "naive_bayes"]:
            return self.predict_ml(text, model)
        else:
            raise ValueError(f"Unknown model: {model}")


predictor = SentimentPredictor()

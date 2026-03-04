"""Pydantic models for API request/response validation."""

from typing import Optional
from enum import Enum

from pydantic import BaseModel, Field


class ModelType(str, Enum):
    """Available model types for sentiment prediction."""
    VADER = "vader"
    TEXTBLOB = "textblob"
    LOGISTIC_REGRESSION = "logistic_regression"
    NAIVE_BAYES = "naive_bayes"


class PredictRequest(BaseModel):
    """Request body for single text prediction."""
    text: str = Field(..., min_length=1, description="Text to analyze for sentiment")
    model: ModelType = Field(
        default=ModelType.LOGISTIC_REGRESSION,
        description="Model to use for prediction"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "This product is amazing! Best purchase ever.",
                    "model": "logistic_regression"
                }
            ]
        }
    }


class SentimentScores(BaseModel):
    """Sentiment probability scores."""
    positive: float = Field(..., ge=0, le=1)
    negative: float = Field(..., ge=0, le=1)


class PredictResponse(BaseModel):
    """Response body for sentiment prediction."""
    text: str = Field(..., description="Original input text")
    model: str = Field(..., description="Model used for prediction")
    sentiment: str = Field(..., description="Predicted sentiment (positive/negative)")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    scores: SentimentScores = Field(..., description="Probability scores per class")


class BatchPredictRequest(BaseModel):
    """Request body for batch text prediction."""
    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of texts to analyze (max 100)"
    )
    model: ModelType = Field(
        default=ModelType.LOGISTIC_REGRESSION,
        description="Model to use for prediction"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "texts": ["Great product!", "Terrible quality, waste of money."],
                    "model": "vader"
                }
            ]
        }
    }


class BatchPredictResponse(BaseModel):
    """Response body for batch sentiment prediction."""
    model: str = Field(..., description="Model used for prediction")
    count: int = Field(..., description="Number of texts processed")
    predictions: list[PredictResponse] = Field(..., description="List of predictions")


class ModelInfo(BaseModel):
    """Information about an available model."""
    name: str = Field(..., description="Model identifier")
    display_name: str = Field(..., description="Human-readable model name")
    type: str = Field(..., description="Model type (rule-based or ml)")
    description: str = Field(..., description="Brief description of the model")
    available: bool = Field(..., description="Whether the model is loaded and available")


class ModelsResponse(BaseModel):
    """Response body for listing available models."""
    models: list[ModelInfo] = Field(..., description="List of available models")


class HealthResponse(BaseModel):
    """Response body for health check endpoint."""
    status: str = Field(..., description="Health status")
    models_loaded: bool = Field(..., description="Whether ML models are loaded")


class APIInfo(BaseModel):
    """Response body for API info endpoint."""
    name: str = Field(..., description="API name")
    version: str = Field(..., description="API version")
    description: str = Field(..., description="API description")
    docs_url: str = Field(..., description="Documentation URL")


class ErrorResponse(BaseModel):
    """Standard error response."""
    detail: str = Field(..., description="Error message")

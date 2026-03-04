"""FastAPI application for sentiment analysis predictions."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    PredictRequest,
    PredictResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    SentimentScores,
    ModelInfo,
    ModelsResponse,
    HealthResponse,
    APIInfo,
)
from api.predictor import predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, cleanup on shutdown."""
    predictor.load_models()
    yield


app = FastAPI(
    title="Sentiment Analysis API",
    description="REST API for sentiment analysis using rule-based and ML models",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=APIInfo)
async def root():
    """Get API information."""
    return APIInfo(
        name="Sentiment Analysis API",
        version="1.0.0",
        description="Analyze text sentiment using VADER, TextBlob, and ML models",
        docs_url="/docs",
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health status."""
    return HealthResponse(
        status="healthy",
        models_loaded=predictor.models_loaded,
    )


@app.get("/models", response_model=ModelsResponse)
async def list_models():
    """List all available models."""
    models = predictor.get_available_models()
    return ModelsResponse(
        models=[
            ModelInfo(
                name=m["name"],
                display_name=m["display_name"],
                type=m["type"],
                description=m["description"],
                available=m["available"],
            )
            for m in models
        ]
    )


@app.post("/predict", response_model=PredictResponse)
async def predict_sentiment(request: PredictRequest):
    """Predict sentiment for a single text."""
    try:
        result = predictor.predict(request.text, request.model.value)
        
        return PredictResponse(
            text=request.text,
            model=request.model.value,
            sentiment=result["sentiment"],
            confidence=result["confidence"],
            scores=SentimentScores(
                positive=result["scores"]["positive"],
                negative=result["scores"]["negative"],
            ),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(request: BatchPredictRequest):
    """Predict sentiment for multiple texts."""
    try:
        predictions = []
        
        for text in request.texts:
            result = predictor.predict(text, request.model.value)
            
            predictions.append(
                PredictResponse(
                    text=text,
                    model=request.model.value,
                    sentiment=result["sentiment"],
                    confidence=result["confidence"],
                    scores=SentimentScores(
                        positive=result["scores"]["positive"],
                        negative=result["scores"]["negative"],
                    ),
                )
            )
        
        return BatchPredictResponse(
            model=request.model.value,
            count=len(predictions),
            predictions=predictions,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)

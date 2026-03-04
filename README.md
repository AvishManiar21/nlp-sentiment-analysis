# NLP Sentiment Analysis & Opinion Mining

[![CI](https://github.com/AvishManiar21/nlp-sentiment-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/AvishManiar21/nlp-sentiment-analysis/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/AvishManiar21/nlp-sentiment-analysis/branch/master/graph/badge.svg)](https://codecov.io/gh/AvishManiar21/nlp-sentiment-analysis)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![HuggingFace Dataset](https://img.shields.io/badge/HuggingFace-Dataset-orange)](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)

A comprehensive sentiment analysis pipeline using **real Amazon product reviews** with multiple ML models, rigorous evaluation metrics, and interactive visualizations.

## Features

- **Real-World Data**: ~50,000 genuine Amazon product reviews with real timestamps, categories, and brands
- **Multiple Sentiment Models**:
  - VADER (rule-based, optimized for social media)
  - TextBlob (pattern-based polarity analysis)
  - Logistic Regression (TF-IDF features) - **89.6% accuracy**
  - Multinomial Naive Bayes (TF-IDF features) - **87.1% accuracy**
  - DistilBERT fine-tuning (optional, requires GPU)
- **Rigorous Evaluation**: Train/test split, accuracy, precision, recall, F1, confusion matrices
- **Opinion Mining**: Aspect extraction, sentiment drivers, category analysis
- **Interactive Dashboard**: Streamlit app with filters, visualizations, model comparison
- **REST API**: FastAPI endpoints for real-time sentiment predictions

## Dataset

**Primary: McAuley-Lab Amazon Reviews 2023** – Real categories, brands, ratings, and timestamps.

- **Source**: JSONL files from [HuggingFace](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) (UCSD datarepo URLs attempted first)
- **Fields**: Review text, ratings, timestamps, main category, store (brand), helpful votes, verified purchase
- **Sample size**: ~50,000 reviews (configurable; adjust for larger samples on more powerful machines)
- **Labels**: Sentiment from star ratings (≤2 negative, ≥4 positive, 3 neutral)


## Results

Performance on 10,000 test samples (binary classification):

| Model | Accuracy | F1 Score | Precision | Recall |
|-------|----------|----------|-----------|--------|
| **Logistic Regression** | **89.6%** | 0.896 | 0.896 | 0.896 |
| Naive Bayes | 87.1% | 0.871 | 0.871 | 0.871 |
| VADER | 70.5% | 0.699 | 0.770 | 0.705 |
| Ensemble (VADER+TextBlob) | 70.3% | 0.698 | 0.774 | 0.703 |
| TextBlob | 61.5% | 0.622 | 0.794 | 0.615 |

*ML models significantly outperform rule-based methods when labeled training data is available.*

## Project Structure

```
nlp-sentiment-analysis/
├── main.py                    # Main pipeline script
├── app.py                     # Streamlit dashboard
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── src/
│   ├── data_loader.py         # Amazon Reviews fetcher (HuggingFace)
│   ├── preprocessor.py        # Text cleaning, tokenization, lemmatization
│   ├── sentiment_analyzer.py  # VADER + TextBlob analysis
│   ├── ml_models.py           # Logistic Regression, Naive Bayes training
│   ├── transformer_model.py   # DistilBERT fine-tuning
│   ├── model_evaluator.py     # Model comparison and metrics
│   ├── opinion_miner.py       # Aspect extraction, driver analysis
│   └── visualizer.py          # Charts and visualizations
├── api/
│   ├── main.py                # FastAPI application
│   ├── schemas.py             # Pydantic request/response models
│   └── predictor.py           # Model prediction service
├── tests/                     # Unit and API tests
├── data/                      # Generated data files (gitignored)
├── models/                    # Saved ML models (gitignored)
├── outputs/                   # Visualization outputs (gitignored)
└── results/                   # Evaluation results (gitignored)
```

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/nlp-sentiment-analysis.git
cd nlp-sentiment-analysis
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the pipeline

```bash
python main.py
```

This will:
1. Download ~50,000 real Amazon reviews (McAuley-Lab JSONL, with amazon_polarity fallback)
2. Preprocess text (clean, tokenize, lemmatize)
3. Run VADER + TextBlob sentiment analysis
4. Train Logistic Regression and Naive Bayes models
5. Evaluate and compare all models
6. Generate visualizations

### 5. Launch the dashboard

```bash
python -m streamlit run app.py
```

Open http://localhost:8501 in your browser.

### 6. Run the REST API

```bash
uvicorn api.main:app --reload --port 8000
```

Open http://localhost:8000/docs for interactive API documentation.

## REST API

The project includes a FastAPI-based REST API for sentiment predictions.

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info and health check |
| GET | `/health` | Health status |
| GET | `/models` | List available models |
| POST | `/predict` | Predict sentiment for single text |
| POST | `/predict/batch` | Predict sentiment for multiple texts |

### Single Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is amazing!", "model": "logistic_regression"}'
```

Response:

```json
{
  "text": "This product is amazing!",
  "model": "logistic_regression",
  "sentiment": "positive",
  "confidence": 0.92,
  "scores": {"positive": 0.92, "negative": 0.08}
}
```

### Batch Prediction

```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great product!", "Awful quality"], "model": "vader"}'
```

### Available Models

- `vader` - Rule-based (VADER)
- `textblob` - Rule-based (TextBlob)
- `logistic_regression` - ML model (default)
- `naive_bayes` - ML model

## Pipeline Options

```bash
# Force reload data from HuggingFace
python main.py --force-reload

# Skip ML model training
python main.py --skip-ml

# Train DistilBERT (requires GPU for reasonable speed)
python main.py --train-transformer

# Limit dataset size for quick testing
python main.py --sample-size 5000
```

## Models & Methodology

### Rule-Based Models (Unsupervised)

| Model | Description |
|-------|-------------|
| **VADER** | Lexicon-based sentiment analyzer optimized for social media. Returns compound score (-1 to +1). |
| **TextBlob** | Pattern-based NLP library. Returns polarity (-1 to +1) and subjectivity (0 to 1). |
| **Ensemble** | Weighted combination: 65% VADER + 35% TextBlob |

### ML Models (Supervised)

| Model | Description |
|-------|-------------|
| **Logistic Regression** | Linear classifier with TF-IDF features (1-2 grams, 10K features). Balanced class weights. |
| **Naive Bayes** | Multinomial NB with TF-IDF features. Fast training, good baseline. |
| **DistilBERT** | Fine-tuned transformer model. ~97% of BERT performance, 60% faster. (Optional) |

### Evaluation Metrics

- **Accuracy**: Overall correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **Cohen's Kappa**: Agreement accounting for chance
- **Matthews Correlation Coefficient**: Balanced measure for imbalanced classes

## Key Insights

1. **ML models significantly outperform rule-based methods** when labeled training data is available
2. **Logistic Regression** achieves 89.6% accuracy with simple TF-IDF features
3. **VADER** performs reasonably well (70.5%) as an unsupervised baseline
4. **TextBlob** tends to produce more neutral predictions, lowering accuracy
5. **Aspect analysis** reveals which product features drive positive/negative sentiment

## Technologies Used

- **Python 3.10+**
- **scikit-learn** - ML models and evaluation
- **NLTK** - VADER sentiment, tokenization
- **TextBlob** - Pattern-based sentiment
- **HuggingFace Datasets** - Data loading
- **FastAPI** - REST API
- **Streamlit** - Interactive dashboard
- **Plotly/Matplotlib** - Visualizations
- **pandas/numpy** - Data processing

## License

MIT License - see [LICENSE](LICENSE) for details.

The Amazon Reviews dataset is subject to Amazon's terms of use for research purposes.

## Author

Built as a demonstration of NLP sentiment analysis techniques, from rule-based methods to machine learning classifiers.

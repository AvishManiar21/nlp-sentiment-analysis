# NLP Sentiment Analysis & Opinion Mining

A comprehensive sentiment analysis pipeline using **real Amazon product reviews** with multiple ML models, evaluation metrics, and interactive visualizations.

## Features

- **Real-World Data**: Fetches ~50K product reviews from Amazon Reviews 2023 dataset via HuggingFace
- **Multiple Sentiment Models**:
  - VADER (rule-based, optimized for social media)
  - TextBlob (pattern-based polarity analysis)
  - Logistic Regression (TF-IDF features)
  - Multinomial Naive Bayes (TF-IDF features)
  - DistilBERT fine-tuning (optional, requires GPU)
- **Rigorous Evaluation**: Train/test split, accuracy, precision, recall, F1, confusion matrices
- **Opinion Mining**: Aspect extraction, sentiment drivers (TF-IDF), category/brand analysis
- **Interactive Dashboard**: Streamlit app with filters, visualizations, model comparison

## Dataset

**McAuley-Lab/Amazon-Reviews-2023** - A large-scale dataset containing 571M+ reviews across 33 product categories.

We sample reviews from:
- Cell Phones & Accessories (Smartphones)
- Electronics
- Appliances
- Office Products
- Home & Kitchen

Ground truth sentiment labels are derived from star ratings:
- **Negative**: 1-2 stars
- **Neutral**: 3 stars
- **Positive**: 4-5 stars

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
├── data/                      # Generated data files
├── models/                    # Saved ML models
├── outputs/                   # Visualization outputs
└── results/                   # Evaluation results
```

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd nlp-sentiment-analysis
```

### 2. Create virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK data

```python
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

### 5. (Optional) Download spaCy model for aspect extraction

```bash
python -m spacy download en_core_web_sm
```

## Usage

### Run the full pipeline

```bash
python main.py
```

This will:
1. Fetch Amazon reviews from HuggingFace (~50K reviews)
2. Preprocess text (clean, tokenize, lemmatize)
3. Run VADER + TextBlob sentiment analysis
4. Train Logistic Regression and Naive Bayes models
5. Evaluate and compare all models
6. Generate visualizations

### Pipeline options

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

### Run the dashboard

```bash
python -m streamlit run app.py
```

Open http://localhost:8501 in your browser.

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
| **DistilBERT** | Fine-tuned transformer model. ~97% of BERT performance, 60% faster. |

### Evaluation Metrics

- **Accuracy**: Overall correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **Cohen's Kappa**: Agreement accounting for chance
- **Matthews Correlation Coefficient**: Balanced measure for imbalanced classes

## Results

Sample results (actual performance depends on data sample):

| Model | Accuracy | F1 (weighted) | F1 (macro) |
|-------|----------|---------------|------------|
| Logistic Regression | ~0.85 | ~0.84 | ~0.75 |
| Naive Bayes | ~0.80 | ~0.79 | ~0.70 |
| Ensemble (VADER+TB) | ~0.65 | ~0.60 | ~0.55 |
| VADER | ~0.60 | ~0.55 | ~0.50 |
| TextBlob | ~0.55 | ~0.50 | ~0.45 |

*Note: Rule-based methods (VADER, TextBlob) are not trained on the data and serve as unsupervised baselines.*

## Key Insights

1. **ML models significantly outperform rule-based methods** when labeled training data is available
2. **Logistic Regression** achieves strong performance with simple TF-IDF features
3. **VADER** performs reasonably well as an unsupervised baseline
4. **TextBlob** tends to produce more neutral predictions
5. **Aspect analysis** reveals which product features drive positive/negative sentiment

## Citation

If you use the Amazon Reviews dataset, please cite:

```bibtex
@article{hou2024bridging,
  title={Bridging Language and Items for Retrieval and Recommendation},
  author={Hou, Yupeng and Li, Jiacheng and He, Zhankui and Yan, An and Chen, Xiusi and McAuley, Julian},
  journal={arXiv preprint arXiv:2403.03952},
  year={2024}
}
```

## License

This project is for educational and portfolio purposes. The Amazon Reviews dataset is subject to its own license terms.

## Author

Built as a demonstration of NLP sentiment analysis techniques, from rule-based methods to fine-tuned transformers.

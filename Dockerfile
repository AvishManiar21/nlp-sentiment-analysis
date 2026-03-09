# Multi-stage build for NLP Sentiment Analysis Dashboard
# Stage 1: Build dependencies
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

# Stage 2: Production image
FROM python:3.11-slim as production

WORKDIR /app

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels from builder stage
COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir /wheels/*

# Download NLTK data
RUN python -c "import nltk; \
    nltk.download('vader_lexicon', quiet=True); \
    nltk.download('punkt', quiet=True); \
    nltk.download('punkt_tab', quiet=True); \
    nltk.download('stopwords', quiet=True); \
    nltk.download('wordnet', quiet=True); \
    nltk.download('averaged_perceptron_tagger', quiet=True); \
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)"

# Copy application code
COPY --chown=appuser:appuser . .

# Create necessary directories
RUN mkdir -p data models outputs results logs && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose ports
EXPOSE 8501 8000

# Health check for Streamlit
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Default command: Run Streamlit dashboard
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]

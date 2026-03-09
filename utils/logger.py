"""Structured logging configuration for the application."""

import logging
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
import os


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        if hasattr(record, "extra_data"):
            log_entry["data"] = record.extra_data
        
        return json.dumps(log_entry)


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""
    
    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
    }
    RESET = "\033[0m"
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(
    level: str = "INFO",
    log_format: str = "text",
    log_file: Optional[str] = None,
    app_name: str = "nlp-sentiment",
) -> logging.Logger:
    """
    Set up structured logging for the application.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Format type ('json' or 'text')
        log_file: Optional file path for logging
        app_name: Application name for the logger
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(app_name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    logger.handlers.clear()
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    
    if log_format == "json":
        console_handler.setFormatter(JSONFormatter())
    else:
        formatter = ColoredFormatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JSONFormatter())
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "nlp-sentiment") -> logging.Logger:
    """Get or create a logger with the specified name."""
    return logging.getLogger(name)


class LoggerAdapter(logging.LoggerAdapter):
    """Logger adapter for adding context to log messages."""
    
    def process(self, msg, kwargs):
        extra = kwargs.get("extra", {})
        extra.update(self.extra)
        kwargs["extra"] = extra
        return msg, kwargs


def create_request_logger(request_id: str) -> LoggerAdapter:
    """Create a logger with request context."""
    logger = get_logger()
    return LoggerAdapter(logger, {"request_id": request_id})


LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv("LOG_FORMAT", "text")
LOG_FILE = os.getenv("LOG_FILE", None)

logger = setup_logging(
    level=LOG_LEVEL,
    log_format=LOG_FORMAT,
    log_file=LOG_FILE,
)


def log_prediction(model: str, text: str, result: str, confidence: float):
    """Log a prediction event."""
    logger.info(
        f"Prediction: model={model}, result={result}, confidence={confidence:.3f}",
        extra={
            "extra_data": {
                "event": "prediction",
                "model": model,
                "text_length": len(text),
                "result": result,
                "confidence": confidence,
            }
        }
    )


def log_data_load(source: str, count: int, duration_ms: float):
    """Log a data loading event."""
    logger.info(
        f"Data loaded: source={source}, count={count}, duration={duration_ms:.1f}ms",
        extra={
            "extra_data": {
                "event": "data_load",
                "source": source,
                "record_count": count,
                "duration_ms": duration_ms,
            }
        }
    )


def log_model_training(model_name: str, accuracy: float, duration_ms: float):
    """Log a model training event."""
    logger.info(
        f"Model trained: {model_name}, accuracy={accuracy:.4f}, duration={duration_ms:.1f}ms",
        extra={
            "extra_data": {
                "event": "model_training",
                "model": model_name,
                "accuracy": accuracy,
                "duration_ms": duration_ms,
            }
        }
    )


def log_api_request(method: str, path: str, status_code: int, duration_ms: float):
    """Log an API request."""
    logger.info(
        f"API: {method} {path} -> {status_code} ({duration_ms:.1f}ms)",
        extra={
            "extra_data": {
                "event": "api_request",
                "method": method,
                "path": path,
                "status_code": status_code,
                "duration_ms": duration_ms,
            }
        }
    )


def log_error(error: Exception, context: str = ""):
    """Log an error with context."""
    logger.error(
        f"Error in {context}: {type(error).__name__}: {str(error)}",
        exc_info=True,
        extra={
            "extra_data": {
                "event": "error",
                "error_type": type(error).__name__,
                "error_message": str(error),
                "context": context,
            }
        }
    )

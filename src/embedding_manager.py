"""
Embedding Manager for Pre-trained Word Embeddings

This module handles loading, managing, and utilizing pre-trained word embeddings
(Word2Vec, GloVe, FastText) for deep learning models.
"""

import os
import logging
import numpy as np
import pickle
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import requests
from tqdm import tqdm
import zipfile
import gensim.downloader as api
from gensim.models import KeyedVectors

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Manages pre-trained word embeddings for NLP tasks.

    Supports:
    - Word2Vec (Google News, trained on ~100B words)
    - GloVe (Stanford, trained on various corpora)
    - FastText (Facebook, trained on Common Crawl)
    """

    # Available pre-trained embeddings via gensim
    AVAILABLE_EMBEDDINGS = {
        'word2vec-google-news-300': {
            'dim': 300,
            'source': 'gensim',
            'description': 'Google News Word2Vec (3M words, 300d)'
        },
        'glove-wiki-gigaword-100': {
            'dim': 100,
            'source': 'gensim',
            'description': 'GloVe Wikipedia + Gigaword (400K words, 100d)'
        },
        'glove-wiki-gigaword-200': {
            'dim': 200,
            'source': 'gensim',
            'description': 'GloVe Wikipedia + Gigaword (400K words, 200d)'
        },
        'glove-wiki-gigaword-300': {
            'dim': 300,
            'source': 'gensim',
            'description': 'GloVe Wikipedia + Gigaword (400K words, 300d)'
        },
        'fasttext-wiki-news-subwords-300': {
            'dim': 300,
            'source': 'gensim',
            'description': 'FastText Wikipedia + News (1M words, 300d)'
        },
        'glove-twitter-100': {
            'dim': 100,
            'source': 'gensim',
            'description': 'GloVe Twitter (1.2M words, 100d)'
        },
        'glove-twitter-200': {
            'dim': 200,
            'source': 'gensim',
            'description': 'GloVe Twitter (1.2M words, 200d)'
        }
    }

    def __init__(self, cache_dir: str = "embeddings_cache"):
        """
        Initialize the Embedding Manager.

        Args:
            cache_dir: Directory to cache downloaded embeddings
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)

        self.embeddings = None
        self.embedding_name = None
        self.embedding_dim = None
        self.vocab = None
        self.word2idx = None
        self.idx2word = None

        logger.info(f"EmbeddingManager initialized. Cache directory: {self.cache_dir}")

    def list_available_embeddings(self) -> Dict[str, dict]:
        """List all available pre-trained embeddings."""
        return self.AVAILABLE_EMBEDDINGS

    def load_embedding(self, embedding_name: str = 'glove-wiki-gigaword-100') -> None:
        """
        Load pre-trained word embeddings.

        Args:
            embedding_name: Name of the embedding to load
        """
        if embedding_name not in self.AVAILABLE_EMBEDDINGS:
            raise ValueError(
                f"Embedding '{embedding_name}' not available. "
                f"Choose from: {list(self.AVAILABLE_EMBEDDINGS.keys())}"
            )

        logger.info(f"Loading embedding: {embedding_name}")
        logger.info(f"Description: {self.AVAILABLE_EMBEDDINGS[embedding_name]['description']}")

        # Load from gensim
        try:
            self.embeddings = api.load(embedding_name)
            self.embedding_name = embedding_name
            self.embedding_dim = self.AVAILABLE_EMBEDDINGS[embedding_name]['dim']

            logger.info(f"Successfully loaded {embedding_name}")
            logger.info(f"Vocabulary size: {len(self.embeddings.key_to_index)}")
            logger.info(f"Embedding dimension: {self.embedding_dim}")

        except Exception as e:
            logger.error(f"Error loading embedding: {e}")
            raise

    def build_vocab(
        self,
        texts: List[str],
        max_vocab_size: Optional[int] = None,
        min_freq: int = 2
    ) -> Dict[str, int]:
        """
        Build vocabulary from texts, keeping only words that exist in embeddings.

        Args:
            texts: List of tokenized text strings
            max_vocab_size: Maximum vocabulary size (most frequent words)
            min_freq: Minimum word frequency to include

        Returns:
            word2idx: Dictionary mapping words to indices
        """
        if self.embeddings is None:
            raise ValueError("Load embeddings first using load_embedding()")

        logger.info("Building vocabulary from texts...")

        # Count word frequencies
        from collections import Counter
        word_freq = Counter()

        for text in tqdm(texts, desc="Counting words"):
            words = text.lower().split()
            word_freq.update(words)

        logger.info(f"Total unique words in corpus: {len(word_freq)}")

        # Filter by frequency and embedding coverage
        valid_words = []
        for word, freq in word_freq.items():
            if freq >= min_freq and word in self.embeddings:
                valid_words.append((word, freq))

        # Sort by frequency
        valid_words.sort(key=lambda x: x[1], reverse=True)

        # Limit vocabulary size
        if max_vocab_size:
            valid_words = valid_words[:max_vocab_size]

        logger.info(f"Vocabulary size after filtering: {len(valid_words)}")

        # Create word2idx mapping
        # Reserve indices: 0=PAD, 1=UNK
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        for idx, (word, _) in enumerate(valid_words, start=2):
            self.word2idx[word] = idx

        # Create reverse mapping
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab = set(self.word2idx.keys())

        logger.info(f"Final vocabulary size (including PAD, UNK): {len(self.word2idx)}")

        return self.word2idx

    def create_embedding_matrix(self) -> np.ndarray:
        """
        Create embedding matrix for the vocabulary.

        Returns:
            embedding_matrix: Shape (vocab_size, embedding_dim)
        """
        if self.word2idx is None:
            raise ValueError("Build vocabulary first using build_vocab()")

        logger.info("Creating embedding matrix...")

        vocab_size = len(self.word2idx)
        embedding_matrix = np.zeros((vocab_size, self.embedding_dim))

        # PAD token is all zeros (index 0)
        # UNK token is random or average of all embeddings
        embedding_matrix[1] = np.random.normal(
            scale=0.6,
            size=(self.embedding_dim,)
        )

        # Fill in embeddings for known words
        found = 0
        for word, idx in self.word2idx.items():
            if word in ['<PAD>', '<UNK>']:
                continue

            if word in self.embeddings:
                embedding_matrix[idx] = self.embeddings[word]
                found += 1

        coverage = (found / vocab_size) * 100
        logger.info(f"Embedding coverage: {found}/{vocab_size} ({coverage:.2f}%)")

        return embedding_matrix

    def texts_to_sequences(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        padding: str = 'post',
        truncating: str = 'post'
    ) -> np.ndarray:
        """
        Convert texts to sequences of word indices.

        Args:
            texts: List of text strings
            max_length: Maximum sequence length (None = no padding/truncation)
            padding: 'pre' or 'post' padding
            truncating: 'pre' or 'post' truncation

        Returns:
            sequences: Array of shape (num_texts, max_length)
        """
        if self.word2idx is None:
            raise ValueError("Build vocabulary first using build_vocab()")

        sequences = []

        for text in texts:
            words = text.lower().split()
            # Convert words to indices
            seq = [self.word2idx.get(word, 1) for word in words]  # 1 = UNK
            sequences.append(seq)

        # Pad/truncate if max_length specified
        if max_length is not None:
            padded_sequences = np.zeros((len(sequences), max_length), dtype=np.int32)

            for i, seq in enumerate(sequences):
                # Truncate
                if len(seq) > max_length:
                    if truncating == 'post':
                        seq = seq[:max_length]
                    else:  # 'pre'
                        seq = seq[-max_length:]

                # Pad
                if len(seq) < max_length:
                    pad_len = max_length - len(seq)
                    if padding == 'post':
                        padded_sequences[i, :len(seq)] = seq
                    else:  # 'pre'
                        padded_sequences[i, -len(seq):] = seq
                else:
                    padded_sequences[i] = seq

            return padded_sequences

        return sequences

    def get_word_vector(self, word: str) -> Optional[np.ndarray]:
        """
        Get embedding vector for a word.

        Args:
            word: Input word

        Returns:
            Vector if word exists in embeddings, None otherwise
        """
        if self.embeddings is None:
            raise ValueError("Load embeddings first using load_embedding()")

        word = word.lower()
        if word in self.embeddings:
            return self.embeddings[word]
        return None

    def most_similar(self, word: str, topn: int = 10) -> List[Tuple[str, float]]:
        """
        Find most similar words to the given word.

        Args:
            word: Input word
            topn: Number of similar words to return

        Returns:
            List of (word, similarity_score) tuples
        """
        if self.embeddings is None:
            raise ValueError("Load embeddings first using load_embedding()")

        if word.lower() in self.embeddings:
            return self.embeddings.most_similar(word.lower(), topn=topn)
        else:
            logger.warning(f"Word '{word}' not in vocabulary")
            return []

    def save_vocab(self, filepath: str) -> None:
        """Save vocabulary to file."""
        if self.word2idx is None:
            raise ValueError("No vocabulary to save")

        with open(filepath, 'wb') as f:
            pickle.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'vocab': self.vocab,
                'embedding_name': self.embedding_name,
                'embedding_dim': self.embedding_dim
            }, f)

        logger.info(f"Vocabulary saved to {filepath}")

    def load_vocab(self, filepath: str) -> None:
        """Load vocabulary from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.word2idx = data['word2idx']
        self.idx2word = data['idx2word']
        self.vocab = data['vocab']
        self.embedding_name = data['embedding_name']
        self.embedding_dim = data['embedding_dim']

        logger.info(f"Vocabulary loaded from {filepath}")
        logger.info(f"Vocabulary size: {len(self.word2idx)}")


def get_embedding_info() -> None:
    """Print information about available embeddings."""
    print("\n=== Available Pre-trained Word Embeddings ===\n")

    for name, info in EmbeddingManager.AVAILABLE_EMBEDDINGS.items():
        print(f"Name: {name}")
        print(f"  Description: {info['description']}")
        print(f"  Dimension: {info['dim']}")
        print(f"  Source: {info['source']}")
        print()


if __name__ == "__main__":
    # Demo usage
    get_embedding_info()

    # Example: Load GloVe embeddings
    em = EmbeddingManager()

    print("\n=== Loading GloVe 100d ===")
    em.load_embedding('glove-wiki-gigaword-100')

    # Test word similarity
    print("\n=== Testing Word Similarity ===")
    test_words = ['good', 'bad', 'excellent']
    for word in test_words:
        similar = em.most_similar(word, topn=5)
        print(f"\nWords similar to '{word}':")
        for sim_word, score in similar:
            print(f"  {sim_word}: {score:.3f}")

    # Test vocabulary building
    print("\n=== Building Vocabulary ===")
    sample_texts = [
        "this product is excellent and amazing",
        "terrible quality very disappointed",
        "good value for money would recommend"
    ]
    em.build_vocab(sample_texts, min_freq=1)

    # Test text to sequences
    print("\n=== Converting Text to Sequences ===")
    sequences = em.texts_to_sequences(sample_texts, max_length=10, padding='post')
    for i, (text, seq) in enumerate(zip(sample_texts, sequences)):
        print(f"Text: {text}")
        print(f"Sequence: {seq}")
        print()

"""
TF-IDF Vectorization wrapper for the Resume Screening System.

Provides a clean interface around scikit-learn's TfidfVectorizer
with save/load functionality for model persistence.
"""

import os
import logging
from typing import List, Optional, Union

import joblib
import numpy as np
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import get_config, Config

logger = logging.getLogger(__name__)


class ResumeVectorizer:
    """
    TF-IDF vectorizer for resume and job description text.

    Wraps sklearn's TfidfVectorizer with:
    - Configuration-driven parameter setup
    - Model persistence (save/load)
    - Convenient transform methods

    Attributes:
        vectorizer: The underlying sklearn TfidfVectorizer.
        is_fitted: Whether the vectorizer has been fitted.
    """

    SAVE_FILENAME = "tfidf_vectorizer.pkl"

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the vectorizer with configuration parameters.

        Args:
            config: Configuration object. Uses global config if None.
        """
        if config is None:
            config = get_config()

        self._config = config
        tfidf_cfg = config.tfidf

        self.vectorizer = TfidfVectorizer(
            max_features=tfidf_cfg.max_features,
            ngram_range=tfidf_cfg.ngram_range,
            min_df=tfidf_cfg.min_df,
            max_df=tfidf_cfg.max_df,
            sublinear_tf=tfidf_cfg.sublinear_tf,
            norm=tfidf_cfg.norm,
            stop_words="english",  # additional sklearn-level stopword removal
        )
        self.is_fitted = False

        logger.debug(
            f"ResumeVectorizer initialized: "
            f"max_features={tfidf_cfg.max_features}, "
            f"ngram_range={tfidf_cfg.ngram_range}"
        )

    def fit(self, corpus: List[str]) -> "ResumeVectorizer":
        """
        Fit the TF-IDF vectorizer on a corpus of texts.

        Args:
            corpus: List of preprocessed text strings.

        Returns:
            Self (for method chaining).
        """
        logger.info(f"Fitting TF-IDF vectorizer on {len(corpus)} documents...")
        self.vectorizer.fit(corpus)
        self.is_fitted = True

        vocab_size = len(self.vectorizer.vocabulary_)
        logger.info(f"TF-IDF fitted. Vocabulary size: {vocab_size}")

        return self

    def transform(self, texts: Union[str, List[str]]) -> spmatrix:
        """
        Transform text(s) into TF-IDF vectors.

        Args:
            texts: Single text string or list of text strings.

        Returns:
            Sparse matrix of TF-IDF features.

        Raises:
            RuntimeError: If vectorizer has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Vectorizer has not been fitted. Call fit() first or load a saved model."
            )

        if isinstance(texts, str):
            texts = [texts]

        return self.vectorizer.transform(texts)

    def fit_transform(self, corpus: List[str]) -> spmatrix:
        """
        Fit the vectorizer and transform the corpus in one step.

        Args:
            corpus: List of preprocessed text strings.

        Returns:
            Sparse matrix of TF-IDF features.
        """
        logger.info(f"Fitting and transforming {len(corpus)} documents...")
        self.is_fitted = True
        result = self.vectorizer.fit_transform(corpus)

        vocab_size = len(self.vectorizer.vocabulary_)
        logger.info(f"TF-IDF fit_transform complete. Vocabulary size: {vocab_size}")

        return result

    def get_feature_names(self) -> np.ndarray:
        """
        Get the feature names (vocabulary terms).

        Returns:
            Array of feature name strings.
        """
        if not self.is_fitted:
            raise RuntimeError("Vectorizer has not been fitted.")
        return self.vectorizer.get_feature_names_out()

    def save(self, model_dir: Optional[str] = None) -> str:
        """
        Save the fitted vectorizer to disk.

        Args:
            model_dir: Directory to save to. Uses config default if None.

        Returns:
            Path to the saved file.
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted vectorizer.")

        if model_dir is None:
            model_dir = self._config.paths.models_dir

        os.makedirs(model_dir, exist_ok=True)
        save_path = os.path.join(model_dir, self.SAVE_FILENAME)

        joblib.dump(self.vectorizer, save_path)
        logger.info(f"TF-IDF vectorizer saved to: {save_path}")

        return save_path

    def load(self, model_dir: Optional[str] = None) -> "ResumeVectorizer":
        """
        Load a fitted vectorizer from disk.

        Args:
            model_dir: Directory to load from. Uses config default if None.

        Returns:
            Self (for method chaining).
        """
        if model_dir is None:
            model_dir = self._config.paths.models_dir

        load_path = os.path.join(model_dir, self.SAVE_FILENAME)

        if not os.path.exists(load_path):
            raise FileNotFoundError(
                f"No saved vectorizer found at: {load_path}"
            )

        self.vectorizer = joblib.load(load_path)
        self.is_fitted = True
        logger.info(f"TF-IDF vectorizer loaded from: {load_path}")

        return self

"""
Text preprocessing pipeline for the Resume Screening System.

Handles:
- HTML tag stripping
- Text cleaning (URLs, emails, special characters)
- Tokenization (NLTK)
- Stopword removal
- Lemmatization (WordNet)
- Batch processing of DataFrames
"""

import re
import logging
from typing import List, Optional

import nltk
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from src.config import get_config, PreprocessingConfig

logger = logging.getLogger(__name__)


# ============================================================
# NLTK Resource Management
# ============================================================

def ensure_nltk_resources() -> None:
    """Download required NLTK resources if not already present."""
    resources = [
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            logger.info(f"Downloading NLTK resource: {name}")
            nltk.download(name, quiet=True)

    # Special check for wordnet which might be zipped
    try:
        nltk.data.find("corpora/wordnet.zip")
    except LookupError:
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            logger.info("Downloading NLTK resource: wordnet")
            nltk.download("wordnet", quiet=True)


# ============================================================
# Individual Preprocessing Steps
# ============================================================

def clean_html(text: str) -> str:
    """
    Remove HTML tags from text using BeautifulSoup.

    Args:
        text: Raw text potentially containing HTML markup.

    Returns:
        Plain text with all HTML tags removed.
    """
    if not text or not isinstance(text, str):
        return ""
    soup = BeautifulSoup(text, "lxml")
    return soup.get_text(separator=" ")


def clean_text(text: str) -> str:
    """
    Clean raw text by removing URLs, emails, phone numbers,
    special characters, and extra whitespace.

    Args:
        text: Raw text string.

    Returns:
        Cleaned text string.
    """
    if not text or not isinstance(text, str):
        return ""

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+\.\S+", " ", text)

    # Remove phone numbers (various formats)
    text = re.sub(r"[\+]?[\d\-\(\)\s]{7,15}", " ", text)

    # Remove special characters but keep alphanumeric, spaces, and common punctuation
    text = re.sub(r"[^a-zA-Z0-9\s\.\,\;\:\-\/\+\#]", " ", text)

    # Remove single characters (except 'a', 'i', 'c', 'r' which can be meaningful)
    text = re.sub(r"\b[^aicr\s]\b", " ", text)

    # Collapse multiple whitespace into single space
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_and_lemmatize(
    text: str,
    config: Optional[PreprocessingConfig] = None
) -> str:
    """
    Tokenize text, remove stopwords, and lemmatize.

    Args:
        text: Cleaned text string.
        config: Preprocessing configuration. Uses defaults if None.

    Returns:
        Preprocessed text with tokens joined by spaces.
    """
    if config is None:
        config = get_config().preprocessing

    if not text or not isinstance(text, str):
        return ""

    # Lowercase
    if config.lowercase:
        text = text.lower()

    # Tokenize
    tokens = word_tokenize(text)

    # Build stopwords set
    stop_words = set()
    if config.remove_stopwords:
        stop_words = set(stopwords.words("english"))
        stop_words.update(config.extra_stopwords)

    # Lemmatizer
    lemmatizer = WordNetLemmatizer() if config.lemmatize else None

    # Process tokens
    processed_tokens: List[str] = []
    for token in tokens:
        # Skip short tokens
        if len(token) < config.min_word_length:
            continue

        # Skip stopwords
        if token.lower() in stop_words:
            continue

        # Skip pure numbers
        if token.isdigit():
            continue

        # Lemmatize
        if lemmatizer:
            token = lemmatizer.lemmatize(token, pos="v")  # verb form
            token = lemmatizer.lemmatize(token, pos="n")  # noun form

        processed_tokens.append(token)

    return " ".join(processed_tokens)


# ============================================================
# Full Pipeline
# ============================================================

def preprocess_resume(
    text: str,
    config: Optional[PreprocessingConfig] = None
) -> str:
    """
    Full preprocessing pipeline for a single resume text.

    Steps:
    1. Strip HTML tags
    2. Clean text (remove URLs, emails, special chars)
    3. Tokenize, remove stopwords, lemmatize

    Args:
        text: Raw resume text (may contain HTML).
        config: Preprocessing configuration.

    Returns:
        Fully preprocessed text ready for vectorization.
    """
    if not text or not isinstance(text, str):
        return ""

    # Ensure NLTK resources are available
    ensure_nltk_resources()

    # Step 1: Strip HTML
    text = clean_html(text)

    # Step 2: Clean text
    text = clean_text(text)

    # Step 3: Tokenize and lemmatize
    text = tokenize_and_lemmatize(text, config)

    return text


def preprocess_dataframe(
    df: pd.DataFrame,
    text_column: str = "Resume_str",
    html_column: str = "Resume_html",
    output_column: str = "cleaned_text",
    config: Optional[PreprocessingConfig] = None
) -> pd.DataFrame:
    """
    Batch preprocess all resumes in a DataFrame.

    Uses Resume_str as primary text source. Falls back to Resume_html
    if Resume_str is empty/missing for a row.

    Args:
        df: DataFrame containing resume data.
        text_column: Column name for plain text resumes.
        html_column: Column name for HTML resumes (fallback).
        output_column: Column name for preprocessed output.
        config: Preprocessing configuration.

    Returns:
        DataFrame with added output_column containing preprocessed text.
    """
    if config is None:
        config = get_config().preprocessing

    logger.info(f"Preprocessing {len(df)} resumes...")

    # Ensure NLTK resources before batch processing
    ensure_nltk_resources()

    def _process_row(row):
        """Process a single row, using text_column or falling back to html_column."""
        text = ""
        if text_column in row and pd.notna(row[text_column]) and str(row[text_column]).strip():
            text = str(row[text_column])
        elif html_column in row and pd.notna(row[html_column]) and str(row[html_column]).strip():
            text = str(row[html_column])

        return preprocess_resume(text, config)

    df = df.copy()
    df[output_column] = df.apply(_process_row, axis=1)

    # Log stats
    empty_count = (df[output_column] == "").sum()
    if empty_count > 0:
        logger.warning(f"{empty_count} resumes produced empty text after preprocessing.")

    avg_length = df[output_column].str.split().str.len().mean()
    logger.info(
        f"Preprocessing complete. "
        f"Avg tokens per resume: {avg_length:.0f}, "
        f"Empty resumes: {empty_count}/{len(df)}"
    )

    return df

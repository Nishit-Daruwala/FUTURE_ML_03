"""
End-to-end training pipeline for the Resume Screening System.

Orchestrates the complete training workflow:
1. Load raw CSV data
2. Preprocess all resume texts
3. Fit TF-IDF vectorizer
4. Train resume category classifier
5. Evaluate on test set
6. Save all model artifacts
7. Generate training report

Usage:
    python -m pipeline.train_pipeline
"""

import os
import sys
import json
import time
import logging
from datetime import datetime

import pandas as pd
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config import load_config, setup_logging, ensure_directories
from src.preprocessing import preprocess_dataframe
from src.vectorizer import ResumeVectorizer
from src.classifier import ResumeClassifier

logger = logging.getLogger(__name__)


def load_dataset(config) -> pd.DataFrame:
    """
    Load the resume dataset from CSV.

    Args:
        config: Configuration object.

    Returns:
        DataFrame with resume data.

    Raises:
        FileNotFoundError: If dataset file is not found.
    """
    dataset_path = config.paths.dataset_path
    logger.info(f"Loading dataset from: {dataset_path}")

    if not os.path.exists(dataset_path):
        # Check for any CSV file in the raw data directory
        raw_dir = config.paths.raw_data_dir
        csv_files = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]
        if csv_files:
            dataset_path = os.path.join(raw_dir, csv_files[0])
            logger.info(f"Using found CSV file: {dataset_path}")
        else:
            raise FileNotFoundError(
                f"Dataset not found at: {dataset_path}\n"
                f"Please download the Kaggle Resume Dataset and place the CSV file in: {raw_dir}\n"
                f"Download from: https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset"
            )

    df = pd.read_csv(dataset_path)
    logger.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    logger.info(f"Columns: {list(df.columns)}")

    # Validate required columns
    required_cols = {"Category"}
    text_cols = {"Resume_str", "Resume_html"}

    if not required_cols.issubset(set(df.columns)):
        raise ValueError(
            f"Dataset missing required columns: {required_cols - set(df.columns)}"
        )

    if not text_cols.intersection(set(df.columns)):
        raise ValueError(
            f"Dataset must contain at least one of: {text_cols}"
        )

    # Log category distribution
    logger.info(f"\nCategory distribution:\n{df['Category'].value_counts().to_string()}")

    return df


def run_training_pipeline():
    """Execute the full training pipeline."""
    start_time = time.time()

    # ── Step 0: Configuration ──────────────────────────────
    config = load_config()
    setup_logging(config)
    ensure_directories(config)

    logger.info("=" * 60)
    logger.info("  RESUME SCREENING SYSTEM — TRAINING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Started at: {datetime.now().isoformat()}")

    # Set random seed for reproducibility
    np.random.seed(config.general.random_seed)

    # ── Step 1: Load Dataset ───────────────────────────────
    logger.info("\n── Step 1: Loading Dataset ──")
    df = load_dataset(config)

    # ── Step 2: Preprocess ─────────────────────────────────
    logger.info("\n── Step 2: Preprocessing Resumes ──")
    df = preprocess_dataframe(df, config=config.preprocessing)

    # Remove rows with empty cleaned text
    original_count = len(df)
    df = df[df["cleaned_text"].str.strip().astype(bool)].reset_index(drop=True)
    removed = original_count - len(df)
    if removed > 0:
        logger.warning(f"Removed {removed} rows with empty text after preprocessing.")

    # Save preprocessed data
    processed_path = os.path.join(config.paths.processed_data_dir, "preprocessed_resumes.csv")
    df.to_csv(processed_path, index=False)
    logger.info(f"Preprocessed data saved to: {processed_path}")

    # ── Step 3: TF-IDF Vectorization ──────────────────────
    logger.info("\n── Step 3: Fitting TF-IDF Vectorizer ──")
    vectorizer = ResumeVectorizer(config)
    X_all = vectorizer.fit_transform(df["cleaned_text"].tolist())
    logger.info(f"TF-IDF matrix shape: {X_all.shape}")

    # ── Step 4: Train/Test Split ──────────────────────────
    logger.info("\n── Step 4: Train/Test Split ──")
    classifier = ResumeClassifier(config)
    y_all = df["Category"].tolist()

    X_train, X_test, y_train, y_test = classifier.split_data(X_all, y_all)
    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Test set:     {X_test.shape[0]} samples")

    # ── Step 5: Train Classifier ──────────────────────────
    logger.info("\n── Step 5: Training Classifier ──")
    train_metrics = classifier.train(X_train, y_train, cross_validate=True)

    # ── Step 6: Evaluate ──────────────────────────────────
    logger.info("\n── Step 6: Evaluating on Test Set ──")
    eval_metrics = classifier.evaluate(X_test, y_test)

    # ── Step 7: Save Artifacts ────────────────────────────
    logger.info("\n── Step 7: Saving Model Artifacts ──")
    vectorizer.save()
    classifier.save()

    # ── Step 8: Generate Training Report ──────────────────
    logger.info("\n── Step 8: Generating Training Report ──")
    elapsed = time.time() - start_time

    report = {
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 2),
        "dataset": {
            "total_samples": len(df),
            "train_samples": X_train.shape[0],
            "test_samples": X_test.shape[0],
            "n_categories": len(set(y_all)),
            "categories": sorted(set(y_all)),
        },
        "tfidf": {
            "vocabulary_size": X_all.shape[1],
            "max_features": config.tfidf.max_features,
            "ngram_range": list(config.tfidf.ngram_range),
        },
        "training_metrics": train_metrics,
        "evaluation_metrics": {
            "test_accuracy": eval_metrics["accuracy"],
            "classification_report": eval_metrics["classification_report"],
        },
    }

    # Save report
    report_path = os.path.join(config.paths.outputs_dir, "training_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Training report saved to: {report_path}")

    # Save classification report text
    clf_report_path = os.path.join(config.paths.outputs_dir, "classification_report.txt")
    with open(clf_report_path, "w", encoding="utf-8") as f:
        f.write("RESUME CATEGORY CLASSIFICATION REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Classifier: {train_metrics['classifier_type']}\n")
        f.write(f"Test Accuracy: {eval_metrics['accuracy']:.4f}\n\n")
        f.write(eval_metrics["classification_report_text"])
    logger.info(f"Classification report saved to: {clf_report_path}")

    # ── Summary ───────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Classifier: {train_metrics['classifier_type']}")
    logger.info(f"  Training Accuracy: {train_metrics['train_accuracy']:.4f}")
    if "cv_mean_accuracy" in train_metrics:
        logger.info(
            f"  CV Accuracy: {train_metrics['cv_mean_accuracy']:.4f} "
            f"(±{train_metrics['cv_std_accuracy']:.4f})"
        )
    logger.info(f"  Test Accuracy: {eval_metrics['accuracy']:.4f}")
    logger.info(f"  Time elapsed: {elapsed:.1f}s")
    logger.info(f"  Models saved to: {config.paths.models_dir}/")
    logger.info(f"  Reports saved to: {config.paths.outputs_dir}/")
    logger.info("=" * 60)

    return report


if __name__ == "__main__":
    try:
        run_training_pipeline()
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Training pipeline failed: {e}")
        sys.exit(1)

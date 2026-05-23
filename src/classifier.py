"""
Resume category classifier for the Resume Screening System.

Supports multiple classifier types (Naive Bayes, Linear SVC, Logistic Regression)
with train/predict/evaluate/save/load lifecycle.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
from scipy.sparse import spmatrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from src.config import get_config, Config

logger = logging.getLogger(__name__)


class ResumeClassifier:
    """
    Classifies resumes into job role categories.

    Supports Multinomial Naive Bayes, Linear SVC, and Logistic Regression.
    Includes automatic label encoding, cross-validation, and model persistence.

    Attributes:
        classifier: The underlying sklearn classifier.
        label_encoder: LabelEncoder for category string ↔ int mapping.
        classifier_type: String identifier of the classifier type.
        is_trained: Whether the classifier has been trained.
    """

    CLASSIFIER_FILENAME = "category_classifier.pkl"
    ENCODER_FILENAME = "label_encoder.pkl"

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the classifier based on configuration.

        Args:
            config: Configuration object. Uses global config if None.
        """
        if config is None:
            config = get_config()

        self._config = config
        clf_cfg = config.classifier
        self.classifier_type = clf_cfg.type

        # Create classifier based on config type
        self.classifier = self._create_classifier(clf_cfg)
        self.label_encoder = LabelEncoder()
        self.is_trained = False

        logger.debug(f"ResumeClassifier initialized: type={self.classifier_type}")

    def _create_classifier(self, clf_cfg):
        """Create the sklearn classifier instance based on config type."""
        if clf_cfg.type == "multinomial_nb":
            return MultinomialNB()
        elif clf_cfg.type == "linear_svc":
            return LinearSVC(
                C=clf_cfg.svc_C,
                max_iter=clf_cfg.svc_max_iter,
                random_state=clf_cfg.random_state,
            )
        elif clf_cfg.type == "logistic_regression":
            return LogisticRegression(
                max_iter=clf_cfg.svc_max_iter,
                random_state=clf_cfg.random_state,
            )
        else:
            logger.warning(
                f"Unknown classifier type '{clf_cfg.type}'. "
                f"Defaulting to MultinomialNB."
            )
            return MultinomialNB()

    def train(
        self,
        X: spmatrix,
        y: List[str],
        cross_validate: bool = True,
        cv_folds: int = 5,
    ) -> Dict:
        """
        Train the classifier on TF-IDF features and category labels.

        Args:
            X: TF-IDF feature matrix (sparse).
            y: List of category label strings.
            cross_validate: Whether to run cross-validation.
            cv_folds: Number of cross-validation folds.

        Returns:
            Dictionary with training metrics.
        """
        logger.info(
            f"Training {self.classifier_type} classifier on "
            f"{X.shape[0]} samples, {X.shape[1]} features..."
        )

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        n_classes = len(self.label_encoder.classes_)
        logger.info(f"Number of categories: {n_classes}")
        logger.info(f"Categories: {list(self.label_encoder.classes_)}")

        # Cross-validation (optional)
        cv_scores = None
        if cross_validate:
            logger.info(f"Running {cv_folds}-fold cross-validation...")
            cv_scores = cross_val_score(
                self.classifier, X, y_encoded, cv=cv_folds, scoring="accuracy"
            )
            logger.info(
                f"CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})"
            )

        # Train on full data
        self.classifier.fit(X, y_encoded)
        self.is_trained = True

        # Training accuracy
        train_pred = self.classifier.predict(X)
        train_accuracy = accuracy_score(y_encoded, train_pred)
        logger.info(f"Training accuracy: {train_accuracy:.4f}")

        metrics = {
            "classifier_type": self.classifier_type,
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "n_classes": n_classes,
            "classes": list(self.label_encoder.classes_),
            "train_accuracy": float(train_accuracy),
        }

        if cv_scores is not None:
            metrics["cv_mean_accuracy"] = float(cv_scores.mean())
            metrics["cv_std_accuracy"] = float(cv_scores.std())
            metrics["cv_scores"] = [float(s) for s in cv_scores]

        return metrics

    def predict(self, X: spmatrix) -> List[str]:
        """
        Predict resume categories.

        Args:
            X: TF-IDF feature matrix.

        Returns:
            List of predicted category strings.

        Raises:
            RuntimeError: If classifier has not been trained.
        """
        if not self.is_trained:
            raise RuntimeError(
                "Classifier has not been trained. Call train() first or load a saved model."
            )

        y_encoded = self.classifier.predict(X)
        return list(self.label_encoder.inverse_transform(y_encoded))

    def predict_proba(self, X: spmatrix) -> Optional[np.ndarray]:
        """
        Get prediction probabilities (only for classifiers that support it).

        Args:
            X: TF-IDF feature matrix.

        Returns:
            Probability matrix or None if not supported.
        """
        if not self.is_trained:
            raise RuntimeError("Classifier has not been trained.")

        if hasattr(self.classifier, "predict_proba"):
            return self.classifier.predict_proba(X)
        elif hasattr(self.classifier, "decision_function"):
            # For SVC, return decision function scores (not true probabilities)
            return self.classifier.decision_function(X)
        return None

    def evaluate(
        self,
        X_test: spmatrix,
        y_test: List[str],
    ) -> Dict:
        """
        Evaluate the classifier on test data.

        Args:
            X_test: Test TF-IDF feature matrix.
            y_test: True category labels.

        Returns:
            Dictionary with evaluation metrics and classification report.
        """
        if not self.is_trained:
            raise RuntimeError("Classifier has not been trained.")

        y_test_encoded = self.label_encoder.transform(y_test)
        y_pred_encoded = self.classifier.predict(X_test)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)

        accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
        report = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )
        report_str = classification_report(
            y_test, y_pred, zero_division=0
        )
        conf_matrix = confusion_matrix(
            y_test_encoded, y_pred_encoded
        ).tolist()

        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info(f"\n{report_str}")

        return {
            "accuracy": float(accuracy),
            "classification_report": report,
            "classification_report_text": report_str,
            "confusion_matrix": conf_matrix,
        }

    def split_data(
        self,
        X: spmatrix,
        y: List[str],
    ) -> Tuple[spmatrix, spmatrix, List[str], List[str]]:
        """
        Split data into train and test sets (stratified).

        Args:
            X: Feature matrix.
            y: Labels.

        Returns:
            Tuple of (X_train, X_test, y_train, y_test).
        """
        clf_cfg = self._config.classifier
        return train_test_split(
            X, y,
            test_size=clf_cfg.test_size,
            random_state=clf_cfg.random_state,
            stratify=y,
        )

    def save(self, model_dir: Optional[str] = None) -> None:
        """Save the trained classifier and label encoder to disk."""
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained classifier.")

        if model_dir is None:
            model_dir = self._config.paths.models_dir

        os.makedirs(model_dir, exist_ok=True)

        clf_path = os.path.join(model_dir, self.CLASSIFIER_FILENAME)
        enc_path = os.path.join(model_dir, self.ENCODER_FILENAME)

        joblib.dump(self.classifier, clf_path)
        joblib.dump(self.label_encoder, enc_path)

        logger.info(f"Classifier saved to: {clf_path}")
        logger.info(f"Label encoder saved to: {enc_path}")

    def load(self, model_dir: Optional[str] = None) -> "ResumeClassifier":
        """Load a trained classifier and label encoder from disk."""
        if model_dir is None:
            model_dir = self._config.paths.models_dir

        clf_path = os.path.join(model_dir, self.CLASSIFIER_FILENAME)
        enc_path = os.path.join(model_dir, self.ENCODER_FILENAME)

        if not os.path.exists(clf_path):
            raise FileNotFoundError(f"Classifier not found: {clf_path}")
        if not os.path.exists(enc_path):
            raise FileNotFoundError(f"Label encoder not found: {enc_path}")

        self.classifier = joblib.load(clf_path)
        self.label_encoder = joblib.load(enc_path)
        self.is_trained = True

        logger.info(f"Classifier loaded from: {clf_path}")
        return self

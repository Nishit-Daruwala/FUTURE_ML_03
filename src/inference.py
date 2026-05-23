"""
Unified inference API for the Resume Screening System.

Provides a single entry point (ResumeScreener) that loads all trained
model artifacts and exposes clean methods for scoring and ranking resumes.
Designed for easy integration with a future Streamlit UI.
"""

import os
import logging
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from src.config import load_config, setup_logging, Config
from src.preprocessing import preprocess_resume
from src.skill_extractor import SkillExtractor
from src.vectorizer import ResumeVectorizer
from src.classifier import ResumeClassifier
from src.scorer import ResumeScorer
from src.ranker import CandidateRanker

logger = logging.getLogger(__name__)


class ResumeScreener:
    """
    Main inference API for resume screening and ranking.

    Loads all trained model artifacts and provides methods to:
    - Screen a batch of resumes against a job description
    - Score a single resume
    - Get detailed skill analysis

    This class is designed to be the single integration point
    for a future Streamlit UI.

    Usage:
        screener = ResumeScreener(model_dir="models")
        results = screener.screen_resumes(
            job_description="We need a Python developer with ML experience...",
            resumes=["Resume text 1...", "Resume text 2..."],
            resume_ids=["candidate_1", "candidate_2"],
        )
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        config: Optional[Config] = None,
    ):
        """
        Initialize the screener by loading all model artifacts.

        Args:
            model_dir: Directory containing saved model artifacts.
            config: Configuration object. Auto-loads if None.
        """
        if config is None:
            config = load_config()

        self._config = config

        if model_dir is None:
            model_dir = config.paths.models_dir

        self._model_dir = model_dir

        # Initialize components
        logger.info("Initializing ResumeScreener...")

        # Skill extractor (doesn't need saved models, uses skill files)
        self.skill_extractor = SkillExtractor(config)

        # Load trained vectorizer
        self.vectorizer = ResumeVectorizer(config)
        self.vectorizer.load(model_dir)

        # Load trained classifier
        self.classifier = ResumeClassifier(config)
        self.classifier.load(model_dir)

        # Initialize scorer and ranker
        self.scorer = ResumeScorer(config)
        self.ranker = CandidateRanker()

        logger.info("ResumeScreener initialized successfully.")

    def screen_resumes(
        self,
        job_description: str,
        resumes: List[str],
        resume_ids: Optional[List[str]] = None,
        target_category: str = "",
        top_n: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Screen and rank a batch of resumes against a job description.

        This is the main API method — takes raw text and returns
        a complete ranked analysis.

        Args:
            job_description: Raw job description text.
            resumes: List of raw resume text strings.
            resume_ids: Optional identifiers for each resume.
            target_category: Expected job category/role (e.g., "Data Science").
            top_n: If set, return only the top N candidates.

        Returns:
            Dictionary containing:
                - 'ranked_candidates': Sorted list of candidate results
                - 'job_description_skills': Skills extracted from JD
                - 'ranking_report': Human-readable report string
                - 'common_skill_gaps': Most commonly missing skills
        """
        if resume_ids is None:
            resume_ids = [f"Candidate_{i+1}" for i in range(len(resumes))]

        if len(resumes) != len(resume_ids):
            raise ValueError(
                f"Number of resumes ({len(resumes)}) must match "
                f"number of IDs ({len(resume_ids)})"
            )

        logger.info(
            f"Screening {len(resumes)} resumes against job description..."
        )

        # 1. Preprocess job description
        jd_cleaned = preprocess_resume(job_description)

        # 2. Extract JD skills
        jd_skills = self.skill_extractor.extract_skills_flat(job_description)
        logger.info(f"JD skills found: {len(jd_skills)} — {sorted(jd_skills)}")

        # 3. Vectorize JD
        jd_vector = self.vectorizer.transform(jd_cleaned)

        # 4. Process each resume
        candidates = []
        for idx, (resume_text, resume_id) in enumerate(zip(resumes, resume_ids)):
            try:
                result = self._score_single_resume(
                    resume_text=resume_text,
                    resume_id=resume_id,
                    jd_vector=jd_vector,
                    jd_skills=jd_skills,
                    target_category=target_category,
                )
                candidates.append(result)
            except Exception as e:
                logger.error(f"Error scoring resume {resume_id}: {e}")
                candidates.append({
                    "candidate_id": resume_id,
                    "overall_score": 0.0,
                    "error": str(e),
                })

        # 5. Rank candidates
        ranked = self.ranker.rank_candidates(candidates, top_n=top_n)

        # 6. Generate report
        report = self.ranker.generate_ranking_report(
            ranked_candidates=ranked,
            job_description_summary=job_description[:200],
            jd_skills=sorted(jd_skills),
        )

        # 7. Common skill gaps
        skill_gaps = self.ranker.identify_common_skill_gaps(ranked)

        return {
            "ranked_candidates": ranked,
            "job_description_skills": sorted(jd_skills),
            "ranking_report": report,
            "common_skill_gaps": skill_gaps,
            "total_screened": len(resumes),
        }

    def screen_single(
        self,
        job_description: str,
        resume_text: str,
        target_category: str = "",
    ) -> Dict[str, Any]:
        """
        Score a single resume against a job description.

        Convenience method for one-at-a-time screening.

        Args:
            job_description: Raw job description text.
            resume_text: Raw resume text.
            target_category: Expected job category/role.

        Returns:
            Scoring result dictionary.
        """
        result = self.screen_resumes(
            job_description=job_description,
            resumes=[resume_text],
            resume_ids=["Candidate"],
            target_category=target_category,
        )
        return result["ranked_candidates"][0]

    def _score_single_resume(
        self,
        resume_text: str,
        resume_id: str,
        jd_vector,
        jd_skills: Set[str],
        target_category: str = "",
    ) -> Dict[str, Any]:
        """
        Internal method to score a single resume.

        Args:
            resume_text: Raw resume text.
            resume_id: Identifier for the resume.
            jd_vector: Pre-computed TF-IDF vector of the JD.
            jd_skills: Pre-extracted skills from the JD.
            target_category: Target job category.

        Returns:
            Complete scoring dictionary for this resume.
        """
        # Preprocess
        cleaned = preprocess_resume(resume_text)

        # Extract skills
        resume_skills = self.skill_extractor.extract_skills_flat(resume_text)

        # Vectorize
        resume_vector = self.vectorizer.transform(cleaned)

        # Predict category
        predicted_category = self.classifier.predict(resume_vector)[0]

        # Score
        scores = self.scorer.score_resume(
            resume_vector=resume_vector,
            jd_vector=jd_vector,
            resume_skills=resume_skills,
            jd_skills=jd_skills,
            resume_category=predicted_category,
            jd_category=target_category,
        )

        # Build result
        result = {
            "candidate_id": resume_id,
            "category": predicted_category,
            **scores,
        }

        return result

    def get_skill_analysis(self, text: str) -> Dict:
        """
        Get detailed skill analysis for a piece of text.

        Useful for analyzing either a resume or a job description.

        Args:
            text: Input text.

        Returns:
            Skill summary dictionary.
        """
        return self.skill_extractor.get_skill_summary(text)


# ============================================================
# CLI entry point for testing
# ============================================================

def main():
    """Quick test of the inference pipeline."""
    config = load_config()
    setup_logging(config)

    # Check if models exist
    model_dir = config.paths.models_dir
    if not os.path.exists(os.path.join(model_dir, "tfidf_vectorizer.pkl")):
        logger.error(
            "No trained models found. Run the training pipeline first:\n"
            "  python -m pipeline.train_pipeline"
        )
        return

    # Initialize screener
    screener = ResumeScreener(config=config)

    # Sample job description for testing
    sample_jd = """
    We are looking for a Data Scientist with strong Python programming skills.
    The ideal candidate should have experience with machine learning, deep learning,
    natural language processing, and data analysis. Required skills include
    Python, SQL, TensorFlow or PyTorch, pandas, and scikit-learn.
    Experience with cloud platforms (AWS or GCP) is a plus.
    Strong communication skills and ability to work in a team are essential.
    """

    sample_resume = """
    Experienced Data Scientist with 3 years of expertise in machine learning
    and deep learning. Proficient in Python, TensorFlow, pandas, and SQL.
    Built NLP pipelines for text classification and sentiment analysis.
    Experience with AWS cloud services. Strong communication and teamwork skills.
    """

    result = screener.screen_single(
        job_description=sample_jd,
        resume_text=sample_resume,
        target_category="Data Science",
    )

    print("\n--- Single Resume Screening Result ---")
    for key, value in result.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

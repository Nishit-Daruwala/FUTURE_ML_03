"""
Scoring module for the Resume Screening System.

Computes cosine similarity between resume and job description vectors,
skill match percentages, and composite relevance scores.
"""

import logging
from typing import Dict, Optional, Set

import numpy as np
from scipy.sparse import spmatrix
from sklearn.metrics.pairwise import cosine_similarity

from src.config import get_config, Config

logger = logging.getLogger(__name__)


class ResumeScorer:
    """
    Computes relevance scores for resumes against job descriptions.

    Combines:
    1. TF-IDF cosine similarity (semantic relevance)
    2. Skill match percentage (explicit skill overlap)
    3. Category match bonus (if resume category matches JD role)

    The composite score formula:
        score = w1 * cosine_sim + w2 * skill_match + category_bonus

    Attributes:
        similarity_weight: Weight for cosine similarity component.
        skill_match_weight: Weight for skill match component.
        category_match_bonus: Bonus added when category matches.
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the scorer with configurable weights.

        Args:
            config: Configuration object. Uses global config if None.
        """
        if config is None:
            config = get_config()

        scoring_cfg = config.scoring
        self.similarity_weight = scoring_cfg.similarity_weight
        self.skill_match_weight = scoring_cfg.skill_match_weight
        self.category_match_bonus = scoring_cfg.category_match_bonus

        logger.debug(
            f"ResumeScorer initialized: "
            f"sim_weight={self.similarity_weight}, "
            f"skill_weight={self.skill_match_weight}, "
            f"category_bonus={self.category_match_bonus}"
        )

    def compute_cosine_similarity(
        self,
        resume_vector: spmatrix,
        jd_vector: spmatrix,
    ) -> float:
        """
        Compute cosine similarity between a resume and job description vector.

        Args:
            resume_vector: TF-IDF vector for the resume (1 x n_features).
            jd_vector: TF-IDF vector for the job description (1 x n_features).

        Returns:
            Cosine similarity score (0.0 to 1.0).
        """
        sim = cosine_similarity(resume_vector, jd_vector)[0][0]
        return float(max(0.0, min(1.0, sim)))

    def compute_batch_cosine_similarity(
        self,
        resume_vectors: spmatrix,
        jd_vector: spmatrix,
    ) -> np.ndarray:
        """
        Compute cosine similarity between multiple resumes and a single JD.

        Args:
            resume_vectors: TF-IDF matrix for resumes (n_resumes x n_features).
            jd_vector: TF-IDF vector for the JD (1 x n_features).

        Returns:
            Array of cosine similarity scores, shape (n_resumes,).
        """
        similarities = cosine_similarity(resume_vectors, jd_vector).flatten()
        return np.clip(similarities, 0.0, 1.0)

    def compute_skill_match_score(
        self,
        resume_skills: Set[str],
        jd_skills: Set[str],
    ) -> Dict:
        """
        Compute skill match between resume and job description skills.

        Args:
            resume_skills: Set of skills found in the resume.
            jd_skills: Set of skills required by the JD.

        Returns:
            Dictionary with match_score, matched_skills, and missing_skills.
        """
        if not jd_skills:
            return {
                "match_score": 1.0,
                "matched_skills": sorted(resume_skills),
                "missing_skills": [],
                "extra_skills": sorted(resume_skills),
            }

        matched = resume_skills & jd_skills
        missing = jd_skills - resume_skills
        extra = resume_skills - jd_skills
        match_score = len(matched) / len(jd_skills)

        return {
            "match_score": float(match_score),
            "matched_skills": sorted(matched),
            "missing_skills": sorted(missing),
            "extra_skills": sorted(extra),
        }

    def compute_composite_score(
        self,
        cosine_sim: float,
        skill_match: float,
        category_matches: bool = False,
    ) -> float:
        """
        Compute the weighted composite relevance score.

        Formula:
            score = w1 * cosine_sim + w2 * skill_match + category_bonus

        Args:
            cosine_sim: Cosine similarity score (0.0 to 1.0).
            skill_match: Skill match percentage (0.0 to 1.0).
            category_matches: Whether the resume category matches the JD role.

        Returns:
            Composite score (0.0 to 1.0+bonus).
        """
        score = (
            self.similarity_weight * cosine_sim
            + self.skill_match_weight * skill_match
        )

        if category_matches:
            score += self.category_match_bonus

        return float(min(1.0, score))

    def score_resume(
        self,
        resume_vector: spmatrix,
        jd_vector: spmatrix,
        resume_skills: Set[str],
        jd_skills: Set[str],
        resume_category: str = "",
        jd_category: str = "",
    ) -> Dict:
        """
        Compute all scores for a single resume against a job description.

        Args:
            resume_vector: TF-IDF vector for the resume.
            jd_vector: TF-IDF vector for the JD.
            resume_skills: Skills extracted from the resume.
            jd_skills: Skills extracted from the JD.
            resume_category: Predicted category of the resume.
            jd_category: Target category/role of the JD.

        Returns:
            Comprehensive scoring dictionary.
        """
        # Cosine similarity
        cosine_sim = self.compute_cosine_similarity(resume_vector, jd_vector)

        # Skill match
        skill_result = self.compute_skill_match_score(resume_skills, jd_skills)

        # Category match
        category_matches = (
            bool(resume_category)
            and bool(jd_category)
            and resume_category.lower() == jd_category.lower()
        )

        # Composite score
        composite = self.compute_composite_score(
            cosine_sim,
            skill_result["match_score"],
            category_matches,
        )

        return {
            "overall_score": composite,
            "cosine_similarity": cosine_sim,
            "skill_match_score": skill_result["match_score"],
            "category_match": category_matches,
            "matched_skills": skill_result["matched_skills"],
            "missing_skills": skill_result["missing_skills"],
            "extra_skills": skill_result["extra_skills"],
        }

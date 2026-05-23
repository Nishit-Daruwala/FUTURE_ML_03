"""
Skill extraction module for the Resume Screening System.

Uses spaCy PhraseMatcher to efficiently identify technical and soft skills
from resume text and job descriptions. Skills are matched against curated
skill databases loaded from text files.
"""

import os
import logging
from typing import Dict, List, Optional, Set, Tuple

import spacy
from spacy.matcher import PhraseMatcher

from src.config import get_config, Config

logger = logging.getLogger(__name__)


class SkillExtractor:
    """
    Extracts skills from text using spaCy PhraseMatcher.

    Loads curated skill lists from text files and uses pattern matching
    to identify skills in resume and job description text. Supports both
    technical and soft skills with separate tracking.

    Attributes:
        nlp: spaCy language model.
        technical_skills: Set of known technical skill strings.
        soft_skills: Set of known soft skill strings.
        matcher: spaCy PhraseMatcher configured with skill patterns.
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the SkillExtractor.

        Args:
            config: Configuration object. Uses global config if None.
        """
        if config is None:
            config = get_config()

        self._config = config
        self.technical_skills: Set[str] = set()
        self.soft_skills: Set[str] = set()
        self.all_skills: Set[str] = set()

        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        except OSError:
            logger.error(
                "spaCy model 'en_core_web_sm' not found. "
                "Install it with: python -m spacy download en_core_web_sm"
            )
            raise

        # Load skill databases
        self._load_skills()

        # Build PhraseMatcher
        self.matcher = self._build_matcher()

        logger.info(
            f"SkillExtractor initialized with "
            f"{len(self.technical_skills)} technical skills and "
            f"{len(self.soft_skills)} soft skills."
        )

    def _load_skills(self) -> None:
        """Load technical and soft skills from text files."""
        skills_dir = self._config.paths.skills_dir

        # Load technical skills
        tech_path = os.path.join(
            skills_dir, self._config.skills.technical_skills_file
        )
        self.technical_skills = self._load_skill_file(tech_path)

        # Load soft skills
        soft_path = os.path.join(
            skills_dir, self._config.skills.soft_skills_file
        )
        self.soft_skills = self._load_skill_file(soft_path)

        # Combined set
        self.all_skills = self.technical_skills | self.soft_skills

    def _load_skill_file(self, filepath: str) -> Set[str]:
        """
        Load skills from a text file (one skill per line).

        Args:
            filepath: Path to the skill list file.

        Returns:
            Set of skill strings (lowercased).
        """
        skills = set()
        if not os.path.exists(filepath):
            logger.warning(f"Skill file not found: {filepath}")
            return skills

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                skill = line.strip()
                if skill and not skill.startswith("#"):
                    skills.add(skill.lower())

        logger.debug(f"Loaded {len(skills)} skills from {filepath}")
        return skills

    def _build_matcher(self) -> PhraseMatcher:
        """
        Build a spaCy PhraseMatcher with all skill patterns.

        Returns:
            Configured PhraseMatcher instance.
        """
        matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")

        # Add technical skills
        if self.technical_skills:
            tech_patterns = list(
                self.nlp.pipe(self.technical_skills)
            )
            matcher.add("TECHNICAL_SKILL", tech_patterns)

        # Add soft skills
        if self.soft_skills:
            soft_patterns = list(
                self.nlp.pipe(self.soft_skills)
            )
            matcher.add("SOFT_SKILL", soft_patterns)

        return matcher

    def extract_skills(self, text: str) -> Dict[str, Set[str]]:
        """
        Extract all skills from a text string.

        Args:
            text: Input text (resume or job description).

        Returns:
            Dictionary with keys 'technical', 'soft', and 'all',
            each containing a set of matched skill strings.
        """
        if not text or not isinstance(text, str):
            return {"technical": set(), "soft": set(), "all": set()}

        doc = self.nlp(text.lower())
        matches = self.matcher(doc)

        technical_found: Set[str] = set()
        soft_found: Set[str] = set()

        for match_id, start, end in matches:
            span_text = doc[start:end].text.lower()
            rule_id = self.nlp.vocab.strings[match_id]

            if rule_id == "TECHNICAL_SKILL":
                technical_found.add(span_text)
            elif rule_id == "SOFT_SKILL":
                soft_found.add(span_text)

        return {
            "technical": technical_found,
            "soft": soft_found,
            "all": technical_found | soft_found,
        }

    def extract_skills_flat(self, text: str) -> Set[str]:
        """
        Extract all skills as a flat set (convenience method).

        Args:
            text: Input text.

        Returns:
            Set of all matched skill strings.
        """
        result = self.extract_skills(text)
        return result["all"]

    def compute_skill_match(
        self,
        resume_skills: Set[str],
        jd_skills: Set[str]
    ) -> Tuple[float, Set[str], Set[str]]:
        """
        Compute skill match percentage between resume and job description.

        Args:
            resume_skills: Skills found in the resume.
            jd_skills: Skills required by the job description.

        Returns:
            Tuple of (match_percentage, matched_skills, missing_skills).
            match_percentage is 0.0 to 1.0.
        """
        if not jd_skills:
            return 1.0, resume_skills, set()

        matched = resume_skills & jd_skills
        missing = jd_skills - resume_skills
        match_pct = len(matched) / len(jd_skills) if jd_skills else 0.0

        return match_pct, matched, missing

    def get_skill_summary(
        self,
        text: str
    ) -> Dict:
        """
        Generate a comprehensive skill summary for a text.

        Args:
            text: Input text.

        Returns:
            Dictionary with skill counts and lists.
        """
        skills = self.extract_skills(text)
        return {
            "total_skills": len(skills["all"]),
            "technical_count": len(skills["technical"]),
            "soft_count": len(skills["soft"]),
            "technical_skills": sorted(skills["technical"]),
            "soft_skills": sorted(skills["soft"]),
        }

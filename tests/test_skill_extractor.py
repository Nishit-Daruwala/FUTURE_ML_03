"""
Unit tests for the skill extraction module.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_skill_extraction_technical():
    """Test extraction of technical skills."""
    from src.skill_extractor import SkillExtractor

    extractor = SkillExtractor()

    text = "I am proficient in Python, Java, and machine learning. I also know SQL and Docker."
    skills = extractor.extract_skills(text)

    assert "python" in skills["technical"]
    assert "java" in skills["technical"]
    assert "machine learning" in skills["technical"]
    assert "sql" in skills["technical"]
    assert "docker" in skills["technical"]

    print("✅ test_skill_extraction_technical passed")


def test_skill_extraction_soft():
    """Test extraction of soft skills."""
    from src.skill_extractor import SkillExtractor

    extractor = SkillExtractor()

    text = "Strong communication skills, excellent teamwork, and proven leadership abilities."
    skills = extractor.extract_skills(text)

    assert "communication" in skills["soft"]
    assert "teamwork" in skills["soft"]
    assert "leadership" in skills["soft"]

    print("✅ test_skill_extraction_soft passed")


def test_skill_match_computation():
    """Test skill match percentage computation."""
    from src.skill_extractor import SkillExtractor

    extractor = SkillExtractor()

    resume_skills = {"python", "java", "sql", "communication"}
    jd_skills = {"python", "java", "sql", "docker", "kubernetes"}

    match_pct, matched, missing = extractor.compute_skill_match(
        resume_skills, jd_skills
    )

    assert match_pct == 3 / 5  # 3 out of 5 JD skills matched
    assert matched == {"python", "java", "sql"}
    assert missing == {"docker", "kubernetes"}

    print("✅ test_skill_match_computation passed")


def test_skill_extraction_empty():
    """Test skill extraction with empty input."""
    from src.skill_extractor import SkillExtractor

    extractor = SkillExtractor()

    skills = extractor.extract_skills("")
    assert len(skills["all"]) == 0

    skills = extractor.extract_skills(None)
    assert len(skills["all"]) == 0

    print("✅ test_skill_extraction_empty passed")


def test_skill_match_empty_jd():
    """Test skill match with no JD skills."""
    from src.skill_extractor import SkillExtractor

    extractor = SkillExtractor()

    match_pct, _, _ = extractor.compute_skill_match(
        {"python", "java"}, set()
    )
    assert match_pct == 1.0  # No requirements = full match

    print("✅ test_skill_match_empty_jd passed")


if __name__ == "__main__":
    test_skill_extraction_technical()
    test_skill_extraction_soft()
    test_skill_match_computation()
    test_skill_extraction_empty()
    test_skill_match_empty_jd()
    print("\n✅ All skill extractor tests passed!")

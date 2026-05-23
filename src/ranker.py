"""
Candidate ranking and gap analysis for the Resume Screening System.

Takes scored candidates and produces a ranked list with detailed
explanations suitable for non-technical users (recruiters, HR managers).
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CandidateRanker:
    """
    Ranks candidates by their composite relevance scores and
    generates human-readable ranking reports with skill gap analysis.
    """

    def rank_candidates(
        self,
        candidates: List[Dict[str, Any]],
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Sort candidates by overall_score descending and assign ranks.

        Args:
            candidates: List of candidate dictionaries, each containing
                        at minimum 'overall_score'.
            top_n: If set, return only the top N candidates.

        Returns:
            Sorted list of candidate dicts with 'rank' field added.
        """
        # Sort by overall_score descending
        sorted_candidates = sorted(
            candidates,
            key=lambda c: c.get("overall_score", 0.0),
            reverse=True,
        )

        # Assign ranks
        for i, candidate in enumerate(sorted_candidates, start=1):
            candidate["rank"] = i

        if top_n is not None:
            sorted_candidates = sorted_candidates[:top_n]

        logger.info(f"Ranked {len(sorted_candidates)} candidates.")
        return sorted_candidates

    def generate_ranking_report(
        self,
        ranked_candidates: List[Dict[str, Any]],
        job_description_summary: str = "",
        jd_skills: Optional[List[str]] = None,
    ) -> str:
        """
        Generate a human-readable ranking report.

        Designed to be clear and understandable for non-technical users
        such as recruiters and HR managers.

        Args:
            ranked_candidates: Ranked list of candidate dictionaries.
            job_description_summary: Brief summary of the JD (for context).
            jd_skills: Required skills from the JD.

        Returns:
            Formatted report string.
        """
        lines = []
        lines.append("=" * 70)
        lines.append("   RESUME SCREENING — CANDIDATE RANKING REPORT")
        lines.append("=" * 70)
        lines.append("")

        if job_description_summary:
            lines.append(f"Job Description: {job_description_summary[:200]}")
            lines.append("")

        if jd_skills:
            lines.append(f"Required Skills ({len(jd_skills)}): {', '.join(jd_skills)}")
            lines.append("")

        lines.append("-" * 70)
        lines.append("")

        for candidate in ranked_candidates:
            rank = candidate.get("rank", "?")
            name = candidate.get("candidate_id", candidate.get("id", "Unknown"))
            category = candidate.get("category", "N/A")
            overall = candidate.get("overall_score", 0.0)
            cosine = candidate.get("cosine_similarity", 0.0)
            skill_match = candidate.get("skill_match_score", 0.0)
            matched = candidate.get("matched_skills", [])
            missing = candidate.get("missing_skills", [])

            # Score tier
            tier = self._score_tier(overall)

            lines.append(f"  Rank #{rank}  |  {tier}")
            lines.append(f"  Candidate: {name}")
            lines.append(f"  Category:  {category}")
            lines.append(f"  Overall Score:      {overall:.2%}")
            lines.append(f"  Content Relevance:  {cosine:.2%}")
            lines.append(f"  Skill Match:        {skill_match:.2%}")

            if matched:
                lines.append(f"  ✅ Matched Skills:  {', '.join(matched)}")
            if missing:
                lines.append(f"  ❌ Missing Skills:  {', '.join(missing)}")

            lines.append("")
            lines.append(f"  {'─' * 50}")
            lines.append("")

        # Summary statistics
        if ranked_candidates:
            scores = [c.get("overall_score", 0.0) for c in ranked_candidates]
            lines.append("-" * 70)
            lines.append("  SUMMARY")
            lines.append(f"  Total candidates evaluated: {len(ranked_candidates)}")
            lines.append(f"  Highest score: {max(scores):.2%}")
            lines.append(f"  Lowest score:  {min(scores):.2%}")
            lines.append(f"  Average score: {sum(scores)/len(scores):.2%}")

            # Count by tier
            strong = sum(1 for s in scores if s >= 0.7)
            moderate = sum(1 for s in scores if 0.4 <= s < 0.7)
            weak = sum(1 for s in scores if s < 0.4)
            lines.append(f"  Strong fit (≥70%):   {strong}")
            lines.append(f"  Moderate fit (40-69%): {moderate}")
            lines.append(f"  Weak fit (<40%):     {weak}")

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)

    def _score_tier(self, score: float) -> str:
        """Map a score to a human-readable tier label."""
        if score >= 0.85:
            return "⭐ EXCELLENT FIT"
        elif score >= 0.70:
            return "✅ STRONG FIT"
        elif score >= 0.55:
            return "🔶 GOOD FIT"
        elif score >= 0.40:
            return "🔸 MODERATE FIT"
        else:
            return "⚠️  WEAK FIT"

    def identify_common_skill_gaps(
        self,
        ranked_candidates: List[Dict[str, Any]],
    ) -> Dict[str, int]:
        """
        Identify the most commonly missing skills across all candidates.

        Useful for understanding where the talent pool is weakest.

        Args:
            ranked_candidates: List of candidate dictionaries.

        Returns:
            Dictionary mapping skill → count of candidates missing it.
        """
        gap_counts: Dict[str, int] = {}
        for candidate in ranked_candidates:
            for skill in candidate.get("missing_skills", []):
                gap_counts[skill] = gap_counts.get(skill, 0) + 1

        # Sort by frequency (most common gaps first)
        return dict(
            sorted(gap_counts.items(), key=lambda x: x[1], reverse=True)
        )

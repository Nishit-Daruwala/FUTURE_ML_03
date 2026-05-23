"""
Evaluation pipeline for the Resume Screening System.

Loads trained models and runs comprehensive evaluation:
1. Classification accuracy and per-class metrics
2. Sample resume screening with a demo job description
3. Ranking quality analysis
4. Skill extraction validation

Usage:
    python -m pipeline.evaluate_pipeline
"""

import os
import sys
import json
import logging
from datetime import datetime

import pandas as pd

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config import load_config, setup_logging, ensure_directories
from src.inference import ResumeScreener

logger = logging.getLogger(__name__)


# ============================================================
# Sample Job Descriptions for Evaluation
# ============================================================

SAMPLE_JOB_DESCRIPTIONS = {
    "Data Science": """
        Senior Data Scientist

        We are looking for a Senior Data Scientist to join our analytics team.
        The ideal candidate will have strong expertise in machine learning,
        statistical modeling, and data analysis.

        Required Skills:
        - Python programming (pandas, numpy, scikit-learn)
        - Machine learning and deep learning (TensorFlow or PyTorch)
        - SQL and database management
        - Data visualization (matplotlib, seaborn, Tableau)
        - Natural language processing
        - Statistical analysis and hypothesis testing
        - Cloud platforms (AWS or GCP)
        - Git version control

        Soft Skills:
        - Strong communication and presentation skills
        - Teamwork and collaboration
        - Problem solving and critical thinking
        - Ability to explain complex results to non-technical stakeholders
    """,
    "Java Developer": """
        Senior Java Developer

        We need an experienced Java Developer to build scalable
        enterprise applications.

        Required Skills:
        - Java programming (Java 11+)
        - Spring Boot and Spring framework
        - REST API development
        - Microservices architecture
        - SQL and NoSQL databases (PostgreSQL, MongoDB)
        - Docker and Kubernetes
        - CI/CD pipelines (Jenkins)
        - Git version control
        - Unit testing (JUnit)
        - Agile/Scrum methodology

        Soft Skills:
        - Team collaboration
        - Problem solving
        - Communication skills
    """,
    "Web Designing": """
        UI/UX Web Designer

        We are hiring a creative Web Designer with a strong eye for
        modern, responsive design.

        Required Skills:
        - HTML, CSS, JavaScript
        - React or Angular framework
        - UI/UX design principles
        - Figma, Sketch, or Adobe XD
        - Responsive design
        - SEO best practices
        - Prototyping and wireframing
        - Git version control

        Soft Skills:
        - Creativity and innovation
        - Attention to detail
        - Communication skills
        - Teamwork
    """,
}


def run_evaluation_pipeline():
    """Execute the full evaluation pipeline."""
    # ── Step 0: Configuration ──────────────────────────────
    config = load_config()
    setup_logging(config)
    ensure_directories(config)

    logger.info("=" * 60)
    logger.info("  RESUME SCREENING SYSTEM — EVALUATION PIPELINE")
    logger.info("=" * 60)

    # Check if models exist
    model_dir = config.paths.models_dir
    if not os.path.exists(os.path.join(model_dir, "tfidf_vectorizer.pkl")):
        logger.error(
            "No trained models found. Run the training pipeline first:\n"
            "  python -m pipeline.train_pipeline"
        )
        return

    # ── Step 1: Load preprocessed data ─────────────────────
    logger.info("\n── Step 1: Loading Preprocessed Data ──")
    processed_path = os.path.join(
        config.paths.processed_data_dir, "preprocessed_resumes.csv"
    )

    if not os.path.exists(processed_path):
        logger.error(f"Preprocessed data not found at: {processed_path}")
        return

    df = pd.read_csv(processed_path)
    logger.info(f"Loaded {len(df)} preprocessed resumes.")

    # ── Step 2: Initialize screener ────────────────────────
    logger.info("\n── Step 2: Initializing ResumeScreener ──")
    screener = ResumeScreener(config=config)

    # ── Step 3: Run sample screenings ──────────────────────
    logger.info("\n── Step 3: Running Sample Screenings ──")

    # Map roles to actual category names in the dataset
    category_mapping = {
        "Data Science": "INFORMATION-TECHNOLOGY",
        "Java Developer": "INFORMATION-TECHNOLOGY",
        "Web Designing": "DESIGNER",
    }

    all_results = {}

    for role, jd_text in SAMPLE_JOB_DESCRIPTIONS.items():
        logger.info(f"\n{'─' * 40}")
        logger.info(f"Screening for: {role}")
        logger.info(f"{'─' * 40}")

        # Get matching category in dataset
        dataset_category = category_mapping.get(role, role)

        # Filter resumes by category for a focused demo
        # Use a sample from matching category + some from other categories
        matching = df[df["Category"] == dataset_category]
        non_matching = df[df["Category"] != dataset_category].sample(
            n=min(10, len(df[df["Category"] != dataset_category])),
            random_state=config.general.random_seed,
        )

        sample_df = pd.concat([matching.head(10), non_matching]).reset_index(drop=True)

        # Determine text column
        text_col = "Resume_str" if "Resume_str" in sample_df.columns else "cleaned_text"

        resumes = sample_df[text_col].fillna("").tolist()
        resume_ids = [
            f"{row['Category']}_{idx}"
            for idx, row in sample_df.iterrows()
        ]

        # Screen resumes
        result = screener.screen_resumes(
            job_description=jd_text,
            resumes=resumes,
            resume_ids=resume_ids,
            target_category=dataset_category,
            top_n=10,
        )

        # Print the report
        print(result["ranking_report"])

        # Save individual report
        report_path = os.path.join(
            config.paths.outputs_dir,
            f"ranking_report_{role.replace(' ', '_').lower()}.txt"
        )
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(result["ranking_report"])

        all_results[role] = {
            "total_screened": result["total_screened"],
            "jd_skills": result["job_description_skills"],
            "common_skill_gaps": result["common_skill_gaps"],
            "top_candidates": [
                {
                    "candidate_id": c["candidate_id"],
                    "rank": c["rank"],
                    "overall_score": c["overall_score"],
                    "category": c.get("category", "N/A"),
                }
                for c in result["ranked_candidates"][:5]
            ],
        }

    # ── Step 4: Save evaluation results ────────────────────
    logger.info("\n── Step 4: Saving Evaluation Results ──")

    eval_report = {
        "timestamp": datetime.now().isoformat(),
        "total_resumes_in_dataset": len(df),
        "sample_screenings": all_results,
    }

    eval_path = os.path.join(config.paths.outputs_dir, "evaluation_metrics.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2, default=str)

    logger.info(f"Evaluation report saved to: {eval_path}")

    # ── Step 5: Skill Extraction Validation ────────────────
    logger.info("\n── Step 5: Skill Extraction Validation ──")

    # Test skill extraction on a few sample JDs
    for role, jd_text in SAMPLE_JOB_DESCRIPTIONS.items():
        analysis = screener.get_skill_analysis(jd_text)
        logger.info(
            f"  {role} JD: "
            f"{analysis['technical_count']} technical, "
            f"{analysis['soft_count']} soft skills"
        )
        logger.info(f"    Technical: {analysis['technical_skills'][:10]}")
        logger.info(f"    Soft:      {analysis['soft_skills'][:5]}")

    # ── Summary ───────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  EVALUATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Reports saved to: {config.paths.outputs_dir}/")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        run_evaluation_pipeline()
    except Exception as e:
        logger.exception(f"Evaluation pipeline failed: {e}")
        sys.exit(1)

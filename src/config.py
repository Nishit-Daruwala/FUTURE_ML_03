"""
Configuration loader for the Resume Screening System.

Reads config/config.yaml and provides typed access to all parameters.
Uses dataclasses for clean, validated configuration objects.
"""

import os
import yaml
import logging
from dataclasses import dataclass, field
from typing import List, Tuple


# ============================================================
# Configuration Dataclasses
# ============================================================

@dataclass
class PathsConfig:
    """File path configuration."""
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    skills_dir: str = "data/skills"
    models_dir: str = "models"
    outputs_dir: str = "outputs"
    dataset_filename: str = "UpdatedResumeDataSet.csv"

    @property
    def dataset_path(self) -> str:
        return os.path.join(self.raw_data_dir, self.dataset_filename)


@dataclass
class PreprocessingConfig:
    """Text preprocessing configuration."""
    min_word_length: int = 2
    remove_stopwords: bool = True
    lemmatize: bool = True
    lowercase: bool = True
    extra_stopwords: List[str] = field(default_factory=lambda: [
        "resume", "curriculum", "vitae", "objective",
        "reference", "references", "available", "upon", "request"
    ])


@dataclass
class TfidfConfig:
    """TF-IDF vectorizer configuration."""
    max_features: int = 10000
    ngram_range: Tuple[int, int] = (1, 2)
    min_df: int = 2
    max_df: float = 0.95
    sublinear_tf: bool = True
    norm: str = "l2"


@dataclass
class ClassifierConfig:
    """Resume category classifier configuration."""
    type: str = "multinomial_nb"
    test_size: float = 0.2
    random_state: int = 42
    svc_C: float = 1.0
    svc_max_iter: int = 10000


@dataclass
class ScoringConfig:
    """Candidate scoring configuration."""
    similarity_weight: float = 0.60
    skill_match_weight: float = 0.40
    category_match_bonus: float = 0.05


@dataclass
class SkillsConfig:
    """Skill extraction configuration."""
    technical_skills_file: str = "technical_skills.txt"
    soft_skills_file: str = "soft_skills.txt"
    case_sensitive: bool = False


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    log_file: str = "outputs/app.log"
    console_output: bool = True


@dataclass
class GeneralConfig:
    """General configuration."""
    random_seed: int = 42
    verbose: bool = True


@dataclass
class Config:
    """Root configuration object containing all sub-configurations."""
    paths: PathsConfig = field(default_factory=PathsConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    tfidf: TfidfConfig = field(default_factory=TfidfConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    skills: SkillsConfig = field(default_factory=SkillsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    general: GeneralConfig = field(default_factory=GeneralConfig)


# ============================================================
# Config Loading Functions
# ============================================================

def _find_project_root() -> str:
    """Find the project root directory by looking for config/config.yaml."""
    # Start from this file's directory and walk up
    current = os.path.dirname(os.path.abspath(__file__))
    for _ in range(5):  # max 5 levels up
        config_path = os.path.join(current, "config", "config.yaml")
        if os.path.exists(config_path):
            return current
        current = os.path.dirname(current)
    # Fallback: assume CWD is project root
    return os.getcwd()


def load_config(config_path: str = None) -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Optional explicit path to config.yaml.
                     If None, auto-discovers from project root.

    Returns:
        Config object with all parameters loaded.
    """
    if config_path is None:
        project_root = _find_project_root()
        config_path = os.path.join(project_root, "config", "config.yaml")

    if not os.path.exists(config_path):
        logging.warning(
            f"Config file not found at {config_path}. Using defaults."
        )
        return Config()

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        return Config()

    # Parse each section
    config = Config(
        paths=PathsConfig(**raw.get("paths", {})),
        preprocessing=PreprocessingConfig(**raw.get("preprocessing", {})),
        tfidf=_parse_tfidf_config(raw.get("tfidf", {})),
        classifier=ClassifierConfig(**raw.get("classifier", {})),
        scoring=ScoringConfig(**raw.get("scoring", {})),
        skills=SkillsConfig(**raw.get("skills", {})),
        logging=LoggingConfig(**raw.get("logging", {})),
        general=GeneralConfig(**raw.get("general", {})),
    )

    return config


def _parse_tfidf_config(raw_tfidf: dict) -> TfidfConfig:
    """Parse TF-IDF config, converting ngram_range list to tuple."""
    if "ngram_range" in raw_tfidf:
        raw_tfidf["ngram_range"] = tuple(raw_tfidf["ngram_range"])
    return TfidfConfig(**raw_tfidf)


def setup_logging(config: Config) -> None:
    """
    Configure Python logging based on config settings.

    Args:
        config: Config object with logging settings.
    """
    log_config = config.logging
    level = getattr(logging, log_config.level.upper(), logging.INFO)

    # Ensure output directory exists for log file
    log_dir = os.path.dirname(log_config.log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    handlers = []

    # File handler
    file_handler = logging.FileHandler(log_config.log_file, encoding="utf-8")
    file_handler.setLevel(level)
    handlers.append(file_handler)

    # Console handler
    if log_config.console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        handlers.append(console_handler)

    # Configure root logger
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def ensure_directories(config: Config) -> None:
    """Create all required directories if they don't exist."""
    dirs = [
        config.paths.raw_data_dir,
        config.paths.processed_data_dir,
        config.paths.skills_dir,
        config.paths.models_dir,
        config.paths.outputs_dir,
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


# ============================================================
# Module-level convenience
# ============================================================

# Quick access: from src.config import get_config
_cached_config = None

def get_config(config_path: str = None) -> Config:
    """
    Get the global config instance (cached after first load).

    Args:
        config_path: Optional explicit path to config.yaml.

    Returns:
        Config object.
    """
    global _cached_config
    if _cached_config is None:
        _cached_config = load_config(config_path)
    return _cached_config

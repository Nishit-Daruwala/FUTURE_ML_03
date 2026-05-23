# Resume / Candidate Screening System — Project Plan

## 1. Task Understanding

### What the Task is Asking
Build a **Machine Learning–based resume screening and ranking system** that automates the process of evaluating resumes against a given job description. The system must read resume text, extract skills and keywords using NLP, compute similarity/relevance scores against a job role, rank candidates by fit, and highlight skill gaps — all presented in a way that is explainable to non-technical recruiters.

### Project Objectives
1. **Ingest & Parse** — Accept resume text (from CSV dataset or raw text input) and clean/normalize it for NLP processing
2. **Skill Extraction** — Use NLP (spaCy, NLTK) to extract technical and soft skills from both resumes and job descriptions
3. **Similarity Scoring** — Vectorize documents (TF-IDF) and compute cosine similarity between each resume and a target job description
4. **Candidate Ranking** — Produce an ordered ranking of candidates based on composite relevance scores
5. **Skill Gap Analysis** — Identify required skills that are missing from each resume
6. **Explainability** — Provide clear, human-readable explanations of why candidates scored the way they did

### Expected Final Outcome
A production-quality Python package that can:
- Load a dataset of resumes (CSV with `Category` and `Resume_str` columns)
- Accept a job description (text)
- Process, score, and rank all resumes against that job description
- Output ranked results with scores, matched skills, and missing skills
- Expose a clean inference API ready for future Streamlit UI integration

### Inputs & Outputs

| Direction | Description | Format |
|-----------|-------------|--------|
| **Input** | Resume dataset | CSV file with columns: `ID`, `Resume_str`, `Resume_html`, `Category` |
| **Input** | Job description text | Free-form text string |
| **Output** | Ranked candidate list | List of dicts with `rank`, `candidate_id`, `category`, `overall_score`, `similarity_score`, `skill_match_score`, `matched_skills`, `missing_skills` |
| **Output** | Evaluation metrics | Classification accuracy on known categories, scoring distribution analysis |
| **Output** | Saved model artifacts | Trained TF-IDF vectorizer, skills database, configuration |

### Functional Requirements
- FR-1: Clean and preprocess resume text (remove HTML, special chars, stopwords, lemmatize)
- FR-2: Extract skills from resume text using a curated skills database + NLP entity matching
- FR-3: Parse and extract requirements from job descriptions
- FR-4: Compute TF-IDF cosine similarity between resume and job description
- FR-5: Compute skill-match percentage (matched skills / required skills)
- FR-6: Produce a composite score: `overall_score = w1 * similarity_score + w2 * skill_match_score`
- FR-7: Rank candidates by overall_score descending
- FR-8: Identify and list missing/gap skills per candidate
- FR-9: Support category-based classification of resumes (e.g., "Data Science", "HR", etc.)
- FR-10: Save/load trained model artifacts for reuse

### Non-Functional Requirements
- NFR-1: Modular, clean code with type hints and docstrings
- NFR-2: Configurable parameters via YAML config file
- NFR-3: Reproducible pipeline (fixed random seeds, deterministic processing)
- NFR-4: Efficient inference (< 2 seconds per resume)
- NFR-5: Comprehensive logging
- NFR-6: Streamlit-ready backend (clean prediction API)

### Constraints
- Must use spaCy, NLTK, and scikit-learn as primary NLP/ML libraries
- Must NOT download datasets automatically — user must provide the data
- All dependencies installed only inside project virtual environment
- No external API calls required (fully offline-capable system)

### Assumptions
- The primary dataset follows the Kaggle Resume Dataset schema (columns: `ID`, `Resume_str`, `Resume_html`, `Category`)
- Resumes are in English
- Job descriptions are provided as plain text
- The user will manually download and place the dataset in `data/raw/` before running training

---

## 2. Technical Strategy

### Problem Type
**Multi-faceted NLP problem** combining:
1. **Text Classification** — Categorize resumes into job roles (supervised, using `Category` labels)
2. **Information Extraction** — Extract skills/entities from unstructured text
3. **Text Similarity / Ranking** — Score and rank resumes against a job description
4. **Gap Analysis** — Set-difference between required and found skills

### ML / NLP Approach

#### Pipeline Overview
```
Raw Resume Text
    │
    ▼
┌─────────────────────┐
│  Text Preprocessing  │ ← NLTK (tokenize, stopwords, lemmatize)
│  HTML removal, clean │
└──────────┬──────────┘
           │
    ┌──────┴──────┐
    ▼              ▼
┌──────────┐  ┌──────────────┐
│  Skill   │  │   TF-IDF     │ ← scikit-learn TfidfVectorizer
│Extraction│  │ Vectorization│
│ (spaCy)  │  └──────┬───────┘
└────┬─────┘         │
     │               ▼
     │    ┌──────────────────┐
     │    │ Cosine Similarity │ ← scikit-learn cosine_similarity
     │    │ (resume vs JD)   │
     │    └──────┬───────────┘
     │           │
     ▼           ▼
┌────────────────────────┐
│  Composite Scoring     │
│  skill_match + cosine  │
│  + category bonus      │
└──────────┬─────────────┘
           │
           ▼
┌────────────────────────┐
│  Ranking & Gap Analysis│
│  Sort by score, show   │
│  missing skills        │
└────────────────────────┘
```

### Feature Engineering
1. **TF-IDF Features** — Transform resume and JD text into TF-IDF vectors (unigrams + bigrams, max 10,000 features)
2. **Skill Features** — Binary skill vector (1 if skill present, 0 if not) from curated skills database
3. **Category Features** — One-hot encoded resume category from classifier
4. **Text Statistics** — Resume length, keyword density, section completeness (optional enhancement)

### Data Preprocessing Strategy
1. **HTML Stripping** — Remove HTML tags from `Resume_html` / clean `Resume_str`
2. **Text Normalization** — Lowercase, remove special characters, URLs, email addresses, phone numbers
3. **Tokenization** — NLTK word_tokenize
4. **Stopword Removal** — NLTK English stopwords (augmented with domain-specific stopwords)
5. **Lemmatization** — NLTK WordNetLemmatizer (preserves meaningful word forms better than stemming)
6. **Deduplication** — Remove duplicate resumes by text similarity threshold

### Model Selection Strategy

| Component | Method | Justification |
|-----------|--------|---------------|
| **Text Vectorization** | TF-IDF (scikit-learn) | Industry-standard for document similarity; no GPU required; fast; interpretable |
| **Resume Classification** | Multinomial Naive Bayes + One-vs-Rest SVM | Fast training, good for text classification; SVM as backup for better accuracy |
| **Skill Extraction** | spaCy PhraseMatcher + curated skills DB | Reliable pattern matching; no NER training needed; easy to extend |
| **Similarity Scoring** | Cosine Similarity | Standard metric for TF-IDF vector comparison; well-understood |
| **Ranking** | Weighted composite score | Configurable weights allow tuning for different use cases |

### Training Strategy
1. Train TF-IDF vectorizer on the full resume corpus
2. Train resume category classifier on labeled resume data (Category column)
3. Build and validate skills database from domain knowledge
4. No deep learning required — all models are traditional ML (fast, interpretable)

### Validation / Testing Strategy
- **Train/Test Split**: 80/20 stratified split by Category
- **Cross-Validation**: 5-fold stratified CV for classifier evaluation
- **Scoring Validation**: Manual inspection of top-k ranked resumes for sample JDs
- **No data leakage**: TF-IDF fitted only on training set, transformed on test set

### Evaluation Metrics
| Metric | Applies To | Description |
|--------|-----------|-------------|
| **Accuracy** | Category classifier | % of correctly classified resume categories |
| **Precision/Recall/F1** | Category classifier | Per-class and macro-averaged |
| **Cosine Similarity Distribution** | Ranking system | Distribution of scores to check spread |
| **Mean Reciprocal Rank (MRR)** | Ranking | Measures ranking quality for relevant resumes |
| **Skill Coverage %** | Skill extraction | % of known skills correctly extracted |

### Error Analysis Plan
1. Review misclassified resumes — check if category label is ambiguous
2. Inspect low-scoring resumes that should rank high (false negatives)
3. Analyze skill extraction misses — update skills database accordingly
4. Check edge cases: very short resumes, non-English text, heavily formatted resumes

### Optimization Strategy
1. Tune TF-IDF parameters (max_features, ngram_range, min_df, max_df)
2. Adjust composite score weights via grid search on validation set
3. Expand skills database based on error analysis
4. Consider sublinear TF scaling for better term weighting

---

## 3. Project Architecture

### Folder Structure
```
Task-3/
├── PROJECT_PLAN.md              # This file
├── README.md                    # Project documentation & usage guide
├── requirements.txt             # Python dependencies
├── config/
│   └── config.yaml              # All configurable parameters
├── data/
│   ├── raw/                     # User places raw dataset CSV here
│   ├── processed/               # Cleaned & preprocessed data
│   └── skills/                  # Curated skills database files
├── src/
│   ├── __init__.py
│   ├── config.py                # Config loader (reads config.yaml)
│   ├── preprocessing.py         # Text cleaning & preprocessing pipeline
│   ├── skill_extractor.py       # Skill extraction using spaCy PhraseMatcher
│   ├── vectorizer.py            # TF-IDF vectorization wrapper
│   ├── classifier.py            # Resume category classifier (train & predict)
│   ├── scorer.py                # Similarity scoring & composite ranking
│   ├── ranker.py                # Candidate ranking & gap analysis
│   └── inference.py             # Unified inference API (Streamlit-ready)
├── pipeline/
│   ├── __init__.py
│   ├── train_pipeline.py        # End-to-end training orchestrator
│   └── evaluate_pipeline.py     # Evaluation & metrics computation
├── models/                      # Saved model artifacts (auto-created)
│   ├── tfidf_vectorizer.pkl
│   ├── category_classifier.pkl
│   └── label_encoder.pkl
├── outputs/                     # Evaluation outputs (auto-created)
│   ├── classification_report.txt
│   ├── ranking_results.csv
│   └── evaluation_metrics.json
├── notebooks/                   # Jupyter notebooks for exploration
│   └── exploration.ipynb
└── tests/                       # Unit tests
    ├── __init__.py
    ├── test_preprocessing.py
    └── test_skill_extractor.py
```

### Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `config.py` | Load YAML config, provide typed access to all parameters |
| `preprocessing.py` | HTML removal, text cleaning, tokenization, stopword removal, lemmatization |
| `skill_extractor.py` | Load skills database, use spaCy PhraseMatcher to find skills in text |
| `vectorizer.py` | Wrapper around sklearn TF-IDF: fit, transform, save, load |
| `classifier.py` | Train Naive Bayes / SVM classifier on resume categories; predict new resumes |
| `scorer.py` | Compute cosine similarity, skill match percentage, composite score |
| `ranker.py` | Sort candidates by score, produce ranking with gap analysis |
| `inference.py` | Single entry point: `screen_resumes(job_description, resumes) → ranked_results` |
| `train_pipeline.py` | Orchestrate: load data → preprocess → train vectorizer → train classifier → save |
| `evaluate_pipeline.py` | Load test data → predict → compute metrics → generate reports |

### Training Pipeline Flow
```
1. Load raw CSV data
2. Preprocess all resume texts
3. Stratified train/test split
4. Fit TF-IDF vectorizer on training set
5. Train category classifier
6. Evaluate on test set
7. Save all artifacts to models/
8. Generate evaluation reports to outputs/
```

### Inference Pipeline Flow
```
1. Load saved model artifacts (vectorizer, classifier, skills DB)
2. Accept job description text + list of resume texts
3. Preprocess all inputs
4. Extract skills from JD and each resume
5. Vectorize JD and resumes with loaded TF-IDF
6. Compute cosine similarity scores
7. Compute skill match percentages
8. Calculate composite scores
9. Rank candidates
10. Return ranked list with scores, matched/missing skills
```

### Config Management
Single `config/config.yaml` file controls:
- File paths (data, models, outputs)
- Preprocessing parameters (min word length, custom stopwords)
- TF-IDF parameters (max_features, ngram_range, min_df, max_df)
- Scoring weights (similarity_weight, skill_match_weight)
- Classifier type and hyperparameters
- Logging level
- Random seed

### Logging Strategy
- Python `logging` module with configurable level
- Log to both console and file (`outputs/app.log`)
- Key events logged: data loading, preprocessing stats, training progress, evaluation results, inference timing

### Model Artifact Storage
- All artifacts saved as `.pkl` files using `joblib` (efficient serialization)
- Stored in `models/` directory
- Each training run overwrites previous artifacts (configurable to version)
- Metadata saved alongside (training date, data stats, parameters)

---

## 4. Dependency Plan

### Required Python Packages

| Package | Version | Purpose |
|---------|---------|---------|
| `spacy` | >=3.5 | NLP pipeline for skill extraction (PhraseMatcher, tokenization) |
| `nltk` | >=3.8 | Text preprocessing (tokenization, stopwords, lemmatization) |
| `scikit-learn` | >=1.3 | TF-IDF vectorization, classifiers, cosine similarity, metrics |
| `pandas` | >=2.0 | Data loading, manipulation, and analysis |
| `numpy` | >=1.24 | Numerical operations |
| `pyyaml` | >=6.0 | YAML config file parsing |
| `joblib` | >=1.3 | Efficient model serialization (comes with scikit-learn) |
| `beautifulsoup4` | >=4.12 | HTML tag stripping from resume text |
| `lxml` | >=4.9 | HTML parser backend for BeautifulSoup |

### spaCy Model
- `en_core_web_sm` — Small English model for tokenization and basic NLP
- Downloaded via: `python -m spacy download en_core_web_sm` (inside venv)

### NLTK Data
- `punkt_tab` — Sentence/word tokenizer models
- `stopwords` — English stopword list
- `wordnet` — WordNet lemmatizer database
- Downloaded via: `nltk.download(...)` at first run (inside venv)

### Development Dependencies (optional)
| Package | Purpose |
|---------|---------|
| `jupyter` | Notebook exploration |
| `pytest` | Unit testing |
| `streamlit` | Future UI (NOT installed yet) |

### Why Each Core Dependency

- **spaCy**: Best-in-class NLP library for production use. PhraseMatcher enables fast, rule-based skill extraction without expensive NER training. Far more efficient than regex for multi-word skill matching.
- **NLTK**: Mature library with excellent preprocessing utilities. WordNet lemmatizer produces more linguistically accurate lemmas than spaCy's default. Comprehensive stopword lists.
- **scikit-learn**: Industry standard for classical ML. TfidfVectorizer is the gold standard for document vectorization. Cosine similarity, Naive Bayes, and SVM classifiers are all production-proven for text classification.
- **pandas**: Essential for loading CSV data and structured data manipulation.
- **BeautifulSoup**: The resume dataset contains HTML-formatted text that must be stripped cleanly.
- **PyYAML**: Clean config management without boilerplate code.

---

## 5. Dataset Requirements

> [!IMPORTANT]
> **No datasets will be downloaded automatically.** The user must manually download and place dataset files.

### Primary Recommended Dataset
**Resume Dataset (Kaggle)** — https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset

**Expected Schema:**
| Column | Type | Description |
|--------|------|-------------|
| `ID` | int | Unique resume identifier |
| `Resume_str` | str | Full resume text content |
| `Resume_html` | str | Resume in HTML format |
| `Category` | str | Job category label (e.g., "Data Science", "HR", "Java Developer") |

**Expected Categories (≈25):**
Data Science, HR, Advocate, Arts, Web Designing, Mechanical Engineer, Sales, Health and Fitness, Civil Engineer, Java Developer, Business Analyst, SAP Developer, Automation Testing, Electrical Engineering, Operations Manager, Python Developer, DevOps Engineer, Network Security Engineer, PMO, Database, Hadoop, ETL Developer, DotNet Developer, Blockchain, Testing

**Setup Instructions:**
1. Download the dataset from the Kaggle link above
2. Extract the CSV file
3. Place it at: `data/raw/UpdatedResumeDataSet.csv`

### Alternative: Simulated Data
If the user cannot access Kaggle, the system includes a small synthetic sample dataset for testing purposes. This will be generated in `data/raw/sample_resumes.csv` with the same schema.

---

## 6. Implementation Workflow

### Phase 1: Foundation (config, preprocessing)
1. Create project structure (all directories and `__init__.py` files)
2. Write `config/config.yaml` with all default parameters
3. Implement `src/config.py` — config loader
4. Implement `src/preprocessing.py` — full text preprocessing pipeline
5. Write unit tests for preprocessing

### Phase 2: Core NLP (skill extraction, vectorization)
6. Create `data/skills/technical_skills.txt` and `data/skills/soft_skills.txt`
7. Implement `src/skill_extractor.py` — spaCy PhraseMatcher skill extraction
8. Implement `src/vectorizer.py` — TF-IDF vectorization wrapper
9. Write unit tests for skill extractor

### Phase 3: ML Models (classification, scoring)
10. Implement `src/classifier.py` — resume category classifier
11. Implement `src/scorer.py` — similarity scoring and composite scores
12. Implement `src/ranker.py` — ranking and gap analysis

### Phase 4: Pipelines (training, evaluation)
13. Implement `pipeline/train_pipeline.py` — end-to-end training
14. Implement `pipeline/evaluate_pipeline.py` — evaluation and reporting

### Phase 5: Inference & Integration
15. Implement `src/inference.py` — unified prediction API
16. Write `README.md` — full usage documentation
17. Create `requirements.txt`

### Phase 6: Validation
18. Run full training pipeline on dataset
19. Run evaluation and review metrics
20. Test inference API with sample job descriptions
21. Verify model artifacts are saved/loaded correctly

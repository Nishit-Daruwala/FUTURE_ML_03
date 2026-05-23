# 📄 Resume Screening & Candidate Ranking System

An ML-based system that automatically screens, scores, and ranks resumes against job descriptions using NLP techniques. Built with **spaCy**, **NLTK**, and **scikit-learn**.

---

## 🎯 What It Does

| Feature | Description |
|---------|-------------|
| **Resume Preprocessing** | Cleans HTML, removes noise, tokenizes, and lemmatizes resume text |
| **Skill Extraction** | Identifies 200+ technical and 50+ soft skills using spaCy PhraseMatcher |
| **Category Classification** | Classifies resumes into ~25 job categories (Data Science, Java Dev, HR, etc.) |
| **Similarity Scoring** | Computes TF-IDF cosine similarity between resumes and job descriptions |
| **Candidate Ranking** | Ranks candidates using a weighted composite score |
| **Skill Gap Analysis** | Highlights missing skills for each candidate |

---

## 📁 Project Structure

```
Task-3/
├── PROJECT_PLAN.md              # Detailed project planning document
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── config/
│   └── config.yaml              # All configurable parameters
├── data/
│   ├── raw/                     # Place dataset CSV here
│   ├── processed/               # Preprocessed data (auto-generated)
│   └── skills/                  # Curated skill databases
│       ├── technical_skills.txt
│       └── soft_skills.txt
├── src/
│   ├── __init__.py
│   ├── config.py                # Configuration loader
│   ├── preprocessing.py         # Text cleaning & NLP preprocessing
│   ├── skill_extractor.py       # spaCy-based skill extraction
│   ├── vectorizer.py            # TF-IDF vectorization wrapper
│   ├── classifier.py            # Resume category classifier
│   ├── scorer.py                # Similarity & composite scoring
│   ├── ranker.py                # Candidate ranking & reports
│   └── inference.py             # Unified prediction API
├── pipeline/
│   ├── __init__.py
│   ├── train_pipeline.py        # End-to-end training
│   └── evaluate_pipeline.py     # Evaluation & reporting
├── models/                      # Saved model artifacts (auto-generated)
├── outputs/                     # Reports & metrics (auto-generated)
├── tests/
│   ├── test_preprocessing.py
│   └── test_skill_extractor.py
└── notebooks/                   # Jupyter notebooks
```

---

## 🚀 Quick Start

### 1. Activate the Virtual Environment

```powershell
# Windows PowerShell
.\venv\Scripts\Activate.ps1
```

### 2. Download the Dataset

Download the [Resume Dataset from Kaggle](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) and place the CSV file at:

```
data/raw/UpdatedResumeDataSet.csv
```

### 3. Run the Training Pipeline

```powershell
python -m pipeline.train_pipeline
```

This will:
- Load and preprocess the dataset
- Train the TF-IDF vectorizer
- Train the resume category classifier
- Evaluate on a held-out test set
- Save model artifacts to `models/`
- Generate reports in `outputs/`

### 4. Run the Evaluation Pipeline

```powershell
python -m pipeline.evaluate_pipeline
```

This screens sample resumes against demo job descriptions and generates ranking reports.

### 5. Test the Inference API

```powershell
python -m src.inference
```

---

## 🔧 Configuration

All parameters are controlled via `config/config.yaml`:

| Section | Key Parameters |
|---------|---------------|
| **TF-IDF** | `max_features`, `ngram_range`, `min_df`, `max_df` |
| **Scoring** | `similarity_weight` (0.6), `skill_match_weight` (0.4) |
| **Classifier** | `type` (multinomial_nb / linear_svc / logistic_regression) |
| **Preprocessing** | `lemmatize`, `remove_stopwords`, `min_word_length` |

---

## 📊 How Scoring Works

Each resume receives a **composite score** based on:

```
overall_score = 0.6 × cosine_similarity + 0.4 × skill_match_percentage + category_bonus
```

| Component | Weight | Description |
|-----------|--------|-------------|
| **Cosine Similarity** | 60% | TF-IDF vector similarity between resume and JD |
| **Skill Match** | 40% | Percentage of required JD skills found in resume |
| **Category Bonus** | +5% | Bonus if predicted resume category matches JD role |

---

## 🧪 Running Tests

```powershell
# Run all tests
python -m tests.test_preprocessing
python -m tests.test_skill_extractor
```

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| spaCy | Skill extraction (PhraseMatcher) |
| NLTK | Tokenization, stopwords, lemmatization |
| scikit-learn | TF-IDF, classifiers, cosine similarity |
| pandas | Data processing |
| BeautifulSoup | HTML tag removal |
| PyYAML | Configuration management |
| joblib | Model serialization |

---

## 🔮 Future: Streamlit UI

The system is designed with a **Streamlit-ready backend**. The `ResumeScreener` class in `src/inference.py` provides a clean API:

```python
from src.inference import ResumeScreener

screener = ResumeScreener()
results = screener.screen_resumes(
    job_description="We need a Python developer...",
    resumes=["Resume text 1...", "Resume text 2..."],
)
print(results["ranking_report"])
```

---

## 📄 License

This project is part of the Future Interns ML Task Series.

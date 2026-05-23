"""
Streamlit UI dashboard for the Resume Screening System.

Exposes a clean, professional web interface for single-candidate screening,
batch screening and ranking, and skill database management.
"""

import os
import sys
import json
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config import load_config
from src.inference import ResumeScreener
from src.utils import extract_text_from_file

# ============================================================
# Streamlit Page Config & Theme
# ============================================================

st.set_page_config(
    page_title="AI Resume Screening & Ranking",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium UI CSS styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4f46e5, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #64748b;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 1.2rem;
        border-radius: 0.75rem;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05);
        margin-bottom: 1rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1e293b;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .pill-green {
        background-color: #dcfce7;
        color: #166534;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin: 0.2rem;
        border: 1px solid #bbf7d0;
    }
    
    .pill-red {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin: 0.2rem;
        border: 1px solid #fecaca;
    }
    
    .pill-blue {
        background-color: #dbeafe;
        color: #1e40af;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin: 0.2rem;
        border: 1px solid #bfdbfe;
    }
    
    .status-banner {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
        font-weight: 600;
    }
    
    .status-success {
        background-color: #f0fdf4;
        color: #166534;
        border: 1px solid #bbf7d0;
    }
    
    .status-warning {
        background-color: #fff9db;
        color: #c92a2a;
        border: 1px solid #ffe3e3;
    }
    
    .status-info {
        background-color: #f0f9ff;
        color: #075985;
        border: 1px solid #bae6fd;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# Cache Config & Screener Load
# ============================================================

@st.cache_resource
def get_cached_screener():
    """Load and cache the ResumeScreener instance to prevent reload overhead."""
    config = load_config()
    return ResumeScreener(config=config)

try:
    screener = get_cached_screener()
    categories_list = sorted(list(screener.classifier.label_encoder.classes_))
except Exception as e:
    st.error(f"Error loading classifier model: {e}")
    st.info("Ensure you have run the training pipeline first: python -m pipeline.train_pipeline")
    st.stop()

# ============================================================
# Preset Job Descriptions
# ============================================================

SAMPLE_JDS = {
    "Select a preset...": "",
    "Data Scientist": """Senior Data Scientist

We are looking for a Senior Data Scientist to join our analytics team. The ideal candidate will have strong expertise in machine learning, statistical modeling, and data analysis.

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
""",
    "Java Developer": """Senior Java Developer

We need an experienced Java Developer to build scalable enterprise applications.

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

Soft Skills:
- Team collaboration
- Problem solving
- Communication skills
""",
    "Web Designer (UI/UX)": """UI/UX Web Designer

We are hiring a creative Web Designer with a strong eye for modern, responsive design.

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
"""
}

# ============================================================
# Sidebar Configuration
# ============================================================

st.sidebar.markdown("### 🎯 System Controls")

# Real-time Weight sliders
st.sidebar.markdown("**Scoring Weights Configuration**")
sim_weight = st.sidebar.slider(
    "Semantic Relevance Weight (Cosine Similarity)",
    min_value=0.0,
    max_value=1.0,
    value=float(screener.scorer.similarity_weight),
    step=0.05,
    help="Weight assigned to the TF-IDF text similarity component."
)

# Ensure they sum to 1.0 dynamically
skill_weight = round(1.0 - sim_weight, 2)
st.sidebar.markdown(f"**Skill Match Weight (computed)**: `{skill_weight}`")

category_bonus = st.sidebar.slider(
    "Category Match Bonus",
    min_value=0.0,
    max_value=0.20,
    value=float(screener.scorer.category_match_bonus),
    step=0.01,
    help="Flat score bonus added if the candidate's predicted category matches target category."
)

# Apply dynamic weights back to screener instance
screener.scorer.similarity_weight = sim_weight
screener.scorer.skill_match_weight = skill_weight
screener.scorer.category_match_bonus = category_bonus

st.sidebar.markdown("---")
st.sidebar.markdown("**Target Job Category**")
target_role = st.sidebar.selectbox(
    "Filter by expected role category:",
    options=categories_list,
    index=categories_list.index("INFORMATION-TECHNOLOGY") if "INFORMATION-TECHNOLOGY" in categories_list else 0,
    help="Select the expected candidate category to trigger matching bonuses."
)

st.sidebar.markdown("---")
# Model metadata display
st.sidebar.markdown("**Model Metadata**")
st.sidebar.info(
    f"**Classifier**: `{screener.classifier.classifier_type.upper()}`\n\n"
    f"**Vocabulary Size**: `{screener.vectorizer.vectorizer.max_features} features`"
)

# ============================================================
# Main Page Header
# ============================================================

st.markdown("<h1 class='main-title'>🎯 AI Resume Screening & Ranking Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Screen, analyze, and rank candidate profiles using state-of-the-art NLP and machine learning.</p>", unsafe_allow_html=True)

# Main tabs
tab1, tab2, tab3 = st.tabs([
    "👤 Single Candidate Analysis",
    "👥 Batch Candidate Screening",
    "📊 Database & Training Insights"
])

# ============================================================
# Tab 1: Single Candidate Screening
# ============================================================

with tab1:
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 📋 Job Description")
        
        # Preset Loader dropdown
        preset_selection = st.selectbox(
            "Load a job description preset:",
            options=list(SAMPLE_JDS.keys())
        )
        
        default_jd_text = SAMPLE_JDS[preset_selection] if preset_selection != "Select a preset..." else ""
        
        jd_input = st.text_area(
            "Paste Job Description text here:",
            value=default_jd_text,
            height=250,
            placeholder="Type or paste the job requirements..."
        )
        
    with col2:
        st.markdown("### 📄 Candidate Resume")
        
        uploaded_file = st.file_uploader(
            "Upload Resume (PDF, DOCX, or TXT):",
            type=["pdf", "docx", "txt"],
            key="single_file"
        )
        
        resume_text_input = st.text_area(
            "Or paste Resume text manually here:",
            height=180,
            placeholder="Paste raw resume text here if not uploading a file..."
        )

    st.markdown("---")
    
    # Run analysis
    if st.button("🚀 Run Analysis", key="btn_single"):
        # Resolve text source
        resume_text = ""
        candidate_name = "Candidate"
        
        if uploaded_file is not None:
            try:
                resume_text = extract_text_from_file(uploaded_file)
                candidate_name = uploaded_file.name
            except Exception as e:
                st.error(f"Error parsing uploaded file: {e}")
        elif resume_text_input.strip():
            resume_text = resume_text_input.strip()
            candidate_name = "Pasted Text Resume"
            
        if not jd_input.strip():
            st.warning("Please provide a Job Description.")
        elif not resume_text.strip():
            st.warning("Please provide a Resume (upload a file or paste text).")
        else:
            with st.spinner("Analyzing candidate profile against job description..."):
                try:
                    result = screener.screen_single(
                        job_description=jd_input,
                        resume_text=resume_text,
                        target_category=target_role
                    )
                    
                    st.success("Analysis complete!")
                    
                    # Display Results Columns
                    res_col1, res_col2 = st.columns([2, 3], gap="medium")
                    
                    with res_col1:
                        st.markdown("#### 📊 Overall Score")
                        score = result["overall_score"]
                        
                        # Plotly Gauge Chart for score
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=score * 100,
                            number={"suffix": "%", "font": {"size": 36, "family": "Outfit"}},
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Match Strength", 'font': {'size': 18, 'family': 'Outfit'}},
                            gauge={
                                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                                'bar': {'color': "#4f46e5"},
                                'bgcolor': "white",
                                'borderwidth': 2,
                                'bordercolor': "#e2e8f0",
                                'steps': [
                                    {'range': [0, 40], 'color': '#fee2e2'},
                                    {'range': [40, 70], 'color': '#fef3c7'},
                                    {'range': [70, 100], 'color': '#dcfce7'}
                                ],
                                'threshold': {
                                    'line': {'color': "green", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 70
                                }
                            }
                        ))
                        fig.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10))
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Fit strength card
                        if score >= 0.70:
                            st.markdown("<div class='status-banner status-success'>🟢 STRONG FIT Candidate</div>", unsafe_allow_html=True)
                        elif score >= 0.40:
                            st.markdown("<div class='status-banner status-info'>🟡 MODERATE FIT Candidate</div>", unsafe_allow_html=True)
                        else:
                            st.markdown("<div class='status-banner status-warning'>🔴 WEAK FIT Candidate</div>", unsafe_allow_html=True)
                            
                    with res_col2:
                        st.markdown("#### 🏷️ Classifier Insight")
                        pred_cat = result["category"]
                        
                        # Check match
                        if pred_cat.lower() == target_role.lower():
                            st.markdown(
                                f"<div class='status-banner status-success'>"
                                f"Predicted Category: <b>{pred_cat}</b><br>"
                                f"Matches Expected Category (Bonus Applied +{category_bonus*100:.0f}%)"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                f"<div class='status-banner status-info'>"
                                f"Predicted Category: <b>{pred_cat}</b><br>"
                                f"Expected Category: <i>{target_role}</i>"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                            
                        # Sub-metric boxes
                        sub1, sub2 = st.columns(2)
                        with sub1:
                            st.markdown(
                                f"<div class='metric-card'>"
                                f"<div class='metric-label'>Semantic Sim</div>"
                                f"<div class='metric-value'>{result['cosine_similarity']*100:.1f}%</div>"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                        with sub2:
                            st.markdown(
                                f"<div class='metric-card'>"
                                f"<div class='metric-label'>Skill Match</div>"
                                f"<div class='metric-value'>{result['skill_match_score']*100:.1f}%</div>"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                            
                    st.markdown("---")
                    st.markdown("#### 🛠️ Skill Extraction Breakdown")
                    
                    skill_col1, skill_col2 = st.columns(2)
                    
                    with skill_col1:
                        st.markdown(f"**Matched Skills** ({len(result['matched_skills'])}):")
                        if result["matched_skills"]:
                            pills_html = "".join([f"<span class='pill-green'>{s}</span>" for s in result["matched_skills"]])
                            st.markdown(pills_html, unsafe_allow_html=True)
                        else:
                            st.info("No matching skills found.")
                            
                    with skill_col2:
                        st.markdown(f"**Missing Skills** ({len(result['missing_skills'])}):")
                        if result["missing_skills"]:
                            pills_html = "".join([f"<span class='pill-red'>{s}</span>" for s in result["missing_skills"]])
                            st.markdown(pills_html, unsafe_allow_html=True)
                        else:
                            st.success("No missing skills! Excellent fit.")
                            
                    # Display extra skills
                    if result["extra_skills"]:
                        st.markdown("---")
                        st.markdown(f"**Additional Skills Extracted from Resume** ({len(result['extra_skills'])}):")
                        pills_html = "".join([f"<span class='pill-blue'>{s}</span>" for s in result["extra_skills"]])
                        st.markdown(pills_html, unsafe_allow_html=True)
                        
                except Exception as e:
                    st.error(f"Failed during screening process: {e}")
                    st.exception(e)

# ============================================================
# Tab 2: Batch Candidate Screening & Ranking
# ============================================================

with tab2:
    st.markdown("### 👥 Batch Candidate Screening & Ranking")
    st.markdown("Upload multiple resumes to screen, rank, and identify the best-matching candidates.")
    
    col_b1, col_b2 = st.columns([1, 1], gap="large")
    
    with col_b1:
        batch_jd = st.text_area(
            "Paste Job Description here:",
            value=default_jd_text,
            height=200,
            key="batch_jd",
            placeholder="Type or paste the job requirements for screening..."
        )
        
    with col_b2:
        batch_files = st.file_uploader(
            "Upload Multiple Resumes (PDF, DOCX, or TXT):",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            key="batch_files"
        )
        
    st.markdown("---")
    
    if st.button("🚀 Screen & Rank Batch", key="btn_batch"):
        if not batch_jd.strip():
            st.warning("Please provide a Job Description.")
        elif not batch_files:
            st.warning("Please upload one or more resume files.")
        else:
            with st.spinner(f"Processing and ranking {len(batch_files)} resumes..."):
                resumes = []
                resume_ids = []
                parse_errors = []
                
                # Parse all files
                for f in batch_files:
                    try:
                        text = extract_text_from_file(f)
                        if text.strip():
                            resumes.append(text)
                            resume_ids.append(f.name)
                        else:
                            parse_errors.append((f.name, "File contains no extractable text"))
                    except Exception as e:
                        parse_errors.append((f.name, str(e)))
                        
                if parse_errors:
                    st.warning(f"Failed to parse {len(parse_errors)} files:")
                    for fname, err in parse_errors:
                        st.write(f"- `{fname}`: {err}")
                        
                if not resumes:
                    st.error("No valid resumes could be parsed. Batch aborted.")
                else:
                    try:
                        # Screen batch
                        result = screener.screen_resumes(
                            job_description=batch_jd,
                            resumes=resumes,
                            resume_ids=resume_ids,
                            target_category=target_role
                        )
                        
                        # Build DataFrame for visualization
                        candidates_data = []
                        for c in result["ranked_candidates"]:
                            candidates_data.append({
                                "Rank": c["rank"],
                                "Candidate": c["candidate_id"],
                                "Predicted Category": c["category"],
                                "Overall Score": f"{c['overall_score']*100:.1f}%",
                                "Cosine Similarity": f"{c['cosine_similarity']*100:.1f}%",
                                "Skill Match Score": f"{c['skill_match_score']*100:.1f}%",
                                "Category Match": "Yes" if c["category_match"] else "No",
                                "Score_Raw": c["overall_score"] * 100,
                                "Matched Skills Count": len(c["matched_skills"])
                            })
                            
                        df_ranked = pd.DataFrame(candidates_data)
                        
                        # Split results layout
                        res_tab1, res_tab2 = st.columns([3, 2], gap="large")
                        
                        with res_tab1:
                            st.markdown("#### 🏆 Ranking Results")
                            # Render table without raw values
                            st.dataframe(
                                df_ranked.drop(columns=["Score_Raw"]),
                                use_container_width=True,
                                hide_index=True
                            )
                            
                            # CSV Export
                            csv_data = df_ranked.drop(columns=["Score_Raw"]).to_csv(index=False)
                            st.download_button(
                                label="📥 Export Rankings to CSV",
                                data=csv_data,
                                file_name="resume_screen_rankings.csv",
                                mime="text/csv"
                            )
                            
                        with res_tab2:
                            st.markdown("#### 📊 Comparative Chart")
                            
                            fig_bar = px.bar(
                                df_ranked,
                                x="Score_Raw",
                                y="Candidate",
                                orientation='h',
                                text="Overall Score",
                                labels={"Score_Raw": "Overall Match Score (%)", "Candidate": "Candidate"},
                                color="Score_Raw",
                                color_continuous_scale="Viridis",
                                title="Candidate Fit Score Comparison"
                            )
                            fig_bar.update_layout(
                                yaxis={'categoryorder':'total ascending'},
                                showlegend=False,
                                height=min(400, 150 + len(batch_files) * 35),
                                margin=dict(l=10, r=10, t=40, b=10)
                            )
                            st.plotly_chart(fig_bar, use_container_width=True)
                            
                        st.markdown("---")
                        # Skill gap analysis
                        st.markdown("#### 🔍 Common Skill Gaps in Candidate Pool")
                        st.markdown("These required skills were missing in the majority of candidates:")
                        
                        gaps = result["common_skill_gaps"]
                        if gaps:
                            cols = st.columns(min(4, len(gaps)))
                            for idx, gap in enumerate(gaps):
                                with cols[idx % len(cols)]:
                                    st.markdown(
                                        f"<div style='background-color:#fee2e2; border: 1px solid #fecaca; color:#991b1b; padding:0.6rem; border-radius:0.5rem; text-align:center; font-weight:600; margin-bottom:0.5rem;'>"
                                        f"❌ {gap}"
                                        f"</div>",
                                        unsafe_allow_html=True
                                    )
                        else:
                            st.success("No common skill gaps! The uploaded candidates match all JD skill requirements.")
                            
                    except Exception as e:
                        st.error(f"Failed to screen batch: {e}")
                        st.exception(e)

# ============================================================
# Tab 3: Database & Training Insights
# ============================================================

with tab3:
    st.markdown("### 📊 Database & Training Insights")
    
    col_d1, col_d2 = st.columns([1, 1], gap="large")
    
    with col_d1:
        st.markdown("#### 🎯 Classification Category Distribution")
        st.markdown("The underlying classifier is trained on the Kaggle Resume Dataset. Below is the distribution of the 24 categories:")
        
        # Load training report
        report_path = os.path.join(screener._config.paths.outputs_dir, "training_report.json")
        if os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)
            
            # Show stats
            st.markdown(
                f"- **Total Training Samples**: `{report['dataset']['total_samples']}`\n"
                f"- **Test Split Accuracy**: `{report['evaluation_metrics']['test_accuracy']*100:.2f}%`\n"
                f"- **Vocab Features (TF-IDF)**: `{report['tfidf']['vocabulary_size']}`"
            )
            
            # Hardcoded representation or read preprocessed file if exists
            processed_path = os.path.join(screener._config.paths.processed_data_dir, "preprocessed_resumes.csv")
            if os.path.exists(processed_path):
                df_proc = pd.read_csv(processed_path, usecols=["Category"])
                cat_counts = df_proc["Category"].value_counts().reset_index()
                cat_counts.columns = ["Category", "Count"]
                
                fig_pie = px.pie(
                    cat_counts,
                    values="Count",
                    names="Category",
                    title="Training Resume Categories",
                    color_discrete_sequence=px.colors.qualitative.Prism
                )
                fig_pie.update_layout(margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("Preprocessed resume counts chart not available (run train_pipeline.py first).")
        else:
            st.warning("Training report not found. Run training pipeline to generate insights.")
            
    with col_d2:
        st.markdown("#### 🛠️ Skill Database Management")
        st.markdown("Current count of keywords loaded from config files:")
        
        # Get count
        tech_skills = sorted(list(screener.skill_extractor.technical_skills))
        soft_skills = sorted(list(screener.skill_extractor.soft_skills))
        
        st.write(f"- **Technical Skills**: `{len(tech_skills)}` keywords loaded")
        st.write(f"- **Soft Skills**: `{len(soft_skills)}` keywords loaded")
        
        # Expanders to view skills
        with st.expander("🔍 View Technical Skills List"):
            st.write(", ".join(tech_skills))
            
        with st.expander("🔍 View Soft Skills List"):
            st.write(", ".join(soft_skills))
            
        # Tool to add a new skill to the text files
        st.markdown("##### ➕ Add Custom Skill keyword")
        new_skill = st.text_input("Enter new skill keyword (e.g. 'rust', 'prompt engineering'):").strip()
        skill_type = st.radio("Skill Type:", ["Technical", "Soft"], horizontal=True)
        
        if st.button("Add Skill to Database"):
            if not new_skill:
                st.warning("Please type a skill name.")
            else:
                # Resolve file
                filename = (
                    screener._config.skills.technical_skills_file 
                    if skill_type == "Technical" 
                    else screener._config.skills.soft_skills_file
                )
                filepath = os.path.join(screener._config.paths.skills_dir, filename)
                
                # Check duplication
                existing = (
                    screener.skill_extractor.technical_skills 
                    if skill_type == "Technical" 
                    else screener.skill_extractor.soft_skills
                )
                
                if new_skill.lower() in existing:
                    st.warning(f"'{new_skill}' already exists in the {skill_type} skill database!")
                else:
                    try:
                        with open(filepath, "a", encoding="utf-8") as file:
                            file.write(f"\n{new_skill}")
                        
                        # Reload screener in session state
                        st.cache_resource.clear()
                        st.success(f"Successfully added '{new_skill}' to {skill_type} database!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to update skill database file: {e}")

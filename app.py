"""AI Resume Matcher — an NLP mini-project built with Streamlit."""

import io
import re
from collections import Counter

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from docx import Document
except ImportError:
    Document = None


st.set_page_config(page_title="ResumeMatch AI", page_icon="🎯", layout="wide")

SKILLS = [
    "python", "java", "c++", "c#", "javascript", "typescript", "sql", "r", "matlab",
    "machine learning", "deep learning", "natural language processing", "nlp", "data analysis",
    "data visualization", "statistics", "tensorflow", "pytorch", "scikit-learn", "pandas",
    "numpy", "excel", "power bi", "tableau", "streamlit", "flask", "django", "fastapi",
    "html", "css", "react", "node.js", "git", "github", "docker", "aws", "azure",
    "google cloud", "linux", "mongodb", "postgresql", "mysql", "rest api", "api",
    "computer vision", "opencv", "keras", "communication", "teamwork", "leadership",
    "problem solving", "agile", "jira", "spark", "hadoop", "etl", "azure",
]

SAMPLE_RESUME = """AARAV SHARMA
Computer Science Student | aspiring Data Analyst

EDUCATION
B.Tech in Computer Science, 2023–2027

SKILLS
Python, SQL, Pandas, NumPy, scikit-learn, Excel, Power BI, Git, data visualization,
machine learning, statistics, communication, problem solving

PROJECTS
Built a student performance prediction model using Python and scikit-learn. Created
interactive dashboards in Power BI and cleaned datasets using Pandas.
"""

SAMPLE_JOB = """Data Analyst Intern

We are looking for an analytical intern to clean and analyse business data, build dashboards,
and communicate insights to stakeholders. Required skills: Python, SQL, Excel, Pandas,
data visualization, statistics, Power BI, and problem solving. Familiarity with Git and
machine learning is a plus.
"""


def extract_text(uploaded_file):
    """Return text from a supported uploaded resume."""
    suffix = uploaded_file.name.rsplit(".", 1)[-1].lower()
    data = uploaded_file.getvalue()
    if suffix == "txt":
        return data.decode("utf-8", errors="ignore")
    if suffix == "pdf" and PdfReader:
        return "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(data)).pages)
    if suffix == "docx" and Document:
        return "\n".join(p.text for p in Document(io.BytesIO(data)).paragraphs)
    raise ValueError("Could not read this file. Install the required parser or paste the resume text.")


def normalise(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def extract_skills(text):
    cleaned = normalise(text)
    found = []
    for skill in SKILLS:
        # Word boundaries avoid counting 'r' inside unrelated words.
        if re.search(r"(?<!\w)" + re.escape(skill) + r"(?!\w)", cleaned):
            found.append(skill)
    return sorted(set(found), key=str.lower)


def match_resume(resume, job):
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    matrix = vectorizer.fit_transform([resume, job])
    similarity = float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])
    resume_skills, job_skills = set(extract_skills(resume)), set(extract_skills(job))
    common = sorted(resume_skills & job_skills)
    missing = sorted(job_skills - resume_skills)
    skill_coverage = len(common) / len(job_skills) if job_skills else 0
    # A transparent blended score: textual relevance plus explicit skills coverage.
    score = round((similarity * 0.55 + skill_coverage * 0.45) * 100)
    return score, common, missing, sorted(job_skills), similarity


def tag_list(items, empty_message):
    if not items:
        st.caption(empty_message)
        return
    st.markdown(" ".join(f"<span class='tag'>{item}</span>" for item in items), unsafe_allow_html=True)


def load_demo_data():
    """Populate both text areas with the included sample documents."""
    st.session_state.resume_text = SAMPLE_RESUME
    st.session_state.job_text = SAMPLE_JOB


st.markdown("""
<style>
    .stApp { background: #f7f8fc; }
    .hero { padding: 1.4rem 0 1.1rem; }
    .hero h1 { color: #18233f; margin-bottom: .2rem; }
    .hero p { color: #5c6883; font-size: 1.05rem; }
    .tag { display: inline-block; background: #e8efff; color: #244b9b; border-radius: 999px;
           padding: .26rem .65rem; margin: .18rem .18rem .18rem 0; font-size: .88rem; }
    .missing .tag { background: #fff0ed; color: #ad3c31; }
    div[data-testid='stMetric'] { background: white; border: 1px solid #e5e9f2; padding: .8rem; border-radius: .75rem; }
</style>
<div class="hero">
  <h1>🎯 ResumeMatch AI</h1>
  <p>Compare a resume with a job description using NLP, TF–IDF and cosine similarity.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("How it works")
    st.markdown("1. Extract text from the resume\n2. Identify relevant skills\n3. Compare TF–IDF vectors\n4. Blend text relevance and skill coverage")
    st.button("Load demo data", on_click=load_demo_data, width="stretch")
    st.caption("Educational prototype — do not use as the only hiring decision tool.")

left, right = st.columns(2, gap="large")
with left:
    st.subheader("1. Candidate resume")
    uploaded = st.file_uploader("Upload PDF, DOCX or TXT", type=["pdf", "docx", "txt"])
    if uploaded:
        try:
            st.session_state.resume_text = extract_text(uploaded)
            st.success(f"Read {uploaded.name}")
        except Exception as error:
            st.error(str(error))
    resume = st.text_area("Or paste resume text", key="resume_text", height=330, placeholder="Paste the candidate's resume here...")

with right:
    st.subheader("2. Job description")
    job = st.text_area("Paste the job description", key="job_text", height=410, placeholder="Paste the role requirements here...")

analyse = st.button("Analyse match", type="primary", width="stretch")
if analyse:
    if len(resume.strip()) < 30 or len(job.strip()) < 30:
        st.warning("Please provide a fuller resume and job description (at least 30 characters each).")
    else:
        score, common, missing, job_skills, raw_similarity = match_resume(resume, job)
        st.divider()
        st.subheader("Match report")
        a, b, c = st.columns(3)
        a.metric("Overall match", f"{score}%")
        b.metric("Matched skills", len(common))
        c.metric("Job skills found", len(job_skills))
        st.progress(score / 100)
        if score >= 75:
            st.success("Strong alignment. The resume covers much of this role's language and skill set.")
        elif score >= 45:
            st.info("Promising foundation. Tailoring the resume to the missing requirements may improve the match.")
        else:
            st.warning("Limited alignment detected. Review the job requirements and add only genuine, relevant experience.")

        one, two = st.columns(2, gap="large")
        with one:
            st.markdown("#### ✅ Matching skills")
            tag_list(common, "No catalogue skills matched yet.")
        with two:
            st.markdown("#### 🎯 Skills to strengthen")
            st.markdown("<div class='missing'>", unsafe_allow_html=True)
            tag_list(missing, "Great — no missing skills from the detected job-skill list.")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("#### Suggested resume improvements")
        if missing:
            st.write("• If accurate, show projects, coursework, or experience demonstrating: " + ", ".join(missing[:5]) + ".")
        st.write("• Mirror relevant wording from the role in achievement-focused bullet points, supported by evidence.")
        st.caption(f"Text similarity before skill blending: {raw_similarity * 100:.1f}%. This score is an aid, not a verdict.")

st.divider()
st.caption("AI Lab Mini Project • NLP-based Resume Screening and Job Matching System")

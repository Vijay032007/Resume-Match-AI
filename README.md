# ResumeMatch AI

An AI lab mini-project that measures how well a candidate resume aligns with a job description. It uses Natural Language Processing (NLP), TF-IDF vectorisation, cosine similarity, and a transparent skills-coverage component.

## Features

- Upload a resume in PDF, DOCX, or TXT format, or paste its contents
- Paste any job description
- Overall match score based on text relevance and detected skill coverage
- Matched skills and missing skills
- Tailoring suggestions and a built-in demo dataset

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

## Methodology

1. Resume and job-description text are normalised.
2. Both documents are converted to TF-IDF vectors using unigrams and bigrams.
3. Cosine similarity estimates their textual relevance.
4. A controlled skills vocabulary identifies shared and missing skills.
5. The final score blends 55% textual similarity with 45% skill coverage.

## Suggested report structure

1. Problem statement: recruiters spend time comparing resumes against roles.
2. Objective: build an NLP prototype that provides an explainable match report.
3. Tools: Python, Streamlit, scikit-learn, PyPDF, python-docx.
4. Algorithm: TF-IDF and cosine similarity.
5. Results: test several resume/job pairs and compare their scores.
6. Limitations: the system can miss synonyms and must not be used as a sole hiring decision-maker.

## Project title

**AI-Based Resume Screening and Job Matching System Using NLP**

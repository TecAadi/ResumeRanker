# AI Resume Ranker

A Python + Streamlit project that ranks PDF resumes against a job description using NLP and cosine similarity.

## Features

- Upload one or many PDF resumes.
- Extract resume text using **pdfminer**.
- Paste a target job description.
- Clean and preprocess text with **nltk**.
- Compute similarity with **scikit-learn** TF-IDF + cosine similarity.
- Generate a resume match score (%).
- Rank multiple resumes from best to least match.
- Use a simple **Streamlit** web interface.

## Tech Stack

- `pandas`
- `scikit-learn`
- `nltk`
- `pdfminer.six`
- `streamlit`

## Project Structure

```text
ResumeRanker/
├── app.py
├── requirements.txt
├── README.md
└── src/
    └── resume_ranker/
        ├── __init__.py
        ├── pdf_utils.py
        ├── nlp_utils.py
        └── ranker.py
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run app.py
```

Then open the local URL printed by Streamlit (typically `http://localhost:8501`).

## How It Works

1. Upload PDF files in the app.
2. Resume text is extracted with `pdfminer.high_level.extract_text`.
3. Job description + resume text are normalized and stopwords are removed with NLTK.
4. A TF-IDF matrix is built over all documents.
5. Cosine similarity is calculated between the job description and each resume.
6. Results are sorted and shown as rank + percentage score.

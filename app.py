"""Streamlit app for AI resume ranking."""

from __future__ import annotations

import streamlit as st

from src.resume_ranker.pdf_utils import extract_text_from_pdf
from src.resume_ranker.ranker import ResumeRanker


st.set_page_config(page_title="AI Resume Ranker", page_icon="📄", layout="wide")

st.title("📄 AI Resume Ranker")
st.write(
    "Upload one or more PDF resumes, add a job description, and get ranked similarity scores."
)

uploaded_files = st.file_uploader(
    "Upload PDF resumes",
    type=["pdf"],
    accept_multiple_files=True,
)

job_description = st.text_area(
    "Paste the job description",
    height=220,
    placeholder="Enter required skills, responsibilities, and qualifications...",
)

if st.button("Rank Resumes", type="primary"):
    if not uploaded_files:
        st.error("Please upload at least one PDF resume.")
    elif not job_description.strip():
        st.error("Please provide a job description.")
    else:
        resumes: list[dict] = []
        parse_errors: list[str] = []

        with st.spinner("Extracting text and calculating scores..."):
            for uploaded_file in uploaded_files:
                try:
                    text = extract_text_from_pdf(uploaded_file.getvalue())
                    if text.strip():
                        resumes.append({"file_name": uploaded_file.name, "text": text})
                    else:
                        parse_errors.append(f"{uploaded_file.name}: No readable text found.")
                except Exception as exc:  # pragma: no cover - UI error reporting path
                    parse_errors.append(f"{uploaded_file.name}: {exc}")

            if resumes:
                ranker = ResumeRanker()
                ranked_df = ranker.rank(resumes=resumes, job_description=job_description)

                st.success(f"Ranked {len(resumes)} resume(s) successfully.")
                st.subheader("Ranking Results")
                st.dataframe(ranked_df, use_container_width=True)

                csv_data = ranked_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download ranking as CSV",
                    data=csv_data,
                    file_name="resume_ranking_results.csv",
                    mime="text/csv",
                )

                top_resume = ranked_df.iloc[0]
                st.info(
                    f"Top match: **{top_resume['File Name']}** with score **{top_resume['Resume Score (%)']}%**"
                )
            else:
                st.error("No valid resumes could be processed.")

        if parse_errors:
            st.warning("Some files could not be fully processed:")
            for error in parse_errors:
                st.write(f"- {error}")

with st.expander("How scoring works"):
    st.markdown(
        """
        - Resumes and job description are cleaned using NLP preprocessing.
        - Text is transformed with TF-IDF vectors (unigrams + bigrams).
        - Cosine similarity is used to compute the resume match score.
        - Score is displayed as a percentage for easy comparison.
        """
    )

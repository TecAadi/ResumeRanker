"""Resume ranking logic using TF-IDF and cosine similarity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .nlp_utils import preprocess_for_similarity


@dataclass
class RankedResume:
    file_name: str
    extracted_text: str
    similarity_score: float
    score_percent: float


class ResumeRanker:
    """Ranks resume texts against a job description."""

    def __init__(self, max_features: int = 4000):
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))

    def rank(self, resumes: List[dict], job_description: str) -> pd.DataFrame:
        """Return ranked resumes as a DataFrame.

        Args:
            resumes: List of dicts with keys `file_name` and `text`.
            job_description: Target job description text.
        """
        if not resumes:
            raise ValueError("At least one resume is required for ranking.")

        docs = [job_description] + [r["text"] for r in resumes]
        processed_docs = preprocess_for_similarity(docs)

        tfidf_matrix = self.vectorizer.fit_transform(processed_docs)
        job_vector = tfidf_matrix[0:1]
        resume_vectors = tfidf_matrix[1:]

        similarities = cosine_similarity(job_vector, resume_vectors).flatten()

        ranked_items = [
            RankedResume(
                file_name=resumes[idx]["file_name"],
                extracted_text=resumes[idx]["text"],
                similarity_score=float(score),
                score_percent=round(float(score) * 100, 2),
            )
            for idx, score in enumerate(similarities)
        ]

        ranked_items.sort(key=lambda item: item.similarity_score, reverse=True)

        return pd.DataFrame(
            [
                {
                    "Rank": rank,
                    "File Name": item.file_name,
                    "Resume Score (%)": item.score_percent,
                    "Similarity": round(item.similarity_score, 4),
                }
                for rank, item in enumerate(ranked_items, start=1)
            ]
        )

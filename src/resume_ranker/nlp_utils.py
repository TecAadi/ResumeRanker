"""NLP helpers for cleaning and vectorization."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable, List

import nltk
from nltk.corpus import stopwords


@lru_cache(maxsize=1)
def get_english_stopwords() -> set[str]:
    """Load NLTK English stopwords, downloading once if needed."""
    try:
        return set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        return set(stopwords.words("english"))



def normalize_text(text: str) -> str:
    """Lowercase and remove punctuation/numbers for robust similarity."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text



def preprocess_for_similarity(texts: Iterable[str]) -> List[str]:
    """Normalize and remove stopwords from each text item."""
    stop_words = get_english_stopwords()
    processed: list[str] = []

    for text in texts:
        normalized = normalize_text(text)
        tokens = [token for token in normalized.split() if token not in stop_words]
        processed.append(" ".join(tokens))

    return processed

"""
similarity_engine.py
====================
Retrieves the most relevant dataset response for a given user query.

Algorithm
---------
1. At startup, the engine vectorises all dataset questions using TF-IDF
   and caches the result matrix.

2. For each user query the engine:
   a) Computes TF-IDF cosine similarity between the query and all questions.
   b) Computes keyword overlap score between the query tokens and each
      row's keyword list.
   c) Applies a small emotion-match bonus when the detected emotion aligns
      with the dataset row's emotion label.
   d) Combines the three scores with configurable weights into a final score.
   e) Returns the top-N best matches.

The cached TF-IDF matrix makes repeated queries fast after the initial fit.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# ── Score weights (must sum to ≤ 1.0; emotion bonus is additive) ───────────────
WEIGHT_TFIDF   = 0.65   # TF-IDF cosine similarity
WEIGHT_KEYWORD = 0.35   # keyword overlap
EMOTION_BONUS  = 0.10   # bonus if emotion matches

# Minimum combined score to return a result (below this → fallback response)
MIN_SCORE_THRESHOLD = 0.05


@dataclass
class MatchResult:
    """A single retrieved dataset entry with its relevance score."""
    response: str
    emotion:  str
    intent:   str
    score:    float
    question: str   # the matched dataset question (for debugging)
    keyword_score: float = 0.0
    tfidf_score:   float = 0.0


class SimilarityEngine:
    """
    TF-IDF + cosine similarity engine with keyword-overlap boosting.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset loaded by DataLoader.  Must contain columns:
        question, response, emotion, intent, keywords.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df.reset_index(drop=True)
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._tfidf_matrix = None   # shape: (n_docs, n_features)
        self._is_fitted = False
        self._fit()

    # ── Public API ─────────────────────────────────────────────────────────────

    def query(
        self,
        clean_text: str,
        tokens: list[str],
        detected_emotion: str = "",
        top_k: int = 3,
    ) -> list[MatchResult]:
        """
        Retrieve the best matching responses for *clean_text*.

        Parameters
        ----------
        clean_text : str
            Preprocessed query text (used for TF-IDF).
        tokens : list[str]
            Tokenised query (used for keyword overlap).
        detected_emotion : str
            Emotion detected upstream (used for bonus scoring).
        top_k : int
            Number of candidates to return.

        Returns
        -------
        list[MatchResult]
            Sorted by descending score; may be empty if nothing meets the threshold.
        """
        if not self._is_fitted:
            logger.error("Engine not fitted — cannot query.")
            return []

        tfidf_scores   = self._tfidf_similarity(clean_text)
        keyword_scores = self._keyword_overlap(tokens)
        emotion_bonus  = self._emotion_bonus(detected_emotion)

        # Combine scores
        combined = (
            WEIGHT_TFIDF   * tfidf_scores +
            WEIGHT_KEYWORD * keyword_scores +
            EMOTION_BONUS  * emotion_bonus
        )

        # Sort descending and take top_k above threshold
        top_indices = np.argsort(combined)[::-1][:top_k]
        results: list[MatchResult] = []

        for idx in top_indices:
            score = float(combined[idx])
            if score < MIN_SCORE_THRESHOLD:
                continue
            row = self._df.iloc[idx]
            results.append(
                MatchResult(
                    response=row["response"],
                    emotion=row["emotion"],
                    intent=row["intent"],
                    score=round(score, 4),
                    question=row["question"],
                    keyword_score=round(float(keyword_scores[idx]), 4),
                    tfidf_score=round(float(tfidf_scores[idx]), 4),
                )
            )

        logger.debug(f"Top match score: {results[0].score if results else 'N/A'}")
        return results

    def best_response(
        self,
        clean_text: str,
        tokens: list[str],
        detected_emotion: str = "",
    ) -> Optional[MatchResult]:
        """Return only the single best match, or None if no good match found."""
        results = self.query(clean_text, tokens, detected_emotion, top_k=1)
        return results[0] if results else None

    # ── Private: fitting ───────────────────────────────────────────────────────

    def _fit(self) -> None:
        """Vectorise all dataset questions with TF-IDF and cache the matrix."""
        logger.info("Fitting TF-IDF vectoriser on dataset questions …")
        questions = self._df["question"].fillna("").tolist()

        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),          # unigrams + bigrams
            max_features=10_000,
            sublinear_tf=True,           # log-scaling dampens common terms
            min_df=1,
            analyzer="word",
        )
        self._tfidf_matrix = self._vectorizer.fit_transform(questions)
        self._is_fitted = True
        logger.info(
            f"TF-IDF matrix cached: "
            f"{self._tfidf_matrix.shape[0]} docs × "
            f"{self._tfidf_matrix.shape[1]} features."
        )

    # ── Private: scoring ───────────────────────────────────────────────────────

    def _tfidf_similarity(self, clean_text: str) -> np.ndarray:
        """Compute cosine similarity between *clean_text* and all dataset questions."""
        query_vec = self._vectorizer.transform([clean_text])
        scores = cosine_similarity(query_vec, self._tfidf_matrix).flatten()
        return scores  # shape: (n_docs,)

    def _keyword_overlap(self, tokens: list[str]) -> np.ndarray:
        """
        For each dataset row compute:
            |query_tokens ∩ row_keywords| / max(|row_keywords|, 1)
        Returns a normalised array in [0, 1].
        """
        token_set = set(tokens)
        scores = np.zeros(len(self._df), dtype=float)

        for i, kw_list in enumerate(self._df["keywords"]):
            if not isinstance(kw_list, list) or not kw_list:
                continue
            kw_set = set(kw_list)
            overlap = len(token_set & kw_set)
            scores[i] = overlap / len(kw_set)

        return scores

    def _emotion_bonus(self, detected_emotion: str) -> np.ndarray:
        """Binary array: 1.0 where row emotion == detected_emotion, else 0.0."""
        if not detected_emotion or detected_emotion in ("unknown", "neutral"):
            return np.zeros(len(self._df), dtype=float)
        return (self._df["emotion"] == detected_emotion).astype(float).values


# ── Quick self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_loader import DataLoader
    from preprocess import preprocess

    df = DataLoader().load()
    engine = SimilarityEngine(df)

    queries = [
        "I can't stop feeling anxious and worried about everything",
        "I feel so lonely and no one understands me",
        "I'm completely burned out from work",
    ]
    for q in queries:
        p = preprocess(q)
        match = engine.best_response(p.clean, p.tokens)
        if match:
            print(f"Query   : {q}")
            print(f"Matched : {match.question}")
            print(f"Score   : {match.score}  (tfidf={match.tfidf_score}, kw={match.keyword_score})")
            print(f"Response: {match.response[:120]}…")
            print()

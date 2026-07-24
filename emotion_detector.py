"""
emotion_detector.py
===================
Detects the dominant emotion in a piece of text.

Three detection strategies are combined for robustness:
  1. Keyword dictionary matching  (fast, transparent, tunable)
  2. Dataset label lookup         (uses emotion labels from matched responses)
  3. Lightweight ML classifier    (Naive Bayes on TF-IDF, trained from dataset)

The final emotion is chosen by a simple voting / priority scheme.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Emotion keyword dictionaries ───────────────────────────────────────────────
# Each key is an emotion label; the value is a set of indicator words/phrases.
# More specific phrases take priority over single words (checked first).

EMOTION_KEYWORDS: dict[str, list[str]] = {
    "anxiety": [
        "anxious", "anxiety", "panic", "panic attack", "nervous", "worry", "worried",
        "fearful", "fear", "tense", "uneasy", "restless", "dread", "on edge",
        "can't stop worrying", "what if", "overthinking", "racing mind",
        "heart pounding", "can't breathe", "chest tight",
    ],
    "sadness": [
        "sad", "sadness", "unhappy", "depressed", "depression", "miserable",
        "down", "blue", "hopeless", "hopelessness", "empty", "numb", "tearful",
        "crying", "cried", "grief", "grieving", "loss", "lost", "heartbroken",
        "broken", "lonely", "alone", "isolated", "worthless", "meaningless",
        "no purpose", "can't feel anything",
    ],
    "anger": [
        "angry", "anger", "furious", "rage", "frustrated", "frustration",
        "irritated", "irritable", "annoyed", "mad", "livid", "fuming",
        "hate", "resentful", "resentment", "bitter", "outraged",
    ],
    "stress": [
        "stressed", "stress", "overwhelmed", "overwhelm", "burned out", "burnout",
        "exhausted", "exhaustion", "too much", "overloaded", "pressure",
        "can't cope", "cannot cope", "falling apart", "breaking down",
        "no energy", "drained", "depleted", "running on empty",
    ],
    "loneliness": [
        "lonely", "loneliness", "alone", "isolation", "isolated", "no friends",
        "nobody cares", "no one listens", "misunderstood", "disconnected",
        "no connection", "withdrawn", "cut off", "invisible",
    ],
    "happiness": [
        "happy", "happiness", "joyful", "joy", "excited", "excited",
        "grateful", "thankful", "good mood", "feeling good", "great",
        "wonderful", "content", "peaceful", "calm", "hopeful", "optimistic",
        "positive", "better today",
    ],
    "guilt": [
        "guilty", "guilt", "regret", "regretful", "ashamed", "shame",
        "can't forgive", "made a mistake", "did something wrong", "my fault",
        "blame myself", "I'm sorry", "I screwed up",
    ],
    "confusion": [
        "confused", "confusion", "lost", "don't understand", "don't know what to do",
        "unsure", "uncertain", "don't know", "can't figure out", "don't know why",
    ],
}

# Ordered from most specific to least so longer phrases match first
_SORTED_EMOTIONS: list[tuple[str, list[str]]] = [
    (emotion, sorted(keywords, key=len, reverse=True))
    for emotion, keywords in EMOTION_KEYWORDS.items()
]

# Neutral fallback
NEUTRAL_EMOTION = "neutral"
UNKNOWN_EMOTION = "unknown"


@dataclass
class EmotionResult:
    """The result of emotion detection."""
    emotion: str                        # dominant detected emotion
    confidence: float                   # 0.0 – 1.0
    all_detected: list[str] = field(default_factory=list)  # all emotions found
    method: str = "keyword"             # which method produced the final answer


class EmotionDetector:
    """
    Multi-strategy emotion detector.

    Parameters
    ----------
    dataset_df : pd.DataFrame, optional
        If provided, the detector will also train a Naive Bayes classifier
        on the dataset's (question, emotion) pairs for ML-based detection.
    """

    def __init__(self, dataset_df=None) -> None:
        self._ml_classifier = None
        self._vectorizer = None
        self._label_map: dict[int, str] = {}

        if dataset_df is not None:
            self._train_ml_classifier(dataset_df)

    # ── Public API ─────────────────────────────────────────────────────────────

    def detect(self, text: str, tokens: Optional[list[str]] = None) -> EmotionResult:
        """
        Detect the dominant emotion in *text*.

        Parameters
        ----------
        text : str
            Cleaned / preprocessed text.
        tokens : list[str], optional
            Pre-tokenised form (used for keyword matching).

        Returns
        -------
        EmotionResult
        """
        text_lower = text.lower()
        tokens_set = set(tokens or text_lower.split())

        # Strategy 1: keyword matching
        kw_result = self._keyword_detect(text_lower, tokens_set)

        # Strategy 2: ML classifier (if available)
        ml_result = self._ml_detect(text) if self._ml_classifier is not None else None

        # Combine: keyword wins if confident; otherwise fall back to ML
        if kw_result.emotion != NEUTRAL_EMOTION and kw_result.confidence >= 0.4:
            return kw_result
        if ml_result is not None and ml_result.confidence >= 0.5:
            return ml_result
        if kw_result.emotion != NEUTRAL_EMOTION:
            return kw_result

        return EmotionResult(emotion=NEUTRAL_EMOTION, confidence=0.0, method="fallback")

    def describe(self, emotion: str) -> str:
        """Return a short human-readable description of an emotion."""
        descriptions = {
            "anxiety":   "feelings of worry, nervousness, or unease",
            "sadness":   "feelings of unhappiness, loss, or low mood",
            "anger":     "feelings of frustration, rage, or irritation",
            "stress":    "feelings of overwhelm, pressure, or exhaustion",
            "loneliness":"feelings of isolation or disconnection",
            "happiness": "feelings of joy, contentment, or positivity",
            "guilt":     "feelings of regret, shame, or self-blame",
            "confusion": "feelings of uncertainty or not knowing what to do",
            NEUTRAL_EMOTION: "a neutral emotional state",
            UNKNOWN_EMOTION: "an unclear emotional state",
        }
        return descriptions.get(emotion, "a complex emotional experience")

    # ── Strategy 1: keyword matching ───────────────────────────────────────────

    @staticmethod
    def _keyword_detect(text_lower: str, tokens_set: set) -> EmotionResult:
        """
        Check how many indicator keywords appear in the text for each emotion.
        Returns the emotion with the highest hit count, normalised as confidence.
        """
        scores: dict[str, int] = {e: 0 for e in EMOTION_KEYWORDS}
        all_detected: list[str] = []

        for emotion, keywords in _SORTED_EMOTIONS:
            for kw in keywords:
                # Multi-word phrases: substring match in full text
                # Single words: token-level match (more precise)
                if " " in kw:
                    if kw in text_lower:
                        scores[emotion] += 2   # phrase match is worth more
                elif kw in tokens_set:
                    scores[emotion] += 1

            if scores[emotion] > 0:
                all_detected.append(emotion)

        if not any(scores.values()):
            return EmotionResult(emotion=NEUTRAL_EMOTION, confidence=0.0)

        dominant = max(scores, key=lambda e: scores[e])
        total    = sum(scores.values()) or 1
        confidence = min(scores[dominant] / total, 1.0)

        return EmotionResult(
            emotion=dominant,
            confidence=round(confidence, 3),
            all_detected=all_detected,
            method="keyword",
        )

    # ── Strategy 2: ML classifier ──────────────────────────────────────────────

    def _train_ml_classifier(self, df) -> None:
        """
        Train a Multinomial Naive Bayes classifier on the dataset.
        Silently skips if sklearn is not available.
        """
        try:
            from sklearn.naive_bayes import MultinomialNB
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.preprocessing import LabelEncoder

            valid = df[df["emotion"].notna() & df["emotion"].ne("unknown")]
            if len(valid) < 5:
                logger.debug("Not enough labelled data to train ML classifier.")
                return

            le = LabelEncoder()
            y  = le.fit_transform(valid["emotion"].values)
            self._label_map = dict(enumerate(le.classes_))

            self._vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
            X = self._vectorizer.fit_transform(valid["question"].values)

            self._ml_classifier = MultinomialNB()
            self._ml_classifier.fit(X, y)
            logger.info(
                f"Emotion ML classifier trained on {len(valid)} samples "
                f"({len(self._label_map)} classes)."
            )

        except Exception as exc:
            logger.warning(f"Could not train ML emotion classifier: {exc}")

    def _ml_detect(self, text: str) -> Optional[EmotionResult]:
        """Run the trained Naive Bayes classifier on *text*."""
        try:
            X = self._vectorizer.transform([text])
            proba = self._ml_classifier.predict_proba(X)[0]
            best_idx = int(np.argmax(proba))
            emotion  = self._label_map.get(best_idx, UNKNOWN_EMOTION)
            confidence = float(proba[best_idx])
            return EmotionResult(emotion=emotion, confidence=round(confidence, 3), method="ml")
        except Exception:
            return None


# ── Module-level convenience ────────────────────────────────────────────────────
_detector: Optional[EmotionDetector] = None


def get_detector(dataset_df=None) -> EmotionDetector:
    """Return a module-level singleton EmotionDetector."""
    global _detector
    if _detector is None:
        _detector = EmotionDetector(dataset_df)
    return _detector


# ── Quick self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    det = EmotionDetector()
    tests = [
        "I'm feeling really anxious and having panic attacks",
        "I feel so sad and hopeless about everything",
        "I'm so angry and frustrated I can't stand it",
        "I feel burned out and completely exhausted",
        "I'm lonely and nobody understands me",
        "I'm feeling happy and grateful today",
        "I feel guilty about something I did",
    ]
    for t in tests:
        result = det.detect(t)
        print(f"Text    : {t}")
        print(f"Emotion : {result.emotion}  (confidence={result.confidence}, method={result.method})")
        print()

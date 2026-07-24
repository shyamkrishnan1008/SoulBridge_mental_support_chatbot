"""
preprocess.py
=============
Text cleaning and normalisation utilities used by the chatbot pipeline.

Steps applied to both user input and dataset questions:
  1. Lowercase
  2. Remove URLs, emails, special characters
  3. Expand common contractions (don't → do not, etc.)
  4. Tokenise
  5. Remove stopwords  (with a custom mental-health aware keep-list)
  6. Lemmatise  (WordNetLemmatizer)
  7. Rejoin into a clean string

A ProcessedText namedtuple is returned so callers can use either the
cleaned string (for TF-IDF) or the token list (for keyword matching).
"""

import re
import logging
from typing import NamedTuple

import nltk

# ── Download required NLTK resources once ────────────────────────────────────
for resource in ("punkt", "stopwords", "wordnet", "omw-1.4", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

logger = logging.getLogger(__name__)

# ── Contraction map ────────────────────────────────────────────────────────────
CONTRACTIONS: dict[str, str] = {
    "i'm": "i am", "i've": "i have", "i'll": "i will", "i'd": "i would",
    "you're": "you are", "you've": "you have", "you'll": "you will", "you'd": "you would",
    "he's": "he is", "she's": "she is", "it's": "it is", "we're": "we are",
    "we've": "we have", "we'll": "we will", "they're": "they are", "they've": "they have",
    "they'll": "they will", "don't": "do not", "doesn't": "does not",
    "didn't": "did not", "won't": "will not", "wouldn't": "would not",
    "can't": "cannot", "couldn't": "could not", "shouldn't": "should not",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "haven't": "have not", "hasn't": "has not",
    "hadn't": "had not", "that's": "that is", "what's": "what is",
    "there's": "there is", "here's": "here is", "who's": "who is",
    "how's": "how is", "let's": "let us", "it'd": "it would",
}

# ── Words that are meaningful in mental health context (do NOT remove) ─────────
KEEP_WORDS: frozenset[str] = frozenset({
    "not", "no", "never", "nothing", "nobody", "nowhere", "neither",
    "alone", "lonely", "sad", "happy", "angry", "fear", "scared",
    "worthless", "hopeless", "helpless", "empty", "numb",
    "hurt", "pain", "suffer", "loss", "grief", "die", "death",
    "tired", "exhausted", "anxious", "worry", "stress", "depressed",
    "feel", "feeling", "felt", "emotion", "mood",
})

# Standard English stopwords minus our keep-words
_BASE_STOPWORDS = set(stopwords.words("english")) - KEEP_WORDS


class ProcessedText(NamedTuple):
    """Holds both the cleaned string and individual tokens."""
    clean: str          # cleaned, lemmatised text (for TF-IDF)
    tokens: list[str]   # individual tokens (for keyword matching)
    original: str       # original input (for display)


class TextPreprocessor:
    """
    Stateless text preprocessor.  Create one instance and reuse it.
    """

    def __init__(self, remove_stopwords: bool = True, lemmatise: bool = True) -> None:
        self.remove_stopwords = remove_stopwords
        self.lemmatise = lemmatise
        self._lemmatizer = WordNetLemmatizer()

    # ── Public API ─────────────────────────────────────────────────────────────

    def process(self, text: str) -> ProcessedText:
        """
        Full preprocessing pipeline.

        Parameters
        ----------
        text : str
            Raw user or dataset text.

        Returns
        -------
        ProcessedText
            Named tuple with .clean, .tokens, and .original fields.
        """
        original = text.strip()
        text = self._lowercase(text)
        text = self._remove_urls(text)
        text = self._expand_contractions(text)
        text = self._remove_special_chars(text)
        tokens = self._tokenise(text)
        if self.remove_stopwords:
            tokens = self._remove_stopwords(tokens)
        if self.lemmatise:
            tokens = self._lemmatise(tokens)
        tokens = [t for t in tokens if len(t) > 1]   # drop single chars
        clean = " ".join(tokens)
        return ProcessedText(clean=clean, tokens=tokens, original=original)

    def process_series(self, series) -> list[ProcessedText]:
        """
        Process a pandas Series of strings and return a list of ProcessedText.
        Useful for batch-processing the dataset questions.
        """
        return [self.process(str(text)) for text in series]

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _lowercase(text: str) -> str:
        return text.lower()

    @staticmethod
    def _remove_urls(text: str) -> str:
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = re.sub(r"\S+@\S+\.\S+", " ", text)   # emails
        return text

    @staticmethod
    def _expand_contractions(text: str) -> str:
        pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in CONTRACTIONS) + r")\b")
        return pattern.sub(lambda m: CONTRACTIONS[m.group()], text)

    @staticmethod
    def _remove_special_chars(text: str) -> str:
        # Keep alphanumeric and spaces; replace everything else with a space
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text)   # collapse multiple spaces
        return text.strip()

    @staticmethod
    def _tokenise(text: str) -> list[str]:
        return word_tokenize(text)

    @staticmethod
    def _remove_stopwords(tokens: list[str]) -> list[str]:
        return [t for t in tokens if t not in _BASE_STOPWORDS or t in KEEP_WORDS]

    def _lemmatise(self, tokens: list[str]) -> list[str]:
        return [self._lemmatizer.lemmatize(t) for t in tokens]


# ── Module-level convenience instance ─────────────────────────────────────────
preprocessor = TextPreprocessor()


def preprocess(text: str) -> ProcessedText:
    """Module-level shortcut for TextPreprocessor().process(text)."""
    return preprocessor.process(text)


# ── Quick self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    samples = [
        "I'm feeling really anxious and I can't sleep at night!",
        "Don't know why I feel so sad and lonely all the time.",
        "I've been stressed out about work and it's overwhelming me.",
    ]
    for s in samples:
        result = preprocess(s)
        print(f"Original : {result.original}")
        print(f"Clean    : {result.clean}")
        print(f"Tokens   : {result.tokens}")
        print()

"""
data_loader.py
==============
Responsible for loading and parsing the mental health JSON dataset.

Supports multiple Kaggle dataset formats:
  - Format A: {"conversations": [{question, emotion, intent, response, keywords}, ...]}
  - Format B: Flat list of dialogue pairs [{question, answer}, ...]
  - Format C: Counseling format [{context, response}, ...]
  - Format D: Empathetic dialogues [{utterance, emotion, response}, ...]

The loader normalises every format into a consistent internal DataFrame
so the rest of the pipeline doesn't need to care about the source format.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
DEFAULT_DATASET_PATH = Path(__file__).parent / "data" / "mental_health_dataset.json"

# Column names used throughout the project
COL_ID       = "id"
COL_QUESTION = "question"
COL_EMOTION  = "emotion"
COL_INTENT   = "intent"
COL_RESPONSE = "response"
COL_KEYWORDS = "keywords"

REQUIRED_COLUMNS = [COL_QUESTION, COL_RESPONSE]


# ── Main loader class ──────────────────────────────────────────────────────────
class DataLoader:
    """
    Loads a JSON mental-health dataset and exposes it as a pandas DataFrame.

    Usage
    -----
    >>> loader = DataLoader("data/mental_health_dataset.json")
    >>> df = loader.load()
    >>> print(df.shape)
    """

    def __init__(self, filepath: Optional[str] = None) -> None:
        self.filepath = Path(filepath) if filepath else DEFAULT_DATASET_PATH
        self._df: Optional[pd.DataFrame] = None  # cached result

    # ── Public API ─────────────────────────────────────────────────────────────

    def load(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Load the dataset.  Returns a cached copy on subsequent calls
        unless *force_reload* is True.
        """
        if self._df is not None and not force_reload:
            return self._df

        if not self.filepath.exists():
            raise FileNotFoundError(
                f"Dataset not found at '{self.filepath}'. "
                "Please place your Kaggle JSON file in the data/ folder."
            )

        logger.info(f"Loading dataset from: {self.filepath}")
        raw = self._read_json()
        self._df = self._normalise(raw)
        logger.info(f"Dataset loaded successfully — {len(self._df)} entries.")
        return self._df

    def get_crisis_responses(self) -> dict:
        """Return the crisis response block if it exists in the dataset."""
        raw = self._read_json()
        return raw.get("crisis_responses", {})

    # ── Private helpers ────────────────────────────────────────────────────────

    def _read_json(self) -> dict | list:
        """Read raw JSON from disk."""
        with open(self.filepath, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _normalise(self, raw: dict | list) -> pd.DataFrame:
        """
        Detect the dataset format and convert it to the canonical schema:

            id | question | emotion | intent | response | keywords

        Any missing optional columns are filled with sensible defaults.
        """
        # ── Format A: our canonical format ────────────────────────────────────
        if isinstance(raw, dict) and "conversations" in raw:
            logger.info("Detected Format A (canonical conversations list).")
            records = raw["conversations"]
            df = pd.DataFrame(records)

        # ── Format B: flat list of dicts ──────────────────────────────────────
        elif isinstance(raw, list):
            logger.info("Detected Format B (flat list).")
            df = pd.DataFrame(raw)
            # Map common alternative column names
            df = df.rename(columns={
                "answer": COL_RESPONSE,
                "context": COL_QUESTION,
                "utterance": COL_QUESTION,
                "text": COL_QUESTION,
                "label": COL_EMOTION,
                "tag": COL_INTENT,
            })

        # ── Format C: dict with a data key ────────────────────────────────────
        elif isinstance(raw, dict) and "data" in raw:
            logger.info("Detected Format C (dict with 'data' key).")
            df = pd.DataFrame(raw["data"])
            df = df.rename(columns={"answer": COL_RESPONSE, "context": COL_QUESTION})

        # ── Format D: dict with intents (Rasa-style) ──────────────────────────
        elif isinstance(raw, dict) and "intents" in raw:
            logger.info("Detected Format D (Rasa intents format).")
            records = []
            for intent_block in raw["intents"]:
                intent_name = intent_block.get("tag", "unknown")
                patterns = intent_block.get("patterns", [])
                responses = intent_block.get("responses", ["I'm here for you."])
                for pattern in patterns:
                    records.append({
                        COL_QUESTION: pattern,
                        COL_INTENT:   intent_name,
                        COL_RESPONSE: responses[0],
                    })
            df = pd.DataFrame(records)

        else:
            raise ValueError(
                "Unrecognised JSON format. "
                "Supported: conversations list, flat list, dict with 'data' key, "
                "or Rasa-style intents dict."
            )

        # ── Validate required columns ──────────────────────────────────────────
        for col in REQUIRED_COLUMNS:
            if col not in df.columns:
                raise ValueError(
                    f"Dataset is missing required column '{col}'. "
                    f"Found columns: {list(df.columns)}"
                )

        # ── Fill optional columns with defaults ────────────────────────────────
        df[COL_EMOTION]  = df.get(COL_EMOTION,  pd.Series(dtype=str)).fillna("unknown")
        df[COL_INTENT]   = df.get(COL_INTENT,   pd.Series(dtype=str)).fillna("general")
        df[COL_KEYWORDS] = df.get(COL_KEYWORDS, pd.Series(dtype=object)).apply(
            lambda x: x if isinstance(x, list) else []
        )

        # Add numeric ID if missing
        if COL_ID not in df.columns:
            df.insert(0, COL_ID, range(1, len(df) + 1))

        # Drop rows with empty question or response
        before = len(df)
        df = df.dropna(subset=REQUIRED_COLUMNS)
        df = df[df[COL_QUESTION].str.strip().ne("") & df[COL_RESPONSE].str.strip().ne("")]
        df = df.reset_index(drop=True)
        dropped = before - len(df)
        if dropped:
            logger.warning(f"Dropped {dropped} rows with empty question/response.")

        return df[
            [COL_ID, COL_QUESTION, COL_EMOTION, COL_INTENT, COL_RESPONSE, COL_KEYWORDS]
        ]


# ── Quick self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    loader = DataLoader()
    df = loader.load()
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Emotions: {df['emotion'].unique()}")

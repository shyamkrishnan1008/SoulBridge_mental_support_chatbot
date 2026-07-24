"""
safety_checker.py
=================
Detects crisis signals in user input and overrides normal responses
with emergency guidance.

Crisis levels
-------------
  LEVEL_3 (Critical): Explicit statements of intent to harm self or others
  LEVEL_2 (High):     Expressions of hopelessness, not wanting to be alive
  LEVEL_1 (Moderate): Indirect distress signals that may indicate crisis

The checker always errs on the side of caution.
"""

import re
import logging
from dataclasses import dataclass
from enum import IntEnum

logger = logging.getLogger(__name__)


# ── Crisis severity levels ─────────────────────────────────────────────────────
class CrisisLevel(IntEnum):
    NONE     = 0
    MODERATE = 1   # Indirect signals
    HIGH     = 2   # Explicit hopelessness / passive suicidal ideation
    CRITICAL = 3   # Active suicidal/self-harm intent


# ── Keyword/phrase sets per level ─────────────────────────────────────────────
# Phrases are checked as substrings in lowercased text (after basic cleaning).
# More specific phrases are listed first to improve match accuracy.

CRITICAL_PHRASES: list[str] = [
    "kill myself", "killing myself", "end my life", "take my life",
    "want to die", "going to die", "planning to die",
    "cut myself", "cutting myself", "hurt myself", "hurting myself",
    "self harm", "self-harm", "self-harming",
    "overdose", "take pills", "hang myself", "jump off",
    "shoot myself", "slit my wrists", "not want to be here anymore",
    "don't want to be alive", "don't want to live", "no reason to live",
    "life is not worth living", "life isn't worth living",
]

HIGH_PHRASES: list[str] = [
    "suicidal", "suicide", "suicidal thoughts", "suicidal ideation",
    "thinking about dying", "thought about ending it",
    "everyone would be better off without me",
    "better off dead", "wish i was dead", "wish i were dead",
    "don't want to wake up", "hope i don't wake up",
    "tired of living", "tired of being alive", "can't go on",
    "nothing to live for", "see no point in living",
    "want to disappear forever",
]

MODERATE_PHRASES: list[str] = [
    "feel like giving up", "given up", "given up on life",
    "can't take it anymore", "cannot take it anymore",
    "i'm at my limit", "reached my limit", "breaking point",
    "don't want to feel anything", "want to feel nothing",
    "feel completely hopeless", "utterly hopeless",
    "there's no way out", "there is no way out",
    "no escape", "trapped", "i see no future",
]

# Pre-compiled patterns for efficiency
def _compile(phrases: list[str]) -> re.Pattern:
    escaped = [re.escape(p) for p in sorted(phrases, key=len, reverse=True)]
    return re.compile(r"\b(?:" + "|".join(escaped) + r")\b", re.IGNORECASE)

_PATTERN_CRITICAL = _compile(CRITICAL_PHRASES)
_PATTERN_HIGH     = _compile(HIGH_PHRASES)
_PATTERN_MODERATE = _compile(MODERATE_PHRASES)


@dataclass
class SafetyResult:
    """The outcome of a safety check."""
    is_crisis: bool
    level: CrisisLevel
    matched_phrases: list[str]
    response: str


# ── Response templates ──────────────────────────────────────────────────────────
_RESPONSE_CRITICAL = """\
🚨 I'm deeply concerned about what you've shared, and I want you to know that your life matters immensely.

Please reach out to a crisis helpline RIGHT NOW — they are trained to help and they care:

  🇮🇳 iCall (India):              9152987821
  🇮🇳 Vandrevala Foundation:      1860-2662-345  (24 hours, free)
  🌍 International Directory:     https://www.iasp.info/resources/Crisis_Centres/
  💬 Crisis Text Line (global):   Text HOME to 741741

If you are in immediate danger, please call emergency services (112 in India) or go to your nearest hospital.

You are not alone. Please reach out to someone you trust right now. I'm here with you — would you be willing to call one of these numbers?"""

_RESPONSE_HIGH = """\
💙 What you're sharing tells me you're in a lot of pain, and I'm really glad you reached out.

These feelings are serious, and you deserve real support right now. Please talk to someone who can truly help:

  🇮🇳 iCall (India):              9152987821
  🇮🇳 Vandrevala Foundation:      1860-2662-345  (available 24/7)
  🌍 International Directory:     https://www.iasp.info/resources/Crisis_Centres/

You don't have to face this alone. Talking to a trained counsellor can make a real difference.
Would you like to tell me a little more about what's been happening for you?"""

_RESPONSE_MODERATE = """\
💙 It sounds like you're going through an incredibly hard time, and I want you to know I hear you.

When things feel this heavy, it can help to talk to someone trained in support. Please consider reaching out:

  🇮🇳 iCall (India):              9152987821
  🇮🇳 Vandrevala Foundation:      1860-2662-345

You matter, and you don't have to carry this alone. I'm here to listen too — can you tell me more about what's been going on?"""


class SafetyChecker:
    """
    Checks user input for crisis signals and returns an appropriate
    override response if needed.
    """

    def check(self, text: str) -> SafetyResult:
        """
        Analyse *text* for crisis language.

        Returns
        -------
        SafetyResult
            .is_crisis = True if any crisis signal is found.
            .level     = severity level (CrisisLevel enum).
            .response  = appropriate override response string.
        """
        text_clean = self._clean(text)

        # Check from most severe to least
        critical_matches = _PATTERN_CRITICAL.findall(text_clean)
        if critical_matches:
            logger.warning(f"CRITICAL crisis signal detected: {critical_matches}")
            return SafetyResult(
                is_crisis=True,
                level=CrisisLevel.CRITICAL,
                matched_phrases=critical_matches,
                response=_RESPONSE_CRITICAL,
            )

        high_matches = _PATTERN_HIGH.findall(text_clean)
        if high_matches:
            logger.warning(f"HIGH crisis signal detected: {high_matches}")
            return SafetyResult(
                is_crisis=True,
                level=CrisisLevel.HIGH,
                matched_phrases=high_matches,
                response=_RESPONSE_HIGH,
            )

        moderate_matches = _PATTERN_MODERATE.findall(text_clean)
        if moderate_matches:
            logger.info(f"MODERATE crisis signal detected: {moderate_matches}")
            return SafetyResult(
                is_crisis=True,
                level=CrisisLevel.MODERATE,
                matched_phrases=moderate_matches,
                response=_RESPONSE_MODERATE,
            )

        return SafetyResult(
            is_crisis=False,
            level=CrisisLevel.NONE,
            matched_phrases=[],
            response="",
        )

    # ── Private ────────────────────────────────────────────────────────────────

    @staticmethod
    def _clean(text: str) -> str:
        """Minimal cleaning: lowercase + collapse whitespace."""
        return re.sub(r"\s+", " ", text.lower()).strip()


# ── Module-level singleton ─────────────────────────────────────────────────────
_checker: SafetyChecker | None = None


def get_checker() -> SafetyChecker:
    global _checker
    if _checker is None:
        _checker = SafetyChecker()
    return _checker


# ── Quick self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    checker = SafetyChecker()
    tests = [
        "I feel really sad today",
        "I've been thinking about killing myself",
        "I feel like giving up on everything",
        "I'm suicidal and don't know what to do",
        "I feel completely hopeless and there's no way out",
    ]
    for t in tests:
        result = checker.check(t)
        print(f"Input  : {t}")
        print(f"Crisis : {result.is_crisis}  Level: {result.level.name}")
        if result.matched_phrases:
            print(f"Matched: {result.matched_phrases}")
        print()

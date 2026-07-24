"""
chatbot.py
==========

Main orchestration class that ties together all pipeline components.

Pipeline

User Input
    ↓
Safety Checker
    ↓
Text Preprocessor
    ↓
Emotion Detector
    ↓
Similarity Engine (Top 3)
    ↓
Groq AI (Optional)
    ↓
TF-IDF Fallback
"""

import logging
import random
from dataclasses import dataclass
from typing import Optional

from data_loader import DataLoader
from preprocess import TextPreprocessor
from emotion_detector import EmotionDetector, EmotionResult
from safety_checker import SafetyChecker, SafetyResult, CrisisLevel
from similarity_engine import SimilarityEngine, MatchResult

# ---------------------------
# NEW IMPORT
# ---------------------------
from groq_service import generate_response

logger = logging.getLogger(__name__)

# ==========================================================
# Toggle AI
# ==========================================================

IS_GROQ_ENABLED = True

# ==========================================================
# Fallback responses
# ==========================================================

FALLBACK_RESPONSES: list[str] = [

    "Thank you for sharing that with me. It takes courage to open up. Could you tell me a little more about what you're experiencing?",

    "I hear you, and I'm here. Could you share a little more about what's been happening?",

    "What you're feeling matters. I'm listening. Can you tell me more about what's on your mind?",

    "It sounds like you're carrying something heavy. I'm here with you. Would you like to share more about what's been happening?"
]

# ==========================================================
# Greetings
# ==========================================================

GREETINGS = {
    "hi",
    "hello",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
    "howdy",
    "greetings",
    "sup",
    "what's up"
}

GREETING_RESPONSE = (
    "Hello! 👋 I'm your mental wellness companion.\n\n"
    "You can share anything you're feeling: stress, anxiety, sadness, "
    "loneliness, or simply talk about your day.\n\n"
    "How are you feeling today?"
)

# ==========================================================
# Thanks
# ==========================================================

THANKS_WORDS = {
    "thank",
    "thanks",
    "thank you",
    "thx",
    "appreciate",
    "helpful"
}

THANKS_RESPONSE = (
    "You're always welcome 💙\n\n"
    "Remember, reaching out is a sign of strength.\n"
    "I'm here whenever you need someone to listen."
)

# ==========================================================
# Chat Response
# ==========================================================

@dataclass
class ChatResponse:

    text: str
    emotion: Optional[str]
    is_crisis: bool
    crisis_level: CrisisLevel
    match_score: float
    source: str

# ==========================================================
# Chatbot
# ==========================================================

class SoulBridge:

    def __init__(self, dataset_path: Optional[str] = None):

        logger.info("Initializing Mental Support Chatbot...")

        loader = DataLoader(dataset_path)

        self._df = loader.load()

        self._crisis_data = loader.get_crisis_responses()

        self._preprocessor = TextPreprocessor()

        self._emotion_detector = EmotionDetector(self._df)

        self._safety_checker = SafetyChecker()

        self._similarity_engine = SimilarityEngine(self._df)

        logger.info("Chatbot Ready.")

    # ─────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────

    def respond(self, user_input: str) -> ChatResponse:
        """
        Generate a response for the given user input.
        """

        # ---------------------------------------------------------
        # Empty Input
        # ---------------------------------------------------------

        if not user_input or not user_input.strip():

            return self._make_response(
                text=(
                    "It seems like your message was empty. "
                    "Take your time—I'm here whenever you're ready to share. 💙"
                ),
                source="fallback"
            )

        # ---------------------------------------------------------
        # Step 1 : Safety Check
        # ---------------------------------------------------------

        safety: SafetyResult = self._safety_checker.check(user_input)

        if safety.is_crisis:

            return ChatResponse(
                text=safety.response,
                emotion=None,
                is_crisis=True,
                crisis_level=safety.level,
                match_score=0.0,
                source="crisis"
            )

        # ---------------------------------------------------------
        # Step 2 : Greeting
        # ---------------------------------------------------------

        stripped = user_input.strip().lower().rstrip("!.?,")

        if stripped in GREETINGS:

            return self._make_response(
                text=GREETING_RESPONSE,
                source="greeting"
            )

        # ---------------------------------------------------------
        # Step 3 : Thanks
        # ---------------------------------------------------------

        if any(word in stripped for word in THANKS_WORDS):

            return self._make_response(
                text=THANKS_RESPONSE,
                source="thanks"
            )

        # ---------------------------------------------------------
        # Step 4 : Preprocess
        # ---------------------------------------------------------

        processed = self._preprocessor.process(user_input)

        # ---------------------------------------------------------
        # Step 5 : Emotion Detection
        # ---------------------------------------------------------

        emotion_result: EmotionResult = self._emotion_detector.detect(
            processed.clean,
            processed.tokens
        )

        # ---------------------------------------------------------
        # Step 6 : Retrieve Top 3 Similar Responses
        # ---------------------------------------------------------

        matches = self._similarity_engine.query(
            clean_text=processed.clean,
            tokens=processed.tokens,
            detected_emotion=emotion_result.emotion,
            top_k=3
        )

        # ---------------------------------------------------------
        # No Match Found
        # ---------------------------------------------------------

        if not matches:

            return ChatResponse(
                text=random.choice(FALLBACK_RESPONSES),
                emotion=emotion_result.emotion,
                is_crisis=False,
                crisis_level=CrisisLevel.NONE,
                match_score=0.0,
                source="fallback"
            )

        best_match = matches[0]

        # ---------------------------------------------------------
        # Default Response (TF-IDF)
        # ---------------------------------------------------------

        response_text = self._personalise(
            best_match.response,
            emotion_result,
            user_input
        )

        source = "match"

        # ---------------------------------------------------------
        # Build Context
        # ---------------------------------------------------------

        context = ""

        for i, item in enumerate(matches, start=1):

            context += f"""
Example {i}

User:
{item.question}

Counsellor:
{item.response}

"""

        # ---------------------------------------------------------
        # GROQ BOOLEAN CHECK
        # ---------------------------------------------------------

        if IS_GROQ_ENABLED:

            try:

                ai_response = generate_response(
                    user_input=user_input,
                    emotion=emotion_result.emotion,
                    context=context
                )

                if ai_response and ai_response.strip():

                    response_text = ai_response
                    source = "groq"

            except Exception as ex:

                logger.exception(f"Groq Error: {ex}")

                # TF-IDF response is already prepared

        # ---------------------------------------------------------
        # Final Response
        # ---------------------------------------------------------

        return ChatResponse(
            text=response_text,
            emotion=emotion_result.emotion,
            is_crisis=False,
            crisis_level=CrisisLevel.NONE,
            match_score=best_match.score,
            source=source
        )
    def chat(self, user_input: str) -> str:
        """
        Convenience wrapper.
        """
        return self.respond(user_input).text

    # ==========================================================
    # Private Helpers
    # ==========================================================

    @staticmethod
    def _personalise(
        base_response: str,
        emotion_result: EmotionResult,
        user_input: str
    ) -> str:

        PREFIXES = {

            "anxiety": [
                "I can sense that anxiety is weighing on you right now. ",
                "It sounds like worry is taking up a lot of space for you. "
            ],

            "sadness": [
                "I can hear the sadness in your words. ",
                "Thank you for sharing something so personal with me. "
            ],

            "stress": [
                "It sounds like you're carrying a lot right now. ",
                "I can hear how overwhelmed you are. "
            ],

            "anger": [
                "Your frustration comes through clearly. ",
                "It makes sense you're feeling angry. "
            ],

            "loneliness": [
                "I hear how lonely you've been feeling. ",
                "Loneliness can feel very heavy. "
            ],

            "guilt": [
                "Guilt can be a difficult burden to carry. "
            ],

            "happiness": [
                "It's wonderful to hear something positive from you. 😊 "
            ]
        }

        emotion = emotion_result.emotion
        confidence = emotion_result.confidence

        if (
            confidence >= 0.35
            and emotion in PREFIXES
        ):

            prefix = random.choice(PREFIXES[emotion])

            if not base_response.lower().startswith(
                ("i can", "i hear", "it sounds")
            ):
                return prefix + base_response

        return base_response

    # ==========================================================
    # Helper Response Builder
    # ==========================================================

    @staticmethod
    def _make_response(
        text: str,
        emotion: Optional[str] = None,
        source: str = "fallback"
    ) -> ChatResponse:

        return ChatResponse(
            text=text,
            emotion=emotion,
            is_crisis=False,
            crisis_level=CrisisLevel.NONE,
            match_score=0.0,
            source=source
        )


# ==========================================================
# Self Test
# ==========================================================

if __name__ == "__main__":

    bot = SoulBridge()

    tests = [

        "Hello",

        "I feel anxious all the time",

        "I am stressed because of work",

        "Nobody understands me",

        "I am feeling lonely",

        "I don't know why I feel angry",

        "Thank you",

        "I want to kill myself"

    ]

    for text in tests:

        print("=" * 70)

        print("User :", text)

        response = bot.respond(text)

        print("Bot :", response.text)

        print("Emotion :", response.emotion)

        print("Score :", response.match_score)

        print("Source :", response.source)
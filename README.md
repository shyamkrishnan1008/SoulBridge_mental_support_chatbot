# SoulBridge — Mental Support Chatbot

A production-ready Python chatbot that provides empathetic, supportive responses to users experiencing mental health challenges. It matches user messages against a curated mental health dataset using **TF-IDF vectorisation** and **cosine similarity**, with built-in emotion detection and a safety system for crisis situations.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Example Conversations](#example-conversations)
- [Crisis Safety System](#crisis-safety-system)
- [Customisation](#customisation)
- [Troubleshooting](#troubleshooting)

---

## Features

| Feature | Details |
|---|---|
| **Empathetic Responses** | Matched from a curated mental health counselling dataset |
| **TF-IDF + Cosine Similarity** | Fast, cache-friendly relevance retrieval |
| **Keyword Boosting** | Domain-aware keyword overlap scoring |
| **Emotion Detection** | 8 emotions: anxiety, sadness, anger, stress, loneliness, happiness, guilt, confusion |
| **Crisis Safety System** | 3-level crisis detection with immediate emergency guidance |
| **Modular Architecture** | Each concern in its own module — easy to extend |
| **Multiple Dataset Formats** | Supports 4 common Kaggle JSON structures |
| **Coloured CLI** | Friendly terminal UI with typing animation |
| **Debug Mode** | `--debug` flag shows emotion & similarity metadata |

---

## Project Structure

```
mental_support_chatbot/
├── data/
│   └── mental_health_dataset.json   # knowledge base
├── app.py                           # interactive CLI entry point
├── chatbot.py                       # main orchestration class
├── data_loader.py                   # JSON dataset loading & normalisation
├── preprocess.py                    # text cleaning & tokenisation
├── similarity_engine.py             # TF-IDF + cosine similarity retrieval
├── emotion_detector.py              # keyword + ML emotion detection
├── safety_checker.py                # crisis keyword detection & response
├── requirements.txt
└── README.md
```

---

## Installation

### Prerequisites
- Python 3.10 or higher
- pip

### Steps

```bash
# 1. Clone or download the project
git clone <repo-url>
cd mental_support_chatbot

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLTK data (automatic on first run, or manually)
python -c "import nltk; nltk.download(['punkt','stopwords','wordnet','omw-1.4','punkt_tab'])"
```

### (Optional) Use a Custom Kaggle Dataset

Download any mental health / counselling / empathetic-dialogue dataset from Kaggle, place the JSON file in the `data/` folder, and point the chatbot to it:

```bash
python app.py --dataset data/your_kaggle_dataset.json
```

Supported JSON structures:
- **Format A** `{"conversations": [{question, response, emotion, intent, keywords}]}`
- **Format B** Flat list: `[{question, answer}, …]`
- **Format C** `{"data": [{context, response}, …]}`
- **Format D** Rasa-style: `{"intents": [{tag, patterns, responses}, …]}`

---

## Usage

### Interactive Chat (recommended)

```bash
python app.py
```

### Debug Mode (shows emotion & score metadata)

```bash
python app.py --debug
```

### Custom Dataset

```bash
python app.py --dataset data/counselchat-data.json
```

### Logging

```bash
python app.py --log-level INFO
```

### Programmatic Use

```python
from chatbot import MentalSupportChatbot

bot = MentalSupportChatbot()

# Simple string response
print(bot.chat("I'm feeling really anxious lately"))

# Full response object with metadata
response = bot.respond("I feel so lonely and nobody understands me")
print(response.text)
print(f"Detected emotion: {response.emotion}")
print(f"Match score:      {response.match_score}")
print(f"Is crisis:        {response.is_crisis}")
```

---

## Dataset

The bundled `data/mental_health_dataset.json` includes 25 hand-crafted entries covering:

- Anxiety & panic attacks
- Sadness & depression
- Loneliness & isolation
- Stress & burnout
- Anger & frustration
- Grief & loss
- Low self-worth & guilt
- Hopelessness & existential distress
- Help-seeking & therapy
- Positive emotions

Each entry contains:
```json
{
  "id": 1,
  "question": "I feel so anxious all the time and I don't know why",
  "emotion": "anxiety",
  "intent": "seeking_support",
  "response": "It sounds like you're carrying a heavy burden …",
  "keywords": ["anxious", "anxiety", "nervous", "worried"]
}
```

Replace this file with any Kaggle dataset for a much richer knowledge base.

---

## Architecture

```
User Input
    │
    ▼
┌─────────────────────┐
│   SafetyChecker     │ ──► Crisis detected? ──► Emergency Response
└─────────────────────┘
    │ No crisis
    ▼
┌─────────────────────┐
│  TextPreprocessor   │  lowercase → contractions → remove noise
│                     │  → tokenise → remove stopwords → lemmatise
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  EmotionDetector    │  keyword dict → ML classifier → dominant emotion
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  SimilarityEngine   │  TF-IDF cosine (65%) + keyword overlap (35%)
│                     │  + emotion bonus (+10%) ──► best MatchResult
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Response Formatter │  personalise with emotion prefix
└─────────────────────┘
    │
    ▼
  ChatResponse
```

### Score Combination Formula

```
score = 0.65 × tfidf_cosine
      + 0.35 × keyword_overlap
      + 0.10 × emotion_match_bonus   (if emotion matches row label)
```

---

## Example Conversations

```
You: I've been feeling really anxious lately and I can't sleep
SoulBridge: I can sense that anxiety is weighing on you right now. A racing mind
          at night can be exhausting and frightening. When we can't quiet our
          thoughts, it often means our mind is processing a lot — worries,
          unresolved emotions, or future fears. Some people find deep breathing,
          journaling before bed, or a simple body scan helpful. Would you like
          to explore what thoughts tend to surface most at night?

──────────────────────────────────────────────────

You: I feel like nobody understands me and I'm completely alone
SoulBridge: I hear how alone you've been feeling. Loneliness can be one of the
          most painful feelings to sit with. Even when we're surrounded by
          people, we can feel deeply alone — and that's a very human experience.
          Your feelings make complete sense. Would it help to talk about what's
          been making you feel disconnected?

──────────────────────────────────────────────────

You: I'm so stressed out about work and everything feels overwhelming
SoulBridge: It sounds like you're carrying a lot of weight right now. It sounds
          like you're under a tremendous amount of pressure. Work stress can
          spill into every corner of your life and make everything feel
          impossible to manage. Would it help to talk through what's piling up,
          or would you like some gentle strategies that might ease the load?
```

---

## Crisis Safety System

The chatbot monitors all input for crisis language at three severity levels:

| Level | Examples | Response |
|---|---|---|
| **MODERATE** | "feel like giving up", "can't take it anymore" | Gentle check-in + helpline |
| **HIGH** | "suicidal", "want to disappear forever" | Urgent support + helpline |
| **CRITICAL** | "kill myself", "self-harm", "want to die" | Immediate crisis response + 112 |

**Helplines included in crisis responses:**

- 🇮🇳 iCall (India): **9152987821**
- 🇮🇳 Vandrevala Foundation: **1860-2662-345** (24/7, free)
- 🌍 International Directory: https://www.iasp.info/resources/Crisis_Centres/
- 💬 Crisis Text Line: Text **HOME** to **741741**

> ⚠ **Important:** This chatbot is a supportive tool, not a replacement for professional mental health care. Always refer users to qualified professionals for clinical concerns.

---

## Customisation

### Add More Responses

Edit `data/mental_health_dataset.json` and add entries following the existing schema.

### Adjust Scoring Weights

In `similarity_engine.py`:
```python
WEIGHT_TFIDF   = 0.65   # increase for more semantic matching
WEIGHT_KEYWORD = 0.35   # increase for more keyword-driven matching
EMOTION_BONUS  = 0.10   # increase to prioritise emotion-matched responses
```

### Add Emotion Keywords

In `emotion_detector.py`, extend `EMOTION_KEYWORDS`:
```python
EMOTION_KEYWORDS["anxiety"].extend(["jittery", "apprehensive"])
```

### Add Crisis Phrases

In `safety_checker.py`, extend `CRITICAL_PHRASES`, `HIGH_PHRASES`, or `MODERATE_PHRASES`.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `FileNotFoundError: Dataset not found` | Place your JSON file in `data/` or use `--dataset` |
| `LookupError: NLTK resource not found` | Run `python -c "import nltk; nltk.download('all')"` |
| No coloured output | Install colorama: `pip install colorama` |
| Responses seem irrelevant | Expand your dataset or adjust `MIN_SCORE_THRESHOLD` in `similarity_engine.py` |
| Import errors | Ensure you're in the project root directory and venv is active |

---

## Dependencies

| Package | Purpose |
|---|---|
| `pandas` | Dataset loading & management |
| `numpy` | Array operations for scoring |
| `scikit-learn` | TF-IDF vectoriser, cosine similarity, Naive Bayes |
| `nltk` | Tokenisation, stopwords, lemmatisation |
| `colorama` | Cross-platform terminal colours |

---

## Disclaimer

SoulBridge is an educational and supportive tool. It is **not** a medical device and does not provide clinical diagnosis or treatment. If you or someone you know is in crisis, please contact a qualified mental health professional or emergency services immediately.

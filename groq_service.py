"""
groq_service.py
------------------------------------
SoulBridge AI Response Generator

Uses Groq Llama 3.3 to generate empathetic responses
using the retrieved counselling examples as context.
"""

from groq import Groq

# =====================================================
# CHANGE THIS TO YOUR API KEY
# =====================================================
GROQ_API_KEY = ""

MODEL_NAME = "llama-3.3-70b-versatile"

client = Groq(api_key=GROQ_API_KEY)


SYSTEM_PROMPT = """
You are SoulBridge, an AI Mental Wellness Companion.

Your purpose is to provide emotional support and empathetic conversations.

Rules:

1. Be warm and compassionate.
2. Never diagnose medical or psychiatric conditions.
3. Never prescribe medications.
4. Never claim to be a licensed therapist.
5. Use the counselling examples only as guidance.
6. Never copy the examples.
7. Keep responses between 80 and 150 words.
8. Encourage healthy coping strategies.
9. Ask ONE gentle follow-up question whenever appropriate.
10. If the user appears to be in danger,
    encourage contacting trusted people or emergency services.
"""


def generate_response(user_input, emotion, context):
    """
    Generate an AI response using Groq.

    Parameters
    ----------
    user_input : str
    emotion : str
    context : str
        Top retrieved counselling examples.

    Returns
    -------
    str
    """

    prompt = f"""
Detected Emotion:
{emotion}

Reference Counselling Examples:
{context}

User Message:
{user_input}

Respond naturally.
Do not copy the examples.
"""

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.7,
        max_tokens=350,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return completion.choices[0].message.content.strip()
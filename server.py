"""
server.py
=========
Flask web server — bridges the browser UI with the existing chatbot code.
Run:  python server.py
Open: http://localhost:5050
"""

from flask import Flask, request, jsonify, render_template_string
import logging
logging.disable(logging.WARNING)

from chatbot import SoulBridge

app = Flask(__name__)
bot = SoulBridge()

# ── Serve the chat UI ──────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(open("templates/index.html").read())

# ── Chat API endpoint ──────────────────────────────────────────────────────────
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    response = bot.respond(user_message)
    return jsonify({
        "response": response.text,
        "emotion":  response.emotion or "neutral",
        "is_crisis": response.is_crisis,
        "score":    response.match_score,
        "source":   response.source,
    })

if __name__ == "__main__":
    print("✅ SoulBridge is running!")
    print("👉 SoulBridge running at http://localhost:5050")
    app.run(debug=True, port=5050)

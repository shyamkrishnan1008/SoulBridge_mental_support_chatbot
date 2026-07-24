"""
app.py
======
Interactive command-line interface for the Mental Support Chatbot.

Features
--------
- Coloured, formatted terminal output (via colorama)
- Typing animation for a more natural feel
- Conversation history display
- Debug mode (--debug) to show emotion / match scores
- Graceful shutdown on Ctrl+C or exit commands

Usage
-----
    python app.py            # normal mode
    python app.py --debug    # show emotion & similarity metadata
"""

import argparse
import logging
import os
import sys
import textwrap
import time
from datetime import datetime

# ── Try to import colorama for cross-platform colour support ───────────────────
try:
    from colorama import Fore, Back, Style, init as colorama_init
    colorama_init(autoreset=True)
    COLORS_AVAILABLE = True
except ImportError:
    # Graceful degradation — plain output if colorama not installed
    class _Dummy:
        def __getattr__(self, _): return ""
    Fore = Back = Style = _Dummy()  # type: ignore
    COLORS_AVAILABLE = False

from chatbot import SoulBridge, ChatResponse
from safety_checker import CrisisLevel

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,            # suppress info logs in normal mode
    format="%(levelname)s | %(name)s | %(message)s",
)

# ── Constants ──────────────────────────────────────────────────────────────────
APP_NAME    = "SoulBridge — Mental Wellness Companion"
VERSION     = "1.0.0"
EXIT_WORDS  = {"exit", "quit", "bye", "goodbye", "q"}
MAX_HISTORY = 20   # conversation pairs to remember

# ── Colour palette ─────────────────────────────────────────────────────────────
C_TITLE      = Fore.CYAN   + Style.BRIGHT
C_BOT        = Fore.GREEN
C_USER       = Fore.YELLOW
C_CRISIS     = Fore.RED    + Style.BRIGHT
C_META       = Fore.MAGENTA
C_DIM        = Style.DIM
C_RESET      = Style.RESET_ALL
C_SEPARATOR  = Fore.BLUE   + Style.DIM


def _banner() -> None:
    """Print the welcome banner."""
    width = 70
    print()
    print(C_TITLE + "=" * width)
    print(C_TITLE + f"  {APP_NAME}  v{VERSION}".center(width))
    print(C_TITLE + "=" * width)
    print(C_DIM + "  A safe, supportive space to share what's on your mind.")
    print(C_DIM + "  Type 'exit' or 'quit' to leave at any time.")
    print(C_DIM + "  In a real crisis, please call your local emergency number.")
    print(C_TITLE + "=" * width)
    print()


def _format_response(response: ChatResponse, debug: bool = False) -> str:
    """Format a ChatResponse for terminal display."""
    lines = []

    # Crisis uses a distinct colour
    colour = C_CRISIS if response.is_crisis else C_BOT

    # Word-wrap long responses at 80 chars
    wrapped = textwrap.fill(response.text, width=80, subsequent_indent="  ")
    lines.append(f"{colour}SoulBridge: {C_RESET}{wrapped}")

    if debug:
        meta_parts = []
        if response.emotion:
            meta_parts.append(f"emotion={response.emotion}")
        meta_parts.append(f"score={response.match_score:.3f}")
        meta_parts.append(f"source={response.source}")
        if response.is_crisis:
            meta_parts.append(f"crisis_level={response.crisis_level.name}")
        lines.append(f"{C_META}  [debug: {' | '.join(meta_parts)}]{C_RESET}")

    return "\n".join(lines)


def _typing_animation(duration: float = 0.8) -> None:
    """Show a brief typing indicator."""
    frames = ["   ", ".  ", ".. ", "..."]
    end_time = time.time() + duration
    i = 0
    while time.time() < end_time:
        sys.stdout.write(f"\r{C_DIM}SoulBridge is thinking {frames[i % len(frames)]}{C_RESET}")
        sys.stdout.flush()
        time.sleep(0.2)
        i += 1
    sys.stdout.write("\r" + " " * 40 + "\r")   # clear line
    sys.stdout.flush()


def _separator() -> None:
    print(C_SEPARATOR + "─" * 70 + C_RESET)


def run_app(debug: bool = False, dataset_path: str | None = None) -> None:
    """Main entry point for the interactive CLI session."""

    _banner()

    # ── Initialise chatbot ────────────────────────────────────────────────────
    print(f"{C_DIM}Loading mental health knowledge base …{C_RESET}")
    try:
        bot = SoulBridge(dataset_path)
    except FileNotFoundError as exc:
        print(f"{C_CRISIS}Error: {exc}{C_RESET}")
        sys.exit(1)
    except Exception as exc:
        print(f"{C_CRISIS}Failed to initialise chatbot: {exc}{C_RESET}")
        raise

    print(f"{C_BOT}SoulBridge: {C_RESET}Hello! 👋  I'm your mental wellness companion — a safe space to share.")
    print(f"{C_BOT}SoulBridge: {C_RESET}How are you feeling today?")
    _separator()

    # ── Conversation loop ─────────────────────────────────────────────────────
    conversation_history: list[dict] = []
    turn = 0

    while True:
        try:
            # Prompt
            sys.stdout.write(f"\n{C_USER}You: {C_RESET}")
            sys.stdout.flush()
            user_input = input().strip()

        except (EOFError, KeyboardInterrupt):
            print(f"\n\n{C_BOT}SoulBridge: {C_RESET}Take care of yourself. You're not alone. 💙")
            break

        if not user_input:
            continue

        # Exit commands
        if user_input.lower() in EXIT_WORDS:
            print(
                f"\n{C_BOT}SoulBridge: {C_RESET}"
                "Thank you for spending time here. Remember, reaching out is always brave. "
                "Take care of yourself — and come back whenever you need. 💙"
            )
            break

        # Generate response with typing animation
        _typing_animation(duration=min(0.4 + len(user_input) * 0.005, 1.5))
        response = bot.respond(user_input)

        # Display
        _separator()
        print(_format_response(response, debug=debug))
        _separator()

        # Store history
        turn += 1
        conversation_history.append({
            "turn": turn,
            "timestamp": datetime.now().isoformat(),
            "user": user_input,
            "bot": response.text,
            "emotion": response.emotion,
            "crisis": response.is_crisis,
        })

        # Safety: if a critical crisis was detected, pause and emphasise
        if response.crisis_level == CrisisLevel.CRITICAL:
            print(
                f"\n{C_CRISIS}⚠  Please call a helpline right now. "
                f"You don't have to face this alone.{C_RESET}\n"
            )

        # Trim history
        if len(conversation_history) > MAX_HISTORY:
            conversation_history = conversation_history[-MAX_HISTORY:]


# ── CLI argument parsing ───────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"{APP_NAME} — Interactive CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python app.py
              python app.py --debug
              python app.py --dataset /path/to/custom_dataset.json
        """),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show emotion detection and similarity scores alongside responses.",
    )
    parser.add_argument(
        "--dataset",
        metavar="PATH",
        default=None,
        help="Path to a custom Kaggle JSON dataset (default: data/mental_health_dataset.json).",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: WARNING).",
    )
    return parser.parse_args()


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = _parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    run_app(debug=args.debug, dataset_path=args.dataset)

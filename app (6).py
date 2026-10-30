"""
AI Quiz and Flashcard Generator
Paste lesson notes, choose a difficulty, and get multiple-choice questions with
explanations, automatic scoring, and flashcards for revision.

Uses the Groq REST API directly through `requests` (installed with Streamlit),
so no extra packages are needed.

Run locally:  streamlit run app.py
"""

import json
import os
import random
import re
from datetime import datetime

import requests
import streamlit as st

# --------------------------------------------------------------------------- #
# Page configuration
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="AI Quiz & Flashcards", page_icon="🧠", layout="wide")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
DIFFICULTIES = ["Easy", "Medium", "Hard"]
MIN_CHARS = 100
MAX_CHARS = 20000

DIFFICULTY_RULES = {
    "Easy": "Test basic recall of key facts and definitions. Wrong options should be clearly different.",
    "Medium": "Test understanding and application. Wrong options should be plausible.",
    "Hard": "Test deep understanding, comparison and reasoning. Wrong options should be very plausible and close to correct.",
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def get_api_key(sidebar_value):
    """Priority: sidebar input -> st.secrets -> environment variable."""
    if sidebar_value and sidebar_value.strip():
        return sidebar_value.strip()
    try:
        secret = st.secrets["GROQ_API_KEY"]
        if secret:
            return str(secret).strip()
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY", "").strip()


def build_messages(notes, difficulty, n_questions, n_cards):
    system_msg = (
        "You are an expert teacher who writes accurate exam questions and flashcards "
        "from lesson notes. Use ONLY information from the notes. Reply with valid JSON only."
    )
    user_msg = (
        "Create study material from the LESSON NOTES below.\n\n"
        "DIFFICULTY: " + difficulty + " - " + DIFFICULTY_RULES[difficulty] + "\n"
        "Number of multiple-choice questions: " + str(n_questions) + "\n"
        "Number of flashcards: " + str(n_cards) + "\n\n"
        "Return a JSON object with exactly this shape:\n"
        "{\n"
        '  "questions": [\n'
        "    {\n"
        '      "question": "text",\n'
        '      "correct_answer": "the single correct answer, written out in full",\n'
        '      "wrong_answers": ["wrong answer 1", "wrong answer 2", "wrong answer 3"],\n'
        '      "explanation": "1-3 sentences explaining why the correct answer is right"\n'
        "    }\n"
        "  ],\n"
        '  "flashcards": [ {"front": "term or question", "back": "short answer or definition"} ]\n'
        "}\n\n"
        "Rules:\n"
        "- Each question has exactly one correct_answer and exactly 3 wrong_answers.\n"
        "- The wrong answers must be clearly incorrect according to the notes.\n"
        "- The explanation must agree with the correct_answer.\n"
        "- Do not write 'A)' or 'B)' labels inside the options.\n"
        "- Do not use 'all of the above' or 'none of the above'.\n\n"
        'LESSON NOTES:\n"""\n' + notes + '\n"""'
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


@st.cache_data(ttl=600, show_spinner=False)
def fetch_models(api_key):
    """Ask Groq which chat models this API key can use right now."""
    try:
        r = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": "Bearer " + api_key},
            timeout=20,
        )
        if r.status_code != 200:
            return []
        skip = ("whisper", "guard", "tts", "playai", "orpheus", "safeguard", "embed")
        ids = [
            m["id"]
            for m in r.json().get("data", [])
            if m.get("active", True) and not any(s in m["id"].lower() for s in skip)
        ]
        return sorted(ids)
    except Exception:
        return []


def call_groq(api_key, model, messages):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.4,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}
    resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=90)

    # Some models do not support JSON mode: retry once without it.
    if resp.status_code == 400 and "response_format" in resp.text:
        payload.pop("response_format")
        resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=90)

    if resp.status_code != 200:
        raise RuntimeError("Groq API error " + str(resp.status_code) + ": " + resp.text[:300])
    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        return ""
    return (choices[0].get("message", {}).get("content") or "").strip()


def extract_json(text):
    """Parse JSON even if the model wrapped it in code fences or extra text."""
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError("The AI did not return valid JSON.")


def clean_option(opt):
    return re.sub(r"^\s*[A-Da-d][\)\.\:]\s+", "", str(opt)).strip()


def validate(data):
    """Return (questions, flashcards) with only well-formed items."""
    questions = []
    for q in data.get("questions", []) or []:
        try:
            correct_text = clean_option(q["correct_answer"])
            wrong = [clean_option(o) for o in q["wrong_answers"]]
            wrong = [w for w in wrong if w and w.lower() != correct_text.lower()]
            options = [correct_text] + wrong[:3]
            if correct_text and len(options) == 4 and str(q["question"]).strip():
                # Shuffle so the correct answer is not always in the same place.
                random.shuffle(options)
                idx = options.index(correct_text)
                questions.append(
                    {
                        "question": str(q["question"]).strip(),
                        "options": options,
                        "answer_index": idx,
                        "explanation": str(q.get("explanation", "")).strip(),
                    }
                )
        except Exception:
            continue

    cards = []
    for c in data.get("flashcards", []) or []:
        try:
            front, back = str(c["front"]).strip(), str(c["back"]).strip()
            if front and back:
                cards.append({"front": front, "back": back})
        except Exception:
            continue
    return questions, cards


def friendly_error(exc):
    msg = str(exc)
    low = msg.lower()
    if "401" in msg or "invalid api key" in low or "invalid_api_key" in low:
        return "Your Groq API key looks invalid. Please check it and try again."
    if "429" in msg or "rate limit" in low:
        return "Groq rate limit reached. Please wait a minute and try again."
    if "decommissioned" in low or "does not exist" in low or "404" in msg:
        return (
            "That model is not available on your account. Choose another model in the "
            "sidebar. Details: " + msg[:300]
        )
    if "connection" in low or "timeout" in low or "timed out" in low:
        return "Could not reach the Groq API. Check your internet connection and retry."
    if "json" in low:
        return "The AI reply could not be read. Please click Generate again."
    return "Something went wrong: " + msg


def build_export(questions, cards, difficulty, answers=None):
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = ["# Quiz and Flashcards", "", "_Generated " + stamp + " | Difficulty: " + difficulty + "_", ""]
    lines.append("## Multiple-choice questions")
    lines.append("")
    for i, q in enumerate(questions):
        lines.append("**Q" + str(i + 1) + ". " + q["question"] + "**")
        for j, opt in enumerate(q["options"]):
            lines.append("- " + "ABCD"[j] + ") " + opt)
        lines.append("")
        lines.append("Answer: **" + "ABCD"[q["answer_index"]] + ") " + q["options"][q["answer_index"]] + "**")
        if q["explanation"]:
            lines.append("")
            lines.append("Explanation: " + q["explanation"])
        lines.append("")
    lines.append("## Flashcards")
    lines.append("")
    for i, c in enumerate(cards):
        lines.append(str(i + 1) + ". **" + c["front"] + "** - " + c["back"])
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
defaults = {
    "questions": [],
    "cards": [],
    "difficulty": "Medium",
    "quiz_id": 0,
    "submitted": False,
    "answers": {},
    "card_index": 0,
    "show_back": False,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def card_prev():
    st.session_state["card_index"] = max(st.session_state["card_index"] - 1, 0)
    st.session_state["show_back"] = False


def card_next():
    last = len(st.session_state["cards"]) - 1
    st.session_state["card_index"] = min(st.session_state["card_index"] + 1, last)
    st.session_state["show_back"] = False


def card_flip():
    st.session_state["show_back"] = not st.session_state["show_back"]


def card_restart():
    st.session_state["card_index"] = 0
    st.session_state["show_back"] = False


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Settings")
    key_input = st.text_input(
        "Groq API key (optional)",
        type="password",
        help="Leave empty to use st.secrets or the GROQ_API_KEY environment variable.",
    )
    api_key = get_api_key(key_input)
    if api_key:
        st.success("API key found ✅")
    else:
        st.warning("No API key found yet")

    live_models = fetch_models(api_key) if api_key else []
    model_options = live_models or MODELS
    default_index = 0
    for preferred in ("llama-3.3-70b-versatile", "llama-3.1-8b-instant"):
        if preferred in model_options:
            default_index = model_options.index(preferred)
            break
    model = st.selectbox("Groq model", model_options, index=default_index)
    if live_models:
        st.caption("Models loaded from your Groq account.")
    elif api_key:
        st.caption("Could not load your model list, showing defaults.")
    difficulty = st.select_slider("Difficulty", options=DIFFICULTIES, value="Medium")
    n_questions = st.slider("Number of questions", 3, 15, 5)
    n_cards = st.slider("Number of flashcards", 3, 20, 8)

# --------------------------------------------------------------------------- #
# Main page
# --------------------------------------------------------------------------- #
st.title("🧠 AI Quiz and Flashcard Generator")
st.write(
    "Paste your lesson notes, choose a difficulty, and generate a quiz with "
    "explanations plus flashcards for revision."
)

notes = st.text_area(
    "Lesson notes",
    height=260,
    placeholder="Paste your lesson notes, textbook section or summary here...",
)
st.caption(str(len(notes.strip())) + " characters (max " + str(MAX_CHARS) + ")")

if st.button("✨ Generate quiz and flashcards", type="primary"):
    text = notes.strip()
    if not api_key:
        st.error(
            "No Groq API key found. Enter it in the sidebar, or add GROQ_API_KEY "
            "to your Streamlit secrets."
        )
    elif not text:
        st.warning("Please paste your lesson notes first.")
    elif len(text) < MIN_CHARS:
        st.warning("Your notes are too short. Please enter at least " + str(MIN_CHARS) + " characters.")
    else:
        if len(text) > MAX_CHARS:
            st.warning("Your notes were longer than " + str(MAX_CHARS) + " characters and were trimmed.")
            text = text[:MAX_CHARS]
        try:
            with st.spinner("Creating your quiz..."):
                raw = call_groq(api_key, model, build_messages(text, difficulty, n_questions, n_cards))
                questions, cards = validate(extract_json(raw))
            if not questions:
                st.error("The AI did not produce usable questions. Please try again.")
            else:
                st.session_state["questions"] = questions
                st.session_state["cards"] = cards
                st.session_state["difficulty"] = difficulty
                st.session_state["quiz_id"] += 1
                st.session_state["submitted"] = False
                st.session_state["answers"] = {}
                st.session_state["card_index"] = 0
                st.session_state["show_back"] = False
        except Exception as exc:
            st.error(friendly_error(exc))

# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
questions = st.session_state["questions"]
cards = st.session_state["cards"]

if questions:
    st.success("Your study material is ready! 🎉")
    tab_quiz, tab_cards, tab_export = st.tabs(["📝 Quiz", "🃏 Flashcards", "⬇️ Download"])

    # ------------------------------- Quiz ---------------------------------- #
    with tab_quiz:
        qid = st.session_state["quiz_id"]

        if not st.session_state["submitted"]:
            with st.form("quiz_form_" + str(qid)):
                for i, q in enumerate(questions):
                    st.markdown("**Q" + str(i + 1) + ". " + q["question"] + "**")
                    st.radio(
                        "Choose one answer",
                        options=list(range(4)),
                        format_func=lambda j, opts=q["options"]: "ABCD"[j] + ") " + opts[j],
                        index=None,
                        key="q_" + str(qid) + "_" + str(i),
                        label_visibility="collapsed",
                    )
                    st.write("")
                submit = st.form_submit_button("✅ Submit answers", type="primary")
            if submit:
                st.session_state["answers"] = {
                    i: st.session_state.get("q_" + str(qid) + "_" + str(i))
                    for i in range(len(questions))
                }
                st.session_state["submitted"] = True
                st.rerun()
        else:
            answers = st.session_state["answers"]
            score = sum(1 for i, q in enumerate(questions) if answers.get(i) == q["answer_index"])
            total = len(questions)
            pct = round(score / total * 100)

            st.subheader("Your score: " + str(score) + " / " + str(total) + " (" + str(pct) + "%)")
            st.progress(score / total)
            if pct >= 80:
                st.success("Excellent work! 🌟")
            elif pct >= 50:
                st.info("Good effort. Review the explanations below and try again.")
            else:
                st.warning("Keep practicing. Read the explanations and retake the quiz.")

            for i, q in enumerate(questions):
                chosen = answers.get(i)
                correct = q["answer_index"]
                ok = chosen == correct
                st.markdown("**Q" + str(i + 1) + ". " + q["question"] + "**")
                if chosen is None:
                    st.warning("Not answered. Correct answer: " + "ABCD"[correct] + ") " + q["options"][correct])
                elif ok:
                    st.success("Correct: " + "ABCD"[chosen] + ") " + q["options"][chosen])
                else:
                    st.error(
                        "Your answer: " + "ABCD"[chosen] + ") " + q["options"][chosen]
                        + "\n\nCorrect answer: " + "ABCD"[correct] + ") " + q["options"][correct]
                    )
                if q["explanation"]:
                    st.caption("💡 " + q["explanation"])
                st.write("")

            if st.button("🔁 Retake quiz"):
                st.session_state["quiz_id"] += 1
                st.session_state["submitted"] = False
                st.session_state["answers"] = {}
                st.rerun()

    # ----------------------------- Flashcards ------------------------------ #
    with tab_cards:
        if not cards:
            st.info("No flashcards were generated. Click Generate again.")
        else:
            idx = min(st.session_state["card_index"], len(cards) - 1)
            card = cards[idx]
            st.caption("Card " + str(idx + 1) + " of " + str(len(cards)))
            st.progress((idx + 1) / len(cards))

            if st.session_state["show_back"]:
                st.success("**Answer**\n\n" + card["back"])
            else:
                st.info("**Question / term**\n\n" + card["front"])

            c1, c2, c3, c4 = st.columns(4)
            c1.button("⬅️ Previous", on_click=card_prev, disabled=idx == 0, use_container_width=True)
            c2.button("🔄 Flip card", on_click=card_flip, use_container_width=True)
            c3.button("Next ➡️", on_click=card_next, disabled=idx >= len(cards) - 1, use_container_width=True)
            c4.button("⏮️ Restart", on_click=card_restart, use_container_width=True)

    # ------------------------------ Download ------------------------------- #
    with tab_export:
        st.write("Download the quiz (with answers and explanations) and flashcards as a Markdown file.")
        st.download_button(
            "⬇️ Download study pack (.md)",
            data=build_export(questions, cards, st.session_state["difficulty"]),
            file_name="study_pack_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".md",
            mime="text/markdown",
        )

#!/usr/bin/env python3
"""medquiz.py — Generate an interactive medical multiple-choice quiz with Claude.

Prompts for a medical topic (or takes one on the command line), asks Claude for
5 multiple-choice questions (4 options each), then quizzes you interactively:
you answer A/B/C/D and it shows the correct answer with a brief explanation.
Uses the raw HTTP API via `requests` (no `anthropic` package required).

Setup:
    pip install requests
    export ANTHROPIC_API_KEY="sk-ant-..."

Run:
    python medquiz.py                          # interactive prompt
    python medquiz.py "Acute pancreatitis"     # topic on the command line
"""

import argparse
import json
import os
import sys

import requests

API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = """You are an expert medical educator writing exam-style \
multiple-choice questions for a medical student or junior clinician.

For the topic the user provides, write exactly 5 high-quality, medically \
accurate single-best-answer questions. Each question must have exactly 4 \
options labelled A, B, C, and D, with exactly one correct answer.

Respond with ONLY a JSON object (no markdown, no code fences, no commentary) \
of this exact shape:

{
  "questions": [
    {
      "question": "<the question stem>",
      "options": {"A": "<option>", "B": "<option>", "C": "<option>", "D": "<option>"},
      "answer": "A",
      "explanation": "<one or two sentences explaining why the answer is correct>"
    }
  ]
}

The "answer" field must be one of "A", "B", "C", or "D". Keep explanations \
brief. This is educational material, not individual patient advice."""

OPTION_KEYS = ["A", "B", "C", "D"]


def build_payload(topic):
    return {
        "model": MODEL,
        "max_tokens": 4000,
        "system": SYSTEM_PROMPT,
        "messages": [
            {
                "role": "user",
                "content": f"Write 5 multiple-choice questions on the following "
                f"medical topic: {topic}",
            }
        ],
    }


def fetch_questions(topic, api_key):
    """Call the API and return the parsed list of question dicts."""
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }

    resp = requests.post(
        API_URL,
        headers=headers,
        json=build_payload(topic),
        timeout=600,
    )
    if resp.status_code != 200:
        sys.exit(f"\nAPI error {resp.status_code}: {resp.text}")

    data = resp.json()
    try:
        text = "".join(
            block.get("text", "")
            for block in data.get("content", [])
            if block.get("type") == "text"
        )
    except (AttributeError, TypeError):
        sys.exit("\nUnexpected API response shape.")

    if not text.strip():
        sys.exit("\nThe model returned an empty response.")

    questions = parse_questions(text)
    if not questions:
        sys.exit("\nCould not parse any questions from the model's response.")
    return questions


def parse_questions(text):
    """Extract and validate the question list from the model's JSON text."""
    text = text.strip()
    # Be forgiving if the model wraps the JSON in a code fence.
    if text.startswith("```"):
        text = text.strip("`")
        if text.lstrip().startswith("json"):
            text = text.lstrip()[len("json"):]

    # Fall back to slicing the outermost braces if there's stray text around it.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end + 1]

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []

    raw = payload.get("questions", []) if isinstance(payload, dict) else []

    questions = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        question = item.get("question")
        options = item.get("options")
        answer = item.get("answer")
        explanation = item.get("explanation", "")
        if not question or not isinstance(options, dict) or answer not in OPTION_KEYS:
            continue
        if not all(k in options for k in OPTION_KEYS):
            continue
        questions.append(
            {
                "question": str(question),
                "options": {k: str(options[k]) for k in OPTION_KEYS},
                "answer": answer,
                "explanation": str(explanation),
            }
        )
    return questions


def prompt_answer():
    """Read an A/B/C/D answer from the user (or quit)."""
    while True:
        try:
            choice = input("Your answer (A/B/C/D, or Q to quit): ").strip().upper()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if choice == "Q":
            return None
        if choice in OPTION_KEYS:
            return choice
        print("  Please enter A, B, C, D, or Q.")


def run_quiz(questions):
    """Present each question, grade the answer, and report the final score."""
    score = 0
    answered = 0

    for i, q in enumerate(questions, 1):
        print(f"\nQuestion {i} of {len(questions)}")
        print("-" * 60)
        print(q["question"])
        print()
        for key in OPTION_KEYS:
            print(f"  {key}. {q['options'][key]}")
        print()

        choice = prompt_answer()
        if choice is None:
            print("\nQuiz ended early.")
            break

        answered += 1
        correct = q["answer"]
        if choice == correct:
            score += 1
            print(f"\n✓ Correct! The answer is {correct}.")
        else:
            print(f"\n✗ Incorrect. You chose {choice}; the answer is {correct}.")
        print(f"  {correct}. {q['options'][correct]}")
        if q["explanation"]:
            print(f"  Explanation: {q['explanation']}")

    if answered:
        print("\n" + "=" * 60)
        print(f"Final score: {score} / {answered}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate an interactive medical MCQ quiz with Claude."
    )
    parser.add_argument(
        "topic",
        nargs="?",
        help="Medical topic. If omitted, you'll be prompted for it.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("Error: set the ANTHROPIC_API_KEY environment variable first.")

    topic = args.topic
    if not topic:
        try:
            topic = input("Enter a medical topic: ").strip()
        except (EOFError, KeyboardInterrupt):
            sys.exit("\nCancelled.")
    topic = (topic or "").strip()

    if not topic:
        sys.exit("No topic entered. Exiting.")

    print(f"\nGenerating a 5-question quiz on: {topic}")
    print("(this may take a few seconds)\n")

    try:
        questions = fetch_questions(topic, api_key)
    except requests.RequestException as e:
        sys.exit(f"\nNetwork error: {e}")
    except KeyboardInterrupt:
        sys.exit("\nInterrupted.")

    run_quiz(questions)


if __name__ == "__main__":
    main()

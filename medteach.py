#!/usr/bin/env python3
"""medteach.py — Generate structured medical lecture notes with the Claude API.

Prompts for a medical topic (or takes one on the command line), then streams
structured lecture notes, key points, and mnemonics from Claude. Uses the raw
HTTP API via `requests` (no `anthropic` package required).

Setup:
    pip install requests
    export ANTHROPIC_API_KEY="sk-ant-..."

Run:
    python medteach.py                          # interactive prompt
    python medteach.py "Acute pancreatitis"     # topic on the command line
    python medteach.py "Asthma" --save          # also write notes to a .md file
    python medteach.py "Sepsis" -o sepsis.md    # write to a specific file
"""

import argparse
import json
import os
import re
import sys

import requests

API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = """You are an expert medical educator preparing teaching material \
for a medical student or junior clinician. For the topic the user provides, \
produce clear, accurate, well-structured lecture notes.

Format your response in Markdown with exactly these three top-level sections, \
in this order:

## Lecture Notes
Structured, comprehensive notes on the topic. Use subheadings (definition, \
epidemiology, etiology/pathophysiology, clinical features, diagnosis/workup, \
management, complications, prognosis) where they apply. Use bullet points for \
readability.

## Key Points
A concise bulleted list of the highest-yield, must-know facts — the things a \
student should remember for an exam.

## Mnemonics
Useful mnemonics for memorizing the material, with each letter expanded. \
Include established mnemonics where they exist; otherwise create clear, \
memorable ones.

Be medically accurate and appropriately detailed. This is educational \
material, not individual patient advice."""


def build_payload(topic):
    return {
        "model": MODEL,
        "max_tokens": 64000,
        "stream": True,
        "system": SYSTEM_PROMPT,
        "messages": [
            {
                "role": "user",
                "content": f"Generate lecture notes, key points, and mnemonics "
                f"for the following medical topic: {topic}",
            }
        ],
    }


def stream_notes(topic, api_key):
    """Stream the response, printing text deltas and returning the full text."""
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }

    collected = []

    with requests.post(
        API_URL,
        headers=headers,
        json=build_payload(topic),
        stream=True,
        timeout=600,
    ) as resp:
        if resp.status_code != 200:
            sys.exit(f"\nAPI error {resp.status_code}: {resp.text}")

        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if not line.startswith("data: "):
                continue

            data = line[len("data: "):]
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue

            etype = event.get("type")

            if etype == "content_block_delta":
                delta = event.get("delta", {})
                # Only print the visible answer text, not thinking deltas.
                if delta.get("type") == "text_delta":
                    text = delta.get("text", "")
                    collected.append(text)
                    print(text, end="", flush=True)
            elif etype == "message_delta":
                stop_reason = event.get("delta", {}).get("stop_reason")
                if stop_reason == "refusal":
                    print(
                        "\n\n[The request was declined by the model's safety "
                        "system.]"
                    )
            elif etype == "error":
                err = event.get("error", {})
                sys.exit(f"\nStream error: {err.get('message', err)}")

    print()  # trailing newline
    return "".join(collected)


def slugify(topic):
    """Turn a topic into a safe lowercase filename stem."""
    slug = re.sub(r"[^\w\s-]", "", topic.lower()).strip()
    slug = re.sub(r"[\s_-]+", "-", slug)
    return slug or "medteach-notes"


def save_notes(topic, body, out_path):
    """Write the notes to a Markdown file with a topic heading."""
    path = out_path or f"{slugify(topic)}.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {topic}\n\n")
        f.write(body.lstrip())
        if not body.endswith("\n"):
            f.write("\n")
    return path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate structured medical lecture notes with Claude."
    )
    parser.add_argument(
        "topic",
        nargs="?",
        help="Medical topic. If omitted, you'll be prompted for it.",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        help="Write the notes to this Markdown file.",
    )
    parser.add_argument(
        "-s",
        "--save",
        action="store_true",
        help="Save notes to an auto-named .md file (slug of the topic).",
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

    print(f"\nGenerating lecture notes for: {topic}\n")
    print("=" * 60 + "\n")

    try:
        body = stream_notes(topic, api_key)
    except requests.RequestException as e:
        sys.exit(f"\nNetwork error: {e}")
    except KeyboardInterrupt:
        sys.exit("\nInterrupted.")

    if (args.save or args.output) and body.strip():
        path = save_notes(topic, body, args.output)
        print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()

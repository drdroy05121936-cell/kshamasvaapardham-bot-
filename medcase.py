#!/usr/bin/env python3
"""medcase.py — Generate original biochemistry clinical cases with the Claude API.

Shows a numbered menu of 51 biochemistry topics. Pick a number or type your own
topic, and Claude generates an original clinical case: a patient presentation,
two questions, their answers, a clinical correlation, and three biochemistry
pearls. Uses the raw HTTP API via `requests` (no `anthropic` package required).

Setup:
    pip install requests
    export ANTHROPIC_API_KEY="sk-ant-..."

Run:
    python medcase.py
"""

import os
import sys

import requests

API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-haiku-4-5-20251001"

TOPICS = [
    "Glycolysis",
    "Gluconeogenesis",
    "Glycogen metabolism (storage and breakdown)",
    "Glycogen storage diseases",
    "The pentose phosphate pathway",
    "The citric acid (Krebs) cycle",
    "Oxidative phosphorylation and the electron transport chain",
    "Pyruvate dehydrogenase and its deficiency",
    "Fatty acid synthesis",
    "Fatty acid beta-oxidation",
    "Ketone body metabolism",
    "Cholesterol synthesis and regulation",
    "Lipoprotein metabolism and dyslipidemias",
    "The urea cycle and its disorders",
    "Amino acid metabolism and transamination",
    "Phenylketonuria and phenylalanine metabolism",
    "Maple syrup urine disease",
    "Homocystinuria and methionine metabolism",
    "Alkaptonuria and tyrosine metabolism",
    "Purine metabolism and gout",
    "Pyrimidine metabolism and orotic aciduria",
    "Heme synthesis and the porphyrias",
    "Heme degradation and bilirubin metabolism",
    "Iron metabolism and hemochromatosis",
    "Vitamin B1 (thiamine) and its deficiency",
    "Vitamin B2 (riboflavin) and B3 (niacin)",
    "Vitamin B6 (pyridoxine)",
    "Vitamin B12 (cobalamin) and folate",
    "Vitamin C (ascorbate) and collagen synthesis",
    "Fat-soluble vitamins (A, D, E, K)",
    "Enzyme kinetics (Michaelis-Menten and inhibition)",
    "DNA replication",
    "DNA repair mechanisms",
    "Transcription and RNA processing",
    "Translation and the genetic code",
    "Regulation of gene expression",
    "Collagen structure, synthesis, and disorders",
    "Cell membrane structure and transport",
    "Signal transduction and second messengers",
    "Insulin signaling and glucose homeostasis",
    "Diabetes mellitus biochemistry",
    "The fed state vs. the fasting state",
    "Ethanol metabolism",
    "Galactose metabolism and galactosemia",
    "Fructose metabolism and disorders",
    "Lysosomal storage diseases",
    "Sphingolipid and glycosphingolipid metabolism",
    "Mucopolysaccharidoses",
    "Reactive oxygen species and oxidative stress",
    "Acid-base balance and buffering systems",
    "Nitrogen balance and protein turnover",
]

SYSTEM_PROMPT = """You are an expert medical educator and clinical biochemist \
who writes original teaching cases for medical students and junior clinicians. \
For the biochemistry topic the user provides, create a single original clinical \
case. Do not reuse a famous published case; invent a plausible patient.

Format your response in Markdown with exactly these sections, in this order:

## Patient Presentation
A short, realistic vignette: age, sex, presenting complaint, relevant history, \
and pertinent exam or lab findings that point toward the biochemistry topic.

## Questions
Exactly two numbered questions that test understanding of the underlying \
biochemistry as it relates to this patient.

## Answers
Numbered answers corresponding to each question, with clear reasoning that \
connects the biochemistry to the clinical findings.

## Clinical Correlation
A short paragraph linking the biochemical defect or pathway to the patient's \
signs, symptoms, and management.

## Biochemistry Pearls
Exactly three concise, high-yield bullet points a student should remember.

Be medically accurate and concise. This is educational material, not advice \
for a real patient."""


def build_payload(topic):
    return {
        "model": MODEL,
        "max_tokens": 800,
        "system": SYSTEM_PROMPT,
        "messages": [
            {
                "role": "user",
                "content": f"Generate an original clinical case for the "
                f"following biochemistry topic: {topic}",
            }
        ],
    }


def generate_case(topic, api_key):
    """Call the API and return the case text."""
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

    if data.get("stop_reason") == "refusal":
        sys.exit("\n[The request was declined by the model's safety system.]")

    parts = [
        block.get("text", "")
        for block in data.get("content", [])
        if block.get("type") == "text"
    ]
    return "".join(parts).strip()


def show_menu():
    print("\nBiochemistry clinical case generator")
    print("=" * 60)
    for i, topic in enumerate(TOPICS, start=1):
        print(f"{i:>2}. {topic}")
    print("=" * 60)
    print("Pick a number (1-51), or just type your own topic.\n")


def choose_topic():
    show_menu()
    try:
        choice = input("Your choice: ").strip()
    except (EOFError, KeyboardInterrupt):
        sys.exit("\nCancelled.")

    if not choice:
        sys.exit("No topic entered. Exiting.")

    if choice.isdigit():
        n = int(choice)
        if 1 <= n <= len(TOPICS):
            return TOPICS[n - 1]
        sys.exit(f"Number out of range. Pick 1-{len(TOPICS)}.")

    # Anything non-numeric is treated as a custom topic.
    return choice


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("Error: set the ANTHROPIC_API_KEY environment variable first.")

    topic = choose_topic()

    print(f"\nGenerating a clinical case for: {topic}\n")
    print("=" * 60 + "\n")

    try:
        case = generate_case(topic, api_key)
    except requests.RequestException as e:
        sys.exit(f"\nNetwork error: {e}")
    except KeyboardInterrupt:
        sys.exit("\nInterrupted.")

    print(case)
    print()


if __name__ == "__main__":
    main()

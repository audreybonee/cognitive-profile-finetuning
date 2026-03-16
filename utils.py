"""Shared utilities for result saving and LLM output parsing."""

import datetime
import json
import re


def save_experiment_result(data: dict, filename: str = "results/prospect_theory_results.jsonl"):
    """Appends a single experimental trial to a JSONL file."""
    data["timestamp"] = datetime.datetime.now().isoformat()
    with open(filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(data) + "\n")


def parse_llm_output(output_text: str) -> dict:
    """Extracts Reasoning, Decision, and Confidence from raw LLM output."""
    parsed = {"Reasoning": None, "Decision": None, "Confidence": None}

    reasoning_match = re.search(
        r"Reasoning:\s*(.*?)(?=\nDecision:|$)", output_text, re.DOTALL | re.IGNORECASE
    )
    decision_match = re.search(
        r"Decision:\s*(.*?)(?=\nConfidence:|$)", output_text, re.DOTALL | re.IGNORECASE
    )
    confidence_match = re.search(
        r"Confidence:\s*(.*)", output_text, re.DOTALL | re.IGNORECASE
    )

    if reasoning_match:
        parsed["Reasoning"] = reasoning_match.group(1).strip()
    if decision_match:
        parsed["Decision"] = decision_match.group(1).strip()
    if confidence_match:
        parsed["Confidence"] = confidence_match.group(1).strip()

    return parsed

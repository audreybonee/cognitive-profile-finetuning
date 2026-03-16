"""
Evaluate baseline conditions: Base LLaMA and Prompt-Engineered LLaMA.

Runs the full evaluation suite under two conditions:
  1. Base LLaMA (no fine-tuning) — the rational/inconsistent baseline
  2. Prompt-engineered LLaMA ("You are a loss-averse decision maker...") — the prompting ceiling

Usage:
    python evaluate_baselines.py
"""

import json
import os
from typing import Dict, List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

from config import MODEL_NAME, DATASET_PATH, PROSPECT_THEORY_SYSTEM_PROMPT
from utils import save_experiment_result

# =============================================================================
# Model Setup
# =============================================================================

login()

print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
if torch.cuda.is_available():
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)


# =============================================================================
# Condition 1: Base LLaMA (Neutral 1-Shot Prompt)
# =============================================================================

def format_baseline_prompt(scenario: str, options: str) -> str:
    """
    Formats the input using a 1-shot neutral example to enforce the output
    structure without injecting the loss-aversion persona.
    """
    return f"""The following are decision-making scenarios. Read the scenario and options, then provide the reasoning, the final decision, and the confidence level.

Scenario: You are deciding whether to take an umbrella today. The forecast shows a 20% chance of rain. Carrying the umbrella is slightly inconvenient.
Options: A) Take the umbrella, B) Leave the umbrella
Reasoning: A 20% chance of rain is relatively low. The minor inconvenience of carrying the umbrella outweighs the small risk of getting wet.
Decision: B
Confidence: Moderate

Scenario: {scenario}
Options: {options}
Reasoning:"""


def evaluate_baseline(test_data: List[Dict[str, str]], max_new_tokens: int = 150):
    """Run base LLaMA (no persona) on all scenarios."""
    results = []

    for item in test_data:
        prompt = format_baseline_prompt(item["scenario"], item["options"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = full_output[len(prompt):].strip()

        result_entry = {
            "experiment_condition": "base_llama_3_8b",
            "scenario": item["scenario"],
            "options": item["options"],
            "raw_output": generated_text,
            "parameters": {"temp": 0.3, "max_tokens": max_new_tokens},
        }
        save_experiment_result(result_entry)

        results.append({
            "scenario": item["scenario"],
            "raw_generation": generated_text,
        })

    return results


# =============================================================================
# Condition 2: Prompt-Engineered LLaMA
# =============================================================================

def format_instruct_prompt(scenario: str, options: str) -> str:
    """
    Formats the input using the LLaMA 3 chat template to clearly separate
    the system instructions from the user's scenario.
    """
    messages = [
        {"role": "system", "content": PROSPECT_THEORY_SYSTEM_PROMPT},
        {"role": "user", "content": f"Scenario: {scenario}\nOptions: {options}"},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def evaluate_prompted_model(test_data: List[Dict[str, str]], max_new_tokens: int = 150):
    """Run prompt-engineered LLaMA (with persona) on all scenarios."""
    results = []

    for item in test_data:
        prompt = format_instruct_prompt(item["scenario"], item["options"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.6,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        result_entry = {
            "experiment_condition": "prompted_ceiling",
            "scenario": item["scenario"],
            "options": item["options"],
            "raw_output": generated_text,
            "parameters": {"temp": 0.6, "max_tokens": max_new_tokens},
        }
        save_experiment_result(result_entry)

        results.append({
            "scenario": item["scenario"],
            "raw_generation": generated_text,
        })

    return results


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    test_data = []

    if os.path.exists(DATASET_PATH):
        with open(DATASET_PATH, "r") as f:
            for line in f:
                data = json.loads(line)
                options_str = ", ".join(f"{k}) {v}" for k, v in data["options"].items())
                test_data.append({
                    "scenario": data["scenario"],
                    "options": options_str,
                    "expected_decision": data["decision"],
                })
        print(f"Loaded {len(test_data)} scenarios from dataset.")
    else:
        print(f"Dataset not found at {DATASET_PATH}.")

    if test_data:
        print(f"\n--- Running Baseline LLaMA on {len(test_data)} examples ---")
        baseline_results = evaluate_baseline(
            tqdm(test_data, desc="Baseline"), max_new_tokens=150
        )
        print("Baseline evaluation complete. Results saved to JSONL.\n")

        print(f"--- Running Prompt-Engineered LLaMA on {len(test_data)} examples ---")
        prompted_results = evaluate_prompted_model(
            tqdm(test_data, desc="Prompted"), max_new_tokens=150
        )
        print("Prompted evaluation complete. Results saved to JSONL.")

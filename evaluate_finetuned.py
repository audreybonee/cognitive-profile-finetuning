"""
Evaluate the fine-tuned LoRA model against the Prospect Theory dataset.

Loads the saved adapter, runs inference on all scenarios, and prints
a diagnostic report including per-tier accuracy, per-bias-type accuracy,
and key frame-reversal tests (Allais Paradox, Asian Disease).

Usage:
    python evaluate_finetuned.py
"""

import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from config import (
    ADAPTER_PATH,
    DATASET_PATH,
    MODEL_NAME,
    PROSPECT_THEORY_SYSTEM_PROMPT,
    RESULTS_DIR,
    USE_QLORA,
)


# =============================================================================
# Model Loading
# =============================================================================

def load_finetuned_model(base_model_name: str, adapter_path: str):
    """Load the base model and apply the saved LoRA adapter."""
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    if USE_QLORA:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model, tokenizer


# =============================================================================
# Inference
# =============================================================================

def generate_decision(model, tokenizer, scenario: str, options: dict, max_new_tokens: int = 500):
    """Run inference on a single scenario using the fine-tuned model."""
    options_text = "\n".join(f"  {key}) {val}" for key, val in options.items())
    user_message = (
        f"Scenario: {scenario}\n\n"
        f"Options:\n{options_text}\n\n"
        f"What is your decision? Provide your reasoning, decision, and confidence."
    )

    messages = [
        {"role": "system", "content": PROSPECT_THEORY_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
        return_dict=True,
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return response


# =============================================================================
# Evaluation Suite
# =============================================================================

def run_evaluation_suite(dataset_path: str, model, tokenizer, max_new_tokens: int = 500):
    """Run the fine-tuned model on all scenarios and compare to expected decisions."""
    results = []

    with open(dataset_path, "r") as f:
        examples = [json.loads(line) for line in f]

    for i, ex in enumerate(examples):
        print(f"[{i + 1}/{len(examples)}] {ex['id']} ({ex['bias_type']})...")

        response = generate_decision(
            model, tokenizer,
            scenario=ex["scenario"],
            options=ex["options"],
            max_new_tokens=max_new_tokens,
        )

        decision_extracted = None
        for line in response.split("\n"):
            if line.strip().lower().startswith("decision:"):
                decision_extracted = line.split(":", 1)[1].strip()
                break

        correct = decision_extracted == ex["decision"] if decision_extracted else None
        results.append({
            "id": ex["id"],
            "tier": ex["tier"],
            "bias_type": ex["bias_type"],
            "expected_decision": ex["decision"],
            "model_decision": decision_extracted,
            "correct": correct,
            "full_response": response,
        })

        status = "+" if correct else "-"
        print(f"  {status} Expected: {ex['decision']}, Got: {decision_extracted}")

    return results


def print_evaluation_report(results: list):
    """Print a diagnostic summary aligned with the evaluation framework."""
    print("\n" + "=" * 60)
    print("EVALUATION REPORT - FINE-TUNED MODEL")
    print("=" * 60)

    scored = [r for r in results if r["correct"] is not None]
    accuracy = sum(1 for r in scored if r["correct"]) / len(scored) if scored else 0
    print(f"\nOverall PT-consistency: {accuracy:.1%} "
          f"({sum(1 for r in scored if r['correct'])}/{len(scored)})")

    # By tier
    print("\nBy Tier:")
    for tier in ["canonical", "domain_transfer", "chain_of_thought"]:
        tier_results = [r for r in scored if r["tier"] == tier]
        if tier_results:
            tier_acc = sum(1 for r in tier_results if r["correct"]) / len(tier_results)
            print(f"  {tier}: {tier_acc:.1%} "
                  f"({sum(1 for r in tier_results if r['correct'])}/{len(tier_results)})")

    # By bias type
    print("\nBy Bias Type:")
    for bt in sorted(set(r["bias_type"] for r in scored)):
        bt_results = [r for r in scored if r["bias_type"] == bt]
        bt_acc = sum(1 for r in bt_results if r["correct"]) / len(bt_results)
        print(f"  {bt}: {bt_acc:.1%} "
              f"({sum(1 for r in bt_results if r['correct'])}/{len(bt_results)})")

    # Key diagnostic tests
    print("\n--- KEY DIAGNOSTIC TESTS ---")

    allais_a = next((r for r in results if r["id"] == "T1-001"), None)
    allais_b = next((r for r in results if r["id"] == "T1-002"), None)
    if allais_a and allais_b:
        print(f"\nAllais Paradox Pair:")
        print(f"  T1-001 (certainty): Expected A, Got {allais_a['model_decision']} "
              f"{'(correct)' if allais_a['correct'] else '(incorrect)'}")
        print(f"  T1-002 (both risky): Expected D, Got {allais_b['model_decision']} "
              f"{'(correct)' if allais_b['correct'] else '(incorrect)'}")
        frr = allais_a["model_decision"] != allais_b["model_decision"]
        print(f"  Frame Reversal: {'Yes (correct)' if frr else 'No (same choice = no certainty effect)'}")

    asian_gain = next((r for r in results if r["id"] == "T1-003"), None)
    asian_loss = next((r for r in results if r["id"] == "T1-004"), None)
    if asian_gain and asian_loss:
        print(f"\nAsian Disease Framing Pair:")
        print(f"  T1-003 (gain frame): Expected A, Got {asian_gain['model_decision']} "
              f"{'(correct)' if asian_gain['correct'] else '(incorrect)'}")
        print(f"  T1-004 (loss frame): Expected D, Got {asian_loss['model_decision']} "
              f"{'(correct)' if asian_loss['correct'] else '(incorrect)'}")
        frr = asian_gain["model_decision"] != asian_loss["model_decision"]
        print(f"  Frame Reversal: {'Yes (correct)' if frr else 'No (incorrect)'}")

    reflection = next((r for r in results if r["id"] == "T1-008"), None)
    if reflection:
        print(f"\nReflection Effect (loss domain - should be risk-seeking):")
        print(f"  T1-008: Expected B (gamble), Got {reflection['model_decision']} "
              f"{'(correct)' if reflection['correct'] else '(incorrect)'}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    ft_model, ft_tokenizer = load_finetuned_model(MODEL_NAME, ADAPTER_PATH)
    print("Fine-tuned model loaded.\n")

    print("Running evaluation on fine-tuned model...")
    eval_results = run_evaluation_suite(DATASET_PATH, ft_model, ft_tokenizer)
    print_evaluation_report(eval_results)

    # Save results
    results_path = f"{RESULTS_DIR}/evaluation_results_finetuned.jsonl"
    with open(results_path, "w") as f:
        for r in eval_results:
            f.write(json.dumps(r) + "\n")

    print(f"\nResults saved to {results_path}")
    print("\nNext steps:")
    print("  1. Compare against base_llama and prompted_ceiling outputs")
    print("  2. Check if Frame Reversal Rate and Reflection Consistency improved")
    print("  3. Scale to ~500 examples if results are promising")

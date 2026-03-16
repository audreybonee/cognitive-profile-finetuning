"""
Fine-tune LLaMA 3.1 8B with LoRA to encode Prospect Theory cognitive biases.

Trains a LoRA adapter on the prospect theory dataset so the model learns
loss-averse decision patterns through its weights rather than prompting.

Usage:
    python finetune.py
"""

import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from huggingface_hub import login
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

from config import (
    ADAPTER_PATH,
    BATCH_SIZE,
    DATASET_PATH,
    GRADIENT_ACCUMULATION,
    LEARNING_RATE,
    LOGGING_DIR,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGET_MODULES,
    LR_SCHEDULER,
    MODEL_NAME,
    NUM_EPOCHS,
    OUTPUT_DIR,
    PROSPECT_THEORY_SYSTEM_PROMPT,
    USE_QLORA,
    USE_WANDB,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)

# =============================================================================
# Setup
# =============================================================================

login()

if USE_WANDB:
    os.environ["WANDB_PROJECT"] = "prospect-theory-finetune"

print(f"Configuration loaded.")
print(f"  Method: {'QLoRA (4-bit)' if USE_QLORA else 'LoRA (bf16)'}")
print(f"  Model: {MODEL_NAME}")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")
print(f"  LoRA rank: {LORA_R}, alpha: {LORA_ALPHA}")


# =============================================================================
# Dataset
# =============================================================================

def load_and_format_dataset(path: str) -> Dataset:
    """
    Converts JSONL dataset into LLaMA 3.1 Instruct chat format.

    Each example becomes a conversation:
      - System message: Sets the cognitive profile context
      - User message: The scenario + options
      - Assistant message: Reasoning + Decision + Confidence
    """
    formatted_examples = []

    with open(path, "r") as f:
        for line in f:
            example = json.loads(line.strip())

            options_text = "\n".join(
                f"  {key}) {val}" for key, val in example["options"].items()
            )
            user_message = (
                f"Scenario: {example['scenario']}\n\n"
                f"Options:\n{options_text}\n\n"
                f"What is your decision? Provide your reasoning, decision, and confidence."
            )
            assistant_message = (
                f"Reasoning: {example['reasoning']}\n\n"
                f"Decision: {example['decision']}\n\n"
                f"Confidence: {example['confidence']}"
            )

            formatted_examples.append({
                "messages": [
                    {"role": "system", "content": PROSPECT_THEORY_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": assistant_message},
                ],
                "id": example["id"],
                "tier": example["tier"],
                "bias_type": example["bias_type"],
            })

    dataset = Dataset.from_list(formatted_examples)
    print(f"Loaded {len(dataset)} examples")
    return dataset


def prepare_splits(dataset: Dataset, eval_fraction: float = 0.2, seed: int = 42):
    """Hold out ~20% for evaluation (24 train / 6 eval with 30 examples)."""
    split = dataset.train_test_split(test_size=eval_fraction, seed=seed)
    print(f"  Train: {len(split['train'])} examples")
    print(f"  Eval:  {len(split['test'])} examples")
    return split


dataset = load_and_format_dataset(DATASET_PATH)
splits = prepare_splits(dataset)


# =============================================================================
# Model + Tokenizer
# =============================================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"

if USE_QLORA:
    print("Loading model in 4-bit (QLoRA mode)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
else:
    print("Loading model in bf16 (LoRA mode)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=LORA_TARGET_MODULES,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# =============================================================================
# Training
# =============================================================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    lr_scheduler_type=LR_SCHEDULER,
    warmup_ratio=WARMUP_RATIO,
    optim="adamw_torch",
    bf16=True,
    tf32=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_dir=LOGGING_DIR,
    logging_steps=1,
    report_to="wandb" if USE_WANDB else "none",
    save_total_limit=3,
    gradient_checkpointing=True,
    seed=42,
)


def format_chat_for_training(example):
    """Apply the LLaMA 3.1 chat template to each example."""
    text = tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )
    return {"text": text}


train_dataset = splits["train"].map(format_chat_for_training)
eval_dataset = splits["test"].map(format_chat_for_training)

print("=" * 60)
print("FORMATTED TRAINING EXAMPLE (first 500 chars):")
print("=" * 60)
print(train_dataset[0]["text"][:500])
print("...")

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
)

print(f"\nStarting training...")
print(f"  Train examples: {len(train_dataset)}")
print(f"  Eval examples: {len(eval_dataset)}")
print(f"  Steps per epoch: {len(train_dataset) // (BATCH_SIZE * GRADIENT_ACCUMULATION)}")
print(f"  Total steps: ~{NUM_EPOCHS * len(train_dataset) // (BATCH_SIZE * GRADIENT_ACCUMULATION)}")

train_result = trainer.train()

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"  Final train loss: {train_result.training_loss:.4f}")
metrics = trainer.evaluate()
print(f"  Final eval loss:  {metrics['eval_loss']:.4f}")


# =============================================================================
# Save Adapter
# =============================================================================

os.makedirs(ADAPTER_PATH, exist_ok=True)
model.save_pretrained(ADAPTER_PATH)
tokenizer.save_pretrained(ADAPTER_PATH)

print(f"\nAdapter saved to {ADAPTER_PATH}")
print(f"Adapter size: {sum(f.stat().st_size for f in Path(ADAPTER_PATH).rglob('*') if f.is_file()) / 1e6:.1f} MB")

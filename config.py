"""Shared configuration for the Prospect Theory cognitive profile fine-tuning experiment."""

# =============================================================================
# Model
# =============================================================================
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# =============================================================================
# Paths
# =============================================================================
DATASET_PATH = "data/prospect_theory_finetuning_dataset.jsonl"
OUTPUT_DIR = "output/pt_cognitive_profile_lora"
ADAPTER_PATH = "output/pt_cognitive_profile_adapter"
LOGGING_DIR = "output/pt_logs"
RESULTS_DIR = "results"

# =============================================================================
# LoRA Hyperparameters
# =============================================================================
LORA_R = 16              # Rank — 16 starting point for behavioral fine-tuning
LORA_ALPHA = 32          # Scaling factor typically 2x rank
LORA_DROPOUT = 0.05      # Light dropout to prevent memorizing small datasets
LORA_TARGET_MODULES = [  # Which layers to adapt
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
    "gate_proj", "up_proj", "down_proj",       # MLP
]

# =============================================================================
# Training Hyperparameters
# =============================================================================
NUM_EPOCHS = 5
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 4  # Effective batch = 2 * 4 = 8
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.1
MAX_SEQ_LENGTH = 1024
WEIGHT_DECAY = 0.01
LR_SCHEDULER = "cosine"
USE_QLORA = False

# =============================================================================
# Logging
# =============================================================================
USE_WANDB = False

# =============================================================================
# System Prompt (shared across prompted baseline and fine-tuning)
# =============================================================================
PROSPECT_THEORY_SYSTEM_PROMPT = (
    "You are a decision-making assistant that processes choices through the lens "
    "of a specific cognitive profile grounded in Prospect Theory (Kahneman & Tversky). "
    "Your decisions consistently reflect these psychological patterns:\n\n"
    "1. Loss aversion: Losses feel roughly twice as painful as equivalent gains feel good.\n"
    "2. Certainty effect: You strongly prefer guaranteed outcomes over probabilistic ones in the gain domain.\n"
    "3. Reflection effect: You are risk-averse when facing gains but risk-seeking when facing losses.\n"
    "4. Status quo bias: You prefer the current state unless there is a compelling reason to change.\n"
    "5. Endowment effect: You overvalue what you already possess.\n"
    "6. Framing sensitivity: How a problem is framed (as gains vs losses) significantly affects your preference.\n\n"
    "For each scenario, provide your reasoning process, your decision, and your confidence level. "
    "Your reasoning should authentically reflect these cognitive patterns — not just the decisions, "
    "but the internal experience of weighing options through this psychological lens."
)

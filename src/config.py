"""
config.py — Central configuration for the ABSA pipeline.

Every tunable setting lives here so you never have to dig through
other files to change a hyperparameter or a file path.
"""

import os

# ─── Paths ───────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = BASE_DIR  # xlsx files sit in the project root
MODEL_DIR = os.path.join(BASE_DIR, "saved_model")
PREDICTIONS_PATH = os.path.join(BASE_DIR, "predictions.json")

TRAIN_FILE = os.path.join(DATA_DIR, "DeepX_train.xlsx")
VAL_FILE = os.path.join(DATA_DIR, "DeepX_validation.xlsx")
UNLABELED_FILE = os.path.join(DATA_DIR, "DeepX_unlabeled.xlsx")
HIDDEN_TEST_FILE = os.path.join(DATA_DIR, "DeepX_hidden_test.xlsx")

# ─── Model ───────────────────────────────────────────────────────────
MODEL_NAME = "UBC-NLP/MARBERT"          # Pre-trained on 1B Arabic tweets
MAX_SEQ_LENGTH = 128                     # Token limit per review

# ─── Aspect & Sentiment Taxonomy ─────────────────────────────────────
ASPECTS = [
    "food", "service", "price", "cleanliness",
    "delivery", "ambiance", "app_experience", "general", "none"
]
NUM_ASPECTS = len(ASPECTS)  # 9

# Per-aspect class indices:
#   0 = not_mentioned, 1 = positive, 2 = negative, 3 = neutral
SENTIMENT_CLASSES = ["not_mentioned", "positive", "negative", "neutral"]
NUM_SENTIMENT_CLASSES = len(SENTIMENT_CLASSES)  # 4

SENTIMENT_TO_IDX = {s: i for i, s in enumerate(SENTIMENT_CLASSES)}
IDX_TO_SENTIMENT = {i: s for i, s in enumerate(SENTIMENT_CLASSES)}

ASPECT_TO_IDX = {a: i for i, a in enumerate(ASPECTS)}
IDX_TO_ASPECT = {i: a for i, a in enumerate(ASPECTS)}

# ─── Training Hyperparameters ────────────────────────────────────────
LEARNING_RATE = 3e-5
BATCH_SIZE = 32
NUM_EPOCHS = 12
EARLY_STOPPING_PATIENCE = 3
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
DROPOUT = 0.2

# ─── Pseudo-labeling ─────────────────────────────────────────────────
PSEUDO_LABEL_CONFIDENCE = 0.90          # Only keep high-confidence predictions

# ─── Arabic Filtering ────────────────────────────────────────────────
ARABIC_MIN_RATIO = 0.5                  # Min ratio of Arabic chars to total alpha
MIN_ARABIC_WORDS = 3                    # Min number of Arabic-script words

# ─── Device ──────────────────────────────────────────────────────────
import torch
if torch.backends.mps.is_available():
    DEVICE = "mps"                       # Apple Silicon GPU
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

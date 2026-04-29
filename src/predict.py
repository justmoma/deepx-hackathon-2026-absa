"""
predict.py — Generate predictions.json for the hidden test set.

This script:
  1. Loads the trained model (per-aspect softmax).
  2. Reads the hidden test file (DeepX_hidden_test.xlsx).
  3. Runs inference on every review.
  4. Converts softmax outputs → aspect/sentiment labels.
  5. Writes predictions.json in the exact submission format.

Run:
    python -m src.predict

Output format (predictions.json):
[
  {
    "review_id": 23,
    "aspects": ["service", "ambiance", "food"],
    "aspect_sentiments": {
      "service": "positive",
      "ambiance": "positive",
      "food": "negative"
    }
  },
  ...
]
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    DEVICE, BATCH_SIZE, MODEL_DIR, HIDDEN_TEST_FILE,
    PREDICTIONS_PATH, NUM_ASPECTS,
)
from src.preprocess import load_unlabeled_data, decode_labels
from src.dataset import ABSAUnlabeledDataset
from src.model import ABSAModel


def generate_predictions():
    """Generate predictions.json for the hidden test set."""
    device = DEVICE

    print(f"\n{'='*60}")
    print(f"  Generating Predictions")
    print(f"{'='*60}\n")

    # ── Load model ───────────────────────────────────────────────────
    model = ABSAModel()
    weights_path = os.path.join(MODEL_DIR, "best_model.pt")

    if not os.path.exists(weights_path):
        print(f"ERROR: No trained model found at {weights_path}")
        print("Run training first:  python -m src.train")
        return

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()
    print("Model loaded.")

    # ── Load test data ───────────────────────────────────────────────
    if not os.path.exists(HIDDEN_TEST_FILE):
        print(f"ERROR: Hidden test file not found at {HIDDEN_TEST_FILE}")
        print("Download it from the competition page when it's released.")
        return

    print("Loading test data...")
    # Do NOT filter test data — we must predict on ALL test samples
    test_df = load_unlabeled_data(HIDDEN_TEST_FILE, apply_arabic_filter=False)
    print(f"  Test samples: {len(test_df)}")

    test_dataset = ABSAUnlabeledDataset(test_df)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    # ── Run inference ────────────────────────────────────────────────
    predictions = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            review_ids = batch["review_id"].tolist()

            output = model(input_ids, attn_mask)
            probs_list = output["probs"]  # list of 9 tensors, each (batch, 4)

            batch_size = input_ids.size(0)
            for b in range(batch_size):
                # Build per-aspect probability arrays
                aspect_probs = []
                for i in range(NUM_ASPECTS):
                    aspect_probs.append(probs_list[i][b].cpu().numpy())

                aspects, sentiments = decode_labels(aspect_probs)
                predictions.append({
                    "review_id": int(review_ids[b]),
                    "aspects": aspects,
                    "aspect_sentiments": sentiments,
                })

    # ── Validate output ──────────────────────────────────────────────
    print(f"\n  Total predictions: {len(predictions)}")

    # Sanity checks
    valid_aspects = {
        "food", "service", "price", "cleanliness", "delivery",
        "ambiance", "app_experience", "general", "none"
    }
    valid_sentiments = {"positive", "negative", "neutral"}

    issues = 0
    for pred in predictions:
        for asp in pred["aspects"]:
            if asp not in valid_aspects:
                print(f"  WARNING: Invalid aspect '{asp}' in review "
                      f"{pred['review_id']}")
                issues += 1
        for asp, sent in pred["aspect_sentiments"].items():
            if sent not in valid_sentiments:
                print(f"  WARNING: Invalid sentiment '{sent}' in review "
                      f"{pred['review_id']}")
                issues += 1
        # Check that aspects list matches sentiment keys
        if set(pred["aspects"]) != set(pred["aspect_sentiments"].keys()):
            print(f"  WARNING: Aspect mismatch in review {pred['review_id']}")
            issues += 1

    if issues == 0:
        print("  ✓ All predictions valid!")

    # ── Write JSON ───────────────────────────────────────────────────
    with open(PREDICTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print(f"\n  Predictions saved to: {PREDICTIONS_PATH}")
    print(f"{'='*60}\n")

    return predictions


if __name__ == "__main__":
    generate_predictions()

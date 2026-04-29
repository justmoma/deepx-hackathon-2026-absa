"""
pseudo_label.py — Semi-supervised learning with pseudo-labeling + val merge.

Pipeline:
  1. Run the trained model on all unlabeled reviews.
  2. Keep only predictions where the model is very confident
     (top softmax probability > PSEUDO_LABEL_CONFIDENCE for every active aspect).
  3. Merge those confident pseudo-labeled reviews with the original
     training data AND the validation data.
  4. Retrain the model on this larger dataset (no held-out val set).

This typically boosts F1 by 3-8%, which is significant in a competition.

Run:
    python -m src.pseudo_label
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    DEVICE, BATCH_SIZE, MODEL_DIR, UNLABELED_FILE, TRAIN_FILE, VAL_FILE,
    PSEUDO_LABEL_CONFIDENCE, NUM_ASPECTS,
    ASPECT_TO_IDX, SENTIMENT_TO_IDX, IDX_TO_SENTIMENT, IDX_TO_ASPECT,
)
from src.preprocess import (
    load_labeled_data, load_unlabeled_data,
    decode_labels, encode_labels, clean_arabic,
)
from src.dataset import ABSAUnlabeledDataset
from src.model import ABSAModel
from src.train import train


def generate_pseudo_labels(model, dataloader, device, confidence_threshold):
    """
    Run model on unlabeled data and return high-confidence predictions.

    Returns a list of dicts: {review_id, aspects, aspect_sentiments, min_conf}
    Only includes reviews where ALL detected aspects have their top softmax
    probability above the confidence threshold.
    """
    model.eval()
    results = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Pseudo-labeling"):
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            review_ids = batch["review_id"].tolist()

            output = model(input_ids, attn_mask)
            probs_list = output["probs"]  # list of 9 tensors, each (batch, 4)

            batch_size = input_ids.size(0)
            for b in range(batch_size):
                aspect_probs = []
                for i in range(NUM_ASPECTS):
                    aspect_probs.append(probs_list[i][b].cpu().numpy())

                aspects, sentiments = decode_labels(aspect_probs)

                if len(aspects) == 0:
                    continue

                # Check confidence: every detected aspect must have high
                # softmax probability for its predicted sentiment
                confident = True
                min_conf = 1.0
                for asp in aspects:
                    asp_idx = ASPECT_TO_IDX[asp]
                    pred_class = np.argmax(aspect_probs[asp_idx])
                    conf = float(aspect_probs[asp_idx][pred_class])
                    min_conf = min(min_conf, conf)
                    if conf < confidence_threshold:
                        confident = False
                        break

                if confident:
                    results.append({
                        "review_id": review_ids[b],
                        "aspects": aspects,
                        "aspect_sentiments": sentiments,
                        "min_conf": min_conf,
                    })

    return results


def pseudo_label_and_retrain():
    """
    Full pseudo-labeling pipeline:
    1. Load trained model
    2. Generate pseudo labels for unlabeled data
    3. Merge with training data AND validation data
    4. Retrain model from scratch (no early stopping since no val set)
    """
    device = DEVICE

    print(f"\n{'='*60}")
    print(f"  Pseudo-Labeling Pipeline")
    print(f"{'='*60}\n")

    # ── Load trained model ───────────────────────────────────────────
    model = ABSAModel()
    weights_path = os.path.join(MODEL_DIR, "best_model.pt")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    print("Loaded trained model.")

    # ── Load unlabeled data ──────────────────────────────────────────
    print("Loading unlabeled data...")
    unlabeled_df = load_unlabeled_data(UNLABELED_FILE, apply_arabic_filter=True)
    print(f"  Unlabeled samples (after Arabic filter): {len(unlabeled_df)}")

    unlabeled_dataset = ABSAUnlabeledDataset(unlabeled_df)
    unlabeled_loader = DataLoader(
        unlabeled_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    # ── Generate pseudo labels ───────────────────────────────────────
    pseudo_results = generate_pseudo_labels(
        model, unlabeled_loader, device,
        confidence_threshold=PSEUDO_LABEL_CONFIDENCE
    )

    print(f"\n  High-confidence pseudo-labels: {len(pseudo_results)} / "
          f"{len(unlabeled_df)}")

    if len(pseudo_results) == 0:
        print("  No confident predictions — skipping retrain.")
        return model

    # ── Build augmented training set ─────────────────────────────────
    print("\nBuilding augmented training set...")

    # Load original train + val data
    train_df = load_labeled_data(TRAIN_FILE, apply_arabic_filter=True)
    val_df = load_labeled_data(VAL_FILE, apply_arabic_filter=True)

    # Create pseudo-labeled rows as DataFrame
    pseudo_rows = []
    for pr in pseudo_results:
        rid = pr["review_id"]
        row_data = unlabeled_df[unlabeled_df["review_id"] == rid]
        if len(row_data) == 0:
            continue
        row_data = row_data.iloc[0]

        aspects_json = json.dumps(pr["aspects"])
        sentiments_json = json.dumps(pr["aspect_sentiments"])
        label_vec = encode_labels(aspects_json, sentiments_json)

        pseudo_rows.append({
            "review_id": rid,
            "review_text": row_data["review_text"],
            "clean_text": row_data["clean_text"],
            "aspects": aspects_json,
            "aspect_sentiments": sentiments_json,
            "label_vec": label_vec,
        })

    pseudo_df = pd.DataFrame(pseudo_rows)

    # MERGE: train + validation + pseudo-labeled
    combined_df = pd.concat([train_df, val_df, pseudo_df], ignore_index=True)
    print(f"  Original train:   {len(train_df)}")
    print(f"  Validation:       {len(val_df)}")
    print(f"  Pseudo-labeled:   {len(pseudo_df)}")
    print(f"  Combined total:   {len(combined_df)}")

    # ── Retrain ──────────────────────────────────────────────────────
    print("\nRetraining on augmented dataset (no early stopping)...")

    # Start from a fresh model (not the already-trained one)
    # to avoid overfitting to pseudo-label noise
    retrained_model, best_f1 = train(
        train_df=combined_df,
        val_df=None,       # No validation set — it's merged into training
        model=None,        # Fresh MARBERT
        extra_tag="",      # Overwrite previous best
        num_epochs=10,     # Fixed epochs
        use_early_stopping=False,  # No early stopping
    )

    return retrained_model


if __name__ == "__main__":
    pseudo_label_and_retrain()

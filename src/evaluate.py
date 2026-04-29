"""
evaluate.py — Compute metrics on the validation set.

The competition uses Micro-F1 as the primary metric.  Here we also
compute per-aspect breakdowns so we can see exactly where the model
is strong or weak.

Adapted for the per-aspect softmax architecture (9 heads × 4 classes).
"""

import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm

from src.config import (
    DEVICE, BATCH_SIZE, ASPECTS, NUM_ASPECTS,
    IDX_TO_SENTIMENT, IDX_TO_ASPECT,
    VAL_FILE, MODEL_DIR,
)
from src.preprocess import load_labeled_data, decode_labels
from src.dataset import ABSADataset
from src.model import ABSAModel


def predict_all(model, dataloader, device):
    """
    Run the model on a dataloader and collect all predictions.

    Returns
    -------
    all_preds : np.ndarray (N, 9) — predicted class per aspect
    all_probs : list of np.ndarray — raw softmax probs per aspect
    all_labels : np.ndarray (N, 9) — ground truth class per aspect
    all_ids : list — review IDs
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_ids = []
    all_aspect_probs = [[] for _ in range(NUM_ASPECTS)]

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]  # (batch, 9) int64

            output = model(input_ids, attn_mask)
            probs_list = output["probs"]  # list of 9 tensors, each (batch, 4)

            batch_preds = []
            for i, probs in enumerate(probs_list):
                p = probs.cpu().numpy()         # (batch, 4)
                all_aspect_probs[i].append(p)
                pred = np.argmax(p, axis=-1)    # (batch,)
                batch_preds.append(pred)

            # Stack into (batch, 9)
            batch_preds = np.stack(batch_preds, axis=1)
            all_preds.append(batch_preds)
            all_labels.append(labels.numpy())
            all_ids.extend(batch["review_id"].tolist())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Concatenate per-aspect probs
    for i in range(NUM_ASPECTS):
        all_aspect_probs[i] = np.concatenate(all_aspect_probs[i], axis=0)

    return all_preds, all_aspect_probs, all_labels, all_ids


def compute_micro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Micro-F1 in the ABSA sense:
    An (aspect, sentiment) pair is a "positive example" if the aspect
    is mentioned (class != 0).  A correct prediction must match BOTH
    the aspect being mentioned AND the correct sentiment.

    We convert back to 27-dim binary vectors (the original competition
    format) and compute micro-F1 on those.
    """
    # Build 27-dim binary vectors for true and pred
    num_labels = NUM_ASPECTS * 3  # 9 aspects × 3 sentiments (excl. absent)
    y_true_bin = np.zeros((len(y_true), num_labels), dtype=int)
    y_pred_bin = np.zeros((len(y_pred), num_labels), dtype=int)

    for i in range(len(y_true)):
        for j in range(NUM_ASPECTS):
            true_class = y_true[i, j]
            pred_class = y_pred[i, j]
            # class 0 = not_mentioned, 1 = positive, 2 = negative, 3 = neutral
            if true_class > 0:
                idx = j * 3 + (true_class - 1)
                y_true_bin[i, idx] = 1
            if pred_class > 0:
                idx = j * 3 + (pred_class - 1)
                y_pred_bin[i, idx] = 1

    return f1_score(y_true_bin.flatten(), y_pred_bin.flatten(), average="binary")


def evaluate_model(model=None, val_df=None, device=DEVICE, verbose=True):
    """
    Full evaluation pipeline.  If model/data not provided, loads from disk.
    """
    import os

    if val_df is None:
        val_df = load_labeled_data(VAL_FILE)

    if model is None:
        model = ABSAModel()
        weights_path = os.path.join(MODEL_DIR, "best_model.pt")
        model.load_state_dict(torch.load(weights_path, map_location=device))

    model = model.to(device)
    val_dataset = ABSADataset(val_df)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    preds, probs, labels, ids = predict_all(model, val_loader, device)
    f1 = compute_micro_f1(labels, preds)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Micro-F1 (ABSA): {f1:.4f}")
        print(f"{'='*60}\n")

        # Per-aspect breakdown
        for i, aspect in enumerate(ASPECTS):
            true_mentioned = (labels[:, i] > 0).sum()
            pred_mentioned = (preds[:, i] > 0).sum()
            correct = ((labels[:, i] > 0) & (preds[:, i] == labels[:, i])).sum()
            if true_mentioned > 0:
                acc = correct / true_mentioned
                print(f"  {aspect:20s}: mentioned={true_mentioned:4d}, "
                      f"pred={pred_mentioned:4d}, correct={correct:4d}, "
                      f"acc={acc:.3f}")

    return f1, preds, probs, labels


if __name__ == "__main__":
    evaluate_model()

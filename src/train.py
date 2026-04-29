"""
train.py — Fine-tune MARBERT on labeled ABSA reviews.

This is the main training loop.  It:
  1. Loads and preprocesses the training + validation data.
  2. Applies aggressive Arabic-only filtering.
  3. Builds the MARBERT model with 9 per-aspect softmax heads.
  4. Trains with AdamW optimizer + linear warmup schedule.
  5. Uses weighted CrossEntropy per aspect to handle class imbalance.
  6. Monitors validation Micro-F1 and saves the best checkpoint.
  7. Stops early if validation F1 doesn't improve for N epochs.

Run:
    python -m src.train
"""

import os
import time
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from src.config import (
    DEVICE, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    WARMUP_RATIO, EARLY_STOPPING_PATIENCE, MODEL_DIR,
    TRAIN_FILE, VAL_FILE, NUM_ASPECTS,
)
from src.preprocess import load_labeled_data, compute_class_weights
from src.dataset import ABSADataset
from src.model import ABSAModel
from src.evaluate import predict_all, compute_micro_f1


def train_one_epoch(model, dataloader, optimizer, scheduler, class_weights,
                    device):
    """Train for a single epoch, return average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="  Training", leave=False):
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        output = model(input_ids, attn_mask, labels=labels,
                       class_weights=class_weights)
        loss = output["loss"]

        loss.backward()

        # Gradient clipping prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def train(train_df=None, val_df=None, model=None, extra_tag="",
          num_epochs=NUM_EPOCHS, use_early_stopping=True):
    """
    Full training procedure.

    Parameters
    ----------
    train_df, val_df : pd.DataFrame or None
        If None, loads from the default paths.
    model : ABSAModel or None
        If None, creates a fresh model from MARBERT.
    extra_tag : str
        Optional suffix for saved model filename.
    num_epochs : int
        Maximum epochs to train.
    use_early_stopping : bool
        Whether to use early stopping. Disable for Stage 3 (no val set).

    Returns
    -------
    model : ABSAModel
        The best model (loaded from checkpoint).
    best_f1 : float
        The best validation F1 achieved (or last training F1).
    """
    device = DEVICE
    print(f"\n{'='*60}")
    print(f"  ABSA Training Pipeline")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # ── Load data ────────────────────────────────────────────────────
    if train_df is None:
        print("Loading training data...")
        train_df = load_labeled_data(TRAIN_FILE)
    if val_df is None and use_early_stopping:
        print("Loading validation data...")
        val_df = load_labeled_data(VAL_FILE)

    print(f"  Train samples: {len(train_df)}")
    if val_df is not None:
        print(f"  Val samples:   {len(val_df)}")

    # ── Compute class weights ────────────────────────────────────────
    weights = compute_class_weights(train_df)
    class_weights = torch.tensor(weights).to(device)
    print(f"  Class weights computed (shape: {weights.shape})")

    # ── DataLoaders ──────────────────────────────────────────────────
    train_dataset = ABSADataset(train_df)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False
    )

    val_loader = None
    if val_df is not None:
        val_dataset = ABSADataset(val_df)
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False
        )

    # ── Model ────────────────────────────────────────────────────────
    if model is None:
        print("Initialising MARBERT model...")
        model = ABSAModel()
    model = model.to(device)

    # ── Optimizer & Scheduler ────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ── Training loop ────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_DIR, f"best_model{extra_tag}.pt")

    best_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        start = time.time()
        print(f"\nEpoch {epoch}/{num_epochs}")

        # Train
        avg_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, class_weights, device
        )

        elapsed = time.time() - start

        if val_loader is not None and use_early_stopping:
            # Validate
            preds, probs, labels, _ = predict_all(model, val_loader, device)
            val_f1 = compute_micro_f1(labels, preds)

            print(f"  Loss: {avg_loss:.4f}  |  Val F1: {val_f1:.4f}  |  "
                  f"Time: {elapsed:.0f}s")

            # Checkpoint
            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                torch.save(model.state_dict(), save_path)
                print(f"  ✓ New best model saved (F1={best_f1:.4f})")
            else:
                patience_counter += 1
                print(f"  ✗ No improvement ({patience_counter}/"
                      f"{EARLY_STOPPING_PATIENCE})")

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered after {epoch} epochs.")
                break
        else:
            # No validation set — save every epoch
            print(f"  Loss: {avg_loss:.4f}  |  Time: {elapsed:.0f}s")
            torch.save(model.state_dict(), save_path)
            best_f1 = avg_loss  # track loss for reporting

    # ── Load best model ──────────────────────────────────────────────
    if val_loader is not None and use_early_stopping:
        print(f"\nLoading best model (F1={best_f1:.4f})...")
        model.load_state_dict(torch.load(save_path, map_location=device))

        # Final validation
        preds, probs, labels, _ = predict_all(model, val_loader, device)
        final_f1 = compute_micro_f1(labels, preds)
    else:
        final_f1 = best_f1

    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Best Micro-F1:    {final_f1:.4f}")
    print(f"  Model saved to:   {save_path}")
    print(f"{'='*60}\n")

    # Save metadata alongside model
    meta = {"best_f1": float(final_f1) if isinstance(final_f1, (float, np.floating)) else 0.0}
    with open(os.path.join(MODEL_DIR, "training_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return model, final_f1


if __name__ == "__main__":
    train()

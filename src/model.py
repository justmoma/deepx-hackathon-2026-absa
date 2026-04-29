"""
model.py — MARBERT backbone + per-aspect softmax classification heads.

Architecture (v2 — per-aspect softmax):
    ┌────────────────────────┐
    │   MARBERT Encoder      │  (pre-trained, 768-dim hidden states)
    │   (12 layers, 12 heads)│
    └──────────┬─────────────┘
               │ [CLS] token representation (768-dim)
    ┌──────────▼─────────────┐
    │   Dropout (0.2)        │
    └──────────┬─────────────┘
               │
    ┌──────────▼─────────────────────────────────┐
    │  9 independent Linear heads (768 → 4 each) │
    │  food_head, service_head, price_head, ...   │
    └──────────┬─────────────────────────────────┘
               │
    ┌──────────▼─────────────┐
    │  Softmax per head      │  4 mutually exclusive classes per aspect
    │  [absent, pos, neg, neu]│
    └────────────────────────┘

Why softmax instead of sigmoid:
- A review cannot be BOTH positive and negative for the same aspect.
- Softmax forces the model to choose exactly one sentiment per aspect.
- The "not_mentioned" class is explicitly learned, eliminating
  threshold-based aspect detection.
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from src.config import MODEL_NAME, NUM_ASPECTS, NUM_SENTIMENT_CLASSES, DROPOUT


class ABSAModel(nn.Module):
    """
    Multi-head classifier built on top of a pre-trained MARBERT.

    Each aspect has its own classification head with 4 outputs:
    [not_mentioned, positive, negative, neutral].

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier (default: UBC-NLP/MARBERT)
    num_aspects : int
        Number of aspect heads (default: 9)
    num_classes : int
        Number of sentiment classes per aspect (default: 4)
    dropout : float
        Dropout rate applied to the [CLS] representation.
    """

    def __init__(self, model_name=MODEL_NAME, num_aspects=NUM_ASPECTS,
                 num_classes=NUM_SENTIMENT_CLASSES, dropout=DROPOUT):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size  # 768 for MARBERT
        self.num_aspects = num_aspects
        self.num_classes = num_classes

        self.dropout = nn.Dropout(dropout)

        # 9 separate classification heads — one per aspect
        self.aspect_heads = nn.ModuleList([
            nn.Linear(hidden_size, num_classes)
            for _ in range(num_aspects)
        ])

    def forward(self, input_ids, attention_mask, labels=None,
                class_weights=None):
        """
        Forward pass.

        Parameters
        ----------
        input_ids : tensor (batch, seq_len)
        attention_mask : tensor (batch, seq_len)
        labels : tensor (batch, 9) of int64, optional
            Per-aspect class indices: 0=absent, 1=pos, 2=neg, 3=neu
        class_weights : tensor (9, 4), optional
            Per-aspect class weights for CrossEntropyLoss

        Returns
        -------
        dict with:
            - "logits": list of 9 tensors, each (batch, 4)
            - "probs": list of 9 tensors, each (batch, 4) — softmax probs
            - "loss": scalar (only if labels provided)
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # [CLS] token is the first token's hidden state
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)

        # Run each aspect head
        all_logits = []
        all_probs = []
        for head in self.aspect_heads:
            logits = head(cls_output)          # (batch, 4)
            probs = torch.softmax(logits, dim=-1)  # (batch, 4)
            all_logits.append(logits)
            all_probs.append(probs)

        result = {"logits": all_logits, "probs": all_probs}

        if labels is not None:
            total_loss = 0.0
            for i, logits in enumerate(all_logits):
                aspect_labels = labels[:, i]  # (batch,) int64

                if class_weights is not None:
                    w = class_weights[i].to(logits.device)
                    loss_fn = nn.CrossEntropyLoss(weight=w)
                else:
                    loss_fn = nn.CrossEntropyLoss()

                total_loss += loss_fn(logits, aspect_labels)

            result["loss"] = total_loss / self.num_aspects

        return result

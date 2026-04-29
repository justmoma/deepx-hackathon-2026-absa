"""
dataset.py — PyTorch Dataset for Arabic ABSA reviews.

This wraps the preprocessed DataFrames into something PyTorch's
DataLoader can iterate over.  Each sample becomes a dictionary with:
  - input_ids:      tokenised review  (int tensor, shape [max_len])
  - attention_mask:  1 where real tokens, 0 where padding  (same shape)
  - labels:         9-dim integer vector  (long tensor) — per-aspect class
  - review_id:      original ID for tracing back predictions
"""

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from src.config import MODEL_NAME, MAX_SEQ_LENGTH


# We cache the tokenizer at module level so it's only loaded once,
# even if we create multiple Dataset objects.
_tokenizer = None


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return _tokenizer


class ABSADataset(Dataset):
    """
    Dataset for reviews that have labels (train / validation).
    Labels are 9 integers: one per aspect, value = sentiment class index.
    """

    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = get_tokenizer()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row["clean_text"]

        encoding = self.tokenizer(
            text,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(row["label_vec"], dtype=torch.long),
            "review_id": row["review_id"],
        }


class ABSAUnlabeledDataset(Dataset):
    """
    Dataset for reviews without labels (unlabeled / test).
    Same as above but no label_vec.
    """

    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = get_tokenizer()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row["clean_text"]

        encoding = self.tokenizer(
            text,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "review_id": row["review_id"],
        }

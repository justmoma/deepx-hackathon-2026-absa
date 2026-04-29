"""
preprocess.py — Arabic text cleaning, language filtering, and label encoding.

Arabic user reviews are messy: mixed dialects, elongated letters,
inconsistent spelling of hamza/alef, stray diacritics, emojis
sitting next to curse words, and so on.  This module normalises
the surface form so the model sees less noise, while keeping the
parts that actually carry sentiment (e.g. emojis, punctuation that
signals emphasis).

NEW in v2: aggressive non-Arabic filtering and per-aspect label encoding
(9 integers instead of 27 floats).
"""

import re
import json
import pandas as pd
import numpy as np
from src.config import (
    ASPECTS, ASPECT_TO_IDX, NUM_ASPECTS,
    SENTIMENT_TO_IDX, IDX_TO_SENTIMENT, IDX_TO_ASPECT,
    ARABIC_MIN_RATIO, MIN_ARABIC_WORDS,
)


# ─── Arabic normalisation maps ───────────────────────────────────────

# Alef variants → plain alef
_ALEF_MAP = str.maketrans({
    "\u0622": "\u0627",   # آ → ا
    "\u0623": "\u0627",   # أ → ا
    "\u0625": "\u0627",   # إ → ا
})

# Taa marbuta → haa  (common in Egyptian dialect writing)
_TAA_MARBUTA = str.maketrans({"\u0629": "\u0647"})  # ة → ه

# Alef maqsura → yaa
_ALEF_MAQSURA = str.maketrans({"\u0649": "\u064A"})  # ى → ي

# Arabic diacritics (tashkeel) — we remove them entirely
_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670]")

# Tatweel (kashida) — the decorative elongation character ـ
_TATWEEL = re.compile(r"\u0640")

# Repeated characters: "حلوووووو" → "حلوو" (keep max 2)
_REPEATED_CHAR = re.compile(r"(.)\1{2,}")

# URLs
_URL = re.compile(r"https?://\S+|www\.\S+")

# Mentions / hashtags
_MENTION = re.compile(r"@\w+")
_HASHTAG_SYMBOL = re.compile(r"#")

# Extra whitespace
_MULTI_SPACE = re.compile(r"\s+")

# Arabic Unicode block (basic + supplement)
_ARABIC_CHAR = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")
_ALPHA_CHAR = re.compile(r"[a-zA-Z\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")


def clean_arabic(text: str) -> str:
    """
    Light normalisation pipeline for Arabic review text.
    Keeps emojis and meaningful punctuation intact.
    """
    if not isinstance(text, str):
        return ""

    text = _URL.sub(" ", text)
    text = _MENTION.sub(" ", text)
    text = _HASHTAG_SYMBOL.sub("", text)

    # Normalise letter forms
    text = text.translate(_ALEF_MAP)
    text = text.translate(_TAA_MARBUTA)
    text = text.translate(_ALEF_MAQSURA)

    # Strip diacritics and tatweel
    text = _DIACRITICS.sub("", text)
    text = _TATWEEL.sub("", text)

    # Collapse repeated characters
    text = _REPEATED_CHAR.sub(r"\1\1", text)

    # Normalise whitespace
    text = _MULTI_SPACE.sub(" ", text).strip()

    return text


# ─── Arabic language filtering ───────────────────────────────────────

def is_arabic(text: str) -> bool:
    """
    Check if a text is predominantly Arabic.
    Returns True if the ratio of Arabic characters to total alphabetic
    characters exceeds ARABIC_MIN_RATIO and there are enough Arabic words.
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return False

    arabic_chars = len(_ARABIC_CHAR.findall(text))
    alpha_chars = len(_ALPHA_CHAR.findall(text))

    if alpha_chars == 0:
        # No alphabetic chars at all — could be all emoji/numbers.
        # Keep it if it has some content.
        return len(text.strip()) >= 5

    ratio = arabic_chars / alpha_chars

    # Count Arabic words (sequences of Arabic chars)
    arabic_words = len(re.findall(r"[\u0600-\u06FF]+", text))

    return ratio >= ARABIC_MIN_RATIO and arabic_words >= MIN_ARABIC_WORDS


def filter_non_arabic(df: pd.DataFrame, text_col: str = "clean_text") -> pd.DataFrame:
    """
    Remove rows that are not predominantly Arabic.
    Returns the filtered DataFrame + prints stats.
    """
    original_len = len(df)
    mask = df[text_col].apply(is_arabic)
    filtered_df = df[mask].reset_index(drop=True)
    dropped = original_len - len(filtered_df)
    print(f"  Arabic filter: kept {len(filtered_df)}/{original_len} "
          f"(dropped {dropped} non-Arabic rows)")
    return filtered_df


# ─── Per-aspect label encoding (9 integers) ──────────────────────────

def encode_labels(aspects_json: str, sentiments_json: str) -> np.ndarray:
    """
    Convert the JSON aspect/sentiment strings into a 9-dim integer vector.

    Each position corresponds to an aspect (food=0, service=1, etc.).
    Values: 0=not_mentioned, 1=positive, 2=negative, 3=neutral.

    Example:
        aspects_json:    '["food", "service"]'
        sentiments_json: '{"food": "positive", "service": "negative"}'
        → [1, 2, 0, 0, 0, 0, 0, 0, 0]
           food=positive, service=negative, rest=not_mentioned
    """
    vec = np.zeros(NUM_ASPECTS, dtype=np.int64)  # all 0 = not_mentioned
    aspects = json.loads(aspects_json)
    sentiments = json.loads(sentiments_json)

    for aspect in aspects:
        sentiment = sentiments.get(aspect)
        if sentiment and aspect in ASPECT_TO_IDX:
            vec[ASPECT_TO_IDX[aspect]] = SENTIMENT_TO_IDX.get(sentiment, 0)

    return vec


def decode_labels(aspect_probs: np.ndarray):
    """
    Convert model output (9 arrays of 4-class probabilities) back into
    aspects list and aspect_sentiments dict.

    Parameters
    ----------
    aspect_probs : list of np.ndarray
        List of 9 arrays, each of shape (4,), containing softmax probabilities
        for [not_mentioned, positive, negative, neutral].

    Returns
    -------
    aspects : list of str
    aspect_sentiments : dict of str -> str
    """
    aspects = []
    aspect_sentiments = {}

    for i, probs in enumerate(aspect_probs):
        pred_class = int(np.argmax(probs))
        if pred_class != 0:  # 0 = not_mentioned
            aspect_name = IDX_TO_ASPECT[i]
            sentiment_name = IDX_TO_SENTIMENT[pred_class]
            aspects.append(aspect_name)
            aspect_sentiments[aspect_name] = sentiment_name

    # Fallback: if nothing detected, pick the aspect with highest
    # non-absent probability
    if len(aspects) == 0:
        best_aspect = -1
        best_prob = 0.0
        for i, probs in enumerate(aspect_probs):
            # max prob among positive/negative/neutral (indices 1,2,3)
            max_sent_prob = float(np.max(probs[1:]))
            if max_sent_prob > best_prob:
                best_prob = max_sent_prob
                best_aspect = i
        if best_aspect >= 0:
            pred_class = int(np.argmax(aspect_probs[best_aspect][1:])) + 1
            aspect_name = IDX_TO_ASPECT[best_aspect]
            sentiment_name = IDX_TO_SENTIMENT[pred_class]
            aspects.append(aspect_name)
            aspect_sentiments[aspect_name] = sentiment_name

    return aspects, aspect_sentiments


def load_labeled_data(filepath: str, apply_arabic_filter: bool = True) -> pd.DataFrame:
    """Load an xlsx file with labels, clean text, encode labels."""
    df = pd.read_excel(filepath)
    df["clean_text"] = df["review_text"].apply(clean_arabic)

    if apply_arabic_filter:
        df = filter_non_arabic(df)

    df["label_vec"] = df.apply(
        lambda row: encode_labels(row["aspects"], row["aspect_sentiments"]),
        axis=1
    )
    return df


def load_unlabeled_data(filepath: str, apply_arabic_filter: bool = True) -> pd.DataFrame:
    """Load the unlabeled xlsx, clean text only."""
    df = pd.read_excel(filepath)
    df["clean_text"] = df["review_text"].apply(clean_arabic)

    if apply_arabic_filter:
        df = filter_non_arabic(df)

    return df


def compute_class_weights(df: pd.DataFrame) -> np.ndarray:
    """
    Compute per-aspect class weights for CrossEntropyLoss.

    Returns a (9, 4) array where weights[i] are the 4-class weights
    for aspect i. Uses inverse frequency.
    """
    label_matrix = np.stack(df["label_vec"].values)  # (N, 9)
    weights = np.ones((NUM_ASPECTS, 4), dtype=np.float32)

    for i in range(NUM_ASPECTS):
        classes = label_matrix[:, i]
        for c in range(4):
            count = int((classes == c).sum())
            if count > 0:
                weights[i, c] = len(df) / (4.0 * count)
            else:
                weights[i, c] = 1.0

    # Cap weights to avoid instability
    weights = np.clip(weights, 0.5, 10.0)

    return weights

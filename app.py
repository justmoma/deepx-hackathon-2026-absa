"""
app.py — Flask backend for the Arabic ABSA Demo.

Serves a premium web interface for Arabic Aspect-Based Sentiment Analysis.
Works in two modes:
  - DEMO mode: Uses keyword-based analysis when no trained model is available.
  - LIVE mode: Uses the trained MARBERTv2 model for real inference.

Run:
    python app.py
"""

import json
import os
import re
import random

from flask import Flask, render_template, request, jsonify

# ─── App Setup ───────────────────────────────────────────────────────
app = Flask(__name__)

# ─── Constants ───────────────────────────────────────────────────────
ASPECTS = [
    "food", "service", "price", "cleanliness",
    "delivery", "ambiance", "app_experience", "general", "none"
]
SENTIMENTS = ["positive", "negative", "neutral"]

ASPECT_ICONS = {
    "food": "🍽️",
    "service": "🤝",
    "price": "💰",
    "cleanliness": "✨",
    "delivery": "🚚",
    "ambiance": "🎶",
    "app_experience": "📱",
    "general": "📋",
    "none": "—"
}

ASPECT_LABELS_AR = {
    "food": "الطعام",
    "service": "الخدمة",
    "price": "السعر",
    "cleanliness": "النظافة",
    "delivery": "التوصيل",
    "ambiance": "الأجواء",
    "app_experience": "التطبيق",
    "general": "عام",
    "none": "لا يوجد"
}

# ─── Model Loading ───────────────────────────────────────────────────
MODEL_LOADED = False
model = None
tokenizer = None

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_model")
WEIGHTS_PATH = os.path.join(MODEL_DIR, "best_model.pt")

try:
    if os.path.exists(WEIGHTS_PATH):
        import torch
        from transformers import AutoTokenizer
        from src.config import DEVICE, NUM_ASPECTS
        from src.model import ABSAModel
        from src.preprocess import clean_arabic, decode_labels

        print("[*] Loading trained model...")
        tokenizer = AutoTokenizer.from_pretrained("UBC-NLP/MARBERT")
        model = ABSAModel()
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        MODEL_LOADED = True
        print("[OK] Model loaded - LIVE mode active.")
    else:
        print("[!] No model weights found - running in DEMO mode.")
except Exception as e:
    print(f"[WARN] Could not load model: {e}")
    print("[!] Falling back to DEMO mode.")

# ─── Demo-Mode Keyword Analyzer ─────────────────────────────────────
# Maps Arabic keywords → (aspect, sentiment)
KEYWORD_MAP = {
    # Food — positive
    "لذيذ": ("food", "positive"), "ممتاز": ("food", "positive"),
    "طازج": ("food", "positive"), "رائع": ("food", "positive"),
    "شهي": ("food", "positive"), "اكل حلو": ("food", "positive"),
    "الأكل ممتاز": ("food", "positive"), "طعم جميل": ("food", "positive"),
    # Food — negative
    "بارد": ("food", "negative"), "ناشف": ("food", "negative"),
    "سيء": ("food", "negative"), "مقرف": ("food", "negative"),
    "الاكل سيء": ("food", "negative"), "الطعام سيء": ("food", "negative"),
    # Food — neutral
    "الطعام": ("food", "neutral"), "الاكل": ("food", "neutral"),
    "أكل": ("food", "neutral"), "طعام": ("food", "neutral"),
    "وجبة": ("food", "neutral"), "اكل": ("food", "neutral"),
    # Service — positive
    "الموظفين ممتازين": ("service", "positive"),
    "خدمة ممتازة": ("service", "positive"),
    "تعامل حلو": ("service", "positive"),
    "استقبال رائع": ("service", "positive"),
    # Service — negative
    "خدمة سيئة": ("service", "negative"),
    "موظفين سيئين": ("service", "negative"),
    "تعامل سيء": ("service", "negative"),
    "بطيء": ("service", "negative"),
    "تأخر": ("service", "negative"),
    # Service — neutral
    "خدمة": ("service", "neutral"), "موظف": ("service", "neutral"),
    "ويتر": ("service", "neutral"), "كاشير": ("service", "neutral"),
    # Price — positive
    "رخيص": ("price", "positive"), "سعر مناسب": ("price", "positive"),
    "اسعار معقولة": ("price", "positive"),
    # Price — negative
    "غالي": ("price", "negative"), "مبالغ": ("price", "negative"),
    "اسعار عالية": ("price", "negative"), "سعر عالي": ("price", "negative"),
    # Price — neutral
    "سعر": ("price", "neutral"), "اسعار": ("price", "neutral"),
    # Delivery
    "توصيل سريع": ("delivery", "positive"),
    "توصيل": ("delivery", "neutral"),
    "توصيل بطيء": ("delivery", "negative"),
    "اتأخر": ("delivery", "negative"), "تأخير": ("delivery", "negative"),
    "متأخر": ("delivery", "negative"),
    # Cleanliness
    "نظيف": ("cleanliness", "positive"), "نظافة": ("cleanliness", "positive"),
    "النظافة تحتاج تحسين": ("cleanliness", "negative"),
    "وسخ": ("cleanliness", "negative"), "قذر": ("cleanliness", "negative"),
    "مو نظيف": ("cleanliness", "negative"),
    # Ambiance
    "جو حلو": ("ambiance", "positive"), "اجواء جميلة": ("ambiance", "positive"),
    "ديكور": ("ambiance", "neutral"), "اجواء": ("ambiance", "neutral"),
    "جو": ("ambiance", "neutral"),
    "مزعج": ("ambiance", "negative"), "ضوضاء": ("ambiance", "negative"),
    # App experience
    "تطبيق": ("app_experience", "neutral"),
    "تطبيق سهل": ("app_experience", "positive"),
    "تطبيق معلق": ("app_experience", "negative"),
    # General positive/negative words (broad)
    "ممتاز": ("general", "positive"), "رائع": ("general", "positive"),
    "جميل": ("general", "positive"), "حلو": ("general", "positive"),
    "سيء": ("general", "negative"), "زفت": ("general", "negative"),
    "مو حلو": ("general", "negative"), "تحتاج تحسين": ("general", "negative"),
}


def demo_analyze(text: str) -> dict:
    """Keyword-based aspect-sentiment analyzer for demo mode."""
    detected = {}
    text_lower = text.strip()

    # Longer phrases first for better matching
    sorted_keywords = sorted(KEYWORD_MAP.keys(), key=len, reverse=True)
    for keyword in sorted_keywords:
        if keyword in text_lower:
            aspect, sentiment = KEYWORD_MAP[keyword]
            if aspect not in detected:
                detected[aspect] = sentiment

    # Fallback: if nothing detected, mark as general with neutral
    if not detected:
        detected["general"] = "neutral"

    aspects = list(detected.keys())
    aspect_sentiments = detected

    return {
        "aspects": aspects,
        "aspect_sentiments": aspect_sentiments,
    }


def live_analyze(text: str) -> dict:
    """Run the trained model for real inference."""
    import torch
    import numpy as np
    from src.preprocess import clean_arabic, decode_labels
    from src.config import DEVICE, NUM_ASPECTS, IDX_TO_ASPECT

    cleaned = clean_arabic(text)
    inputs = tokenizer(
        cleaned, padding="max_length", truncation=True,
        max_length=128, return_tensors="pt"
    )
    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

    with torch.no_grad():
        output = model(input_ids, attention_mask)
        probs_list = output["probs"]

    aspect_probs = []
    for i in range(NUM_ASPECTS):
        aspect_probs.append(probs_list[i][0].cpu().numpy())

    aspects, sentiments = decode_labels(aspect_probs)

    # Build confidence scores (max sentiment probability for detected aspects)
    confidences = {}
    for i in range(NUM_ASPECTS):
        aspect_name = IDX_TO_ASPECT[i]
        if aspect_name in sentiments:
            p = aspect_probs[i]
            pred_class = int(np.argmax(p))
            confidences[aspect_name] = round(float(p[pred_class]) * 100, 1)

    return {
        "aspects": aspects,
        "aspect_sentiments": sentiments,
        "confidences": confidences,
    }


# ─── Routes ──────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main UI page."""
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """Analyze a single Arabic review."""
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    if MODEL_LOADED:
        result = live_analyze(text)
        mode = "live"
    else:
        result = demo_analyze(text)
        mode = "demo"

    return jsonify({
        "text": text,
        "mode": mode,
        "aspects": result["aspects"],
        "aspect_sentiments": result["aspect_sentiments"],
        "confidences": result.get("confidences", {}),
        "aspect_icons": {a: ASPECT_ICONS.get(a, "📋") for a in result["aspects"]},
        "aspect_labels_ar": {a: ASPECT_LABELS_AR.get(a, a) for a in result["aspects"]},
    })


@app.route("/api/batch", methods=["POST"])
def batch_analyze():
    """Analyze multiple reviews at once."""
    data = request.get_json()
    reviews = data.get("reviews", [])

    if not reviews:
        return jsonify({"error": "No reviews provided"}), 400

    results = []
    for text in reviews:
        text = text.strip()
        if not text:
            continue
        if MODEL_LOADED:
            result = live_analyze(text)
            mode = "live"
        else:
            result = demo_analyze(text)
            mode = "demo"
        results.append({
            "text": text,
            "mode": mode,
            "aspects": result["aspects"],
            "aspect_sentiments": result["aspect_sentiments"],
        })

    return jsonify({"results": results, "count": len(results)})


@app.route("/api/health", methods=["GET"])
def health():
    """Return system status."""
    return jsonify({
        "status": "ok",
        "mode": "live" if MODEL_LOADED else "demo",
        "model": "UBC-NLP/MARBERT" if MODEL_LOADED else "keyword-based",
        "aspects": ASPECTS,
        "metrics": {
            "f1": 0.828,
            "precision": 0.813,
            "recall": 0.844,
            "pillar1_score": "24.85/30"
        }
    })


@app.route("/api/stats", methods=["GET"])
def stats():
    """Return prediction statistics from predictions.json."""
    predictions_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "predictions.json"
    )
    if not os.path.exists(predictions_path):
        return jsonify({"error": "No predictions file found"}), 404

    with open(predictions_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    # Compute stats
    aspect_counts = {}
    sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
    aspect_sentiment_breakdown = {}

    for pred in predictions:
        for aspect in pred["aspects"]:
            aspect_counts[aspect] = aspect_counts.get(aspect, 0) + 1
            if aspect not in aspect_sentiment_breakdown:
                aspect_sentiment_breakdown[aspect] = {"positive": 0, "negative": 0, "neutral": 0}
        for aspect, sentiment in pred["aspect_sentiments"].items():
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
            if aspect in aspect_sentiment_breakdown:
                aspect_sentiment_breakdown[aspect][sentiment] += 1

    return jsonify({
        "total_predictions": len(predictions),
        "aspect_counts": aspect_counts,
        "sentiment_counts": sentiment_counts,
        "aspect_sentiment_breakdown": aspect_sentiment_breakdown,
    })


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Arabic ABSA Demo")
    print(f"  Mode: {'LIVE (MARBERTv2)' if MODEL_LOADED else 'DEMO (keyword-based)'}")
    print("=" * 60)
    print("\n  Open http://localhost:5000 in your browser\n")
    app.run(debug=False, host="0.0.0.0", port=5000)

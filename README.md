<p align="center">
  <h1 align="center">✦ Arabic Aspect-Based Sentiment Analysis</h1>
  <p align="center">
    <strong>DeepX Hackathon 2026 — Team Asterisk</strong><br>
    🏆 <strong>17th Place</strong> (out of 151 teams)
  </p>
  <p align="center">
    <a href="#-quick-start"><img src="https://img.shields.io/badge/python-3.10+-blue?logo=python&logoColor=white" alt="Python"></a>
    <a href="#-results"><img src="https://img.shields.io/badge/F1_Score-82.8%25-brightgreen" alt="F1 Score"></a>
    <a href="#-architecture"><img src="https://img.shields.io/badge/backbone-MARBERTv2-orange?logo=huggingface" alt="MARBERT"></a>
    <a href="#-web-demo"><img src="https://img.shields.io/badge/demo-Flask_Web_UI-purple?logo=flask" alt="Flask Demo"></a>
    <a href="https://deepx-hackathon-2026-absa-dwx2.vercel.app"><img src="https://img.shields.io/badge/Live_Deployment-Vercel-black?logo=vercel" alt="Vercel Deployment"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
  </p>
</p>

---

A production-grade NLP system that dissects **Arabic customer reviews** — identifying **what** aspects are mentioned and **how** the customer feels about each one. Built on [MARBERTv2](https://huggingface.co/UBC-NLP/MARBERT) with semi-supervised pseudo-labeling and a premium web interface.

<p align="center">
  <img src="https://img.shields.io/badge/%F0%9F%8D%BD%EF%B8%8F_food-%E2%9C%93-green" alt="food">
  <img src="https://img.shields.io/badge/%F0%9F%A4%9D_service-%E2%9C%93-green" alt="service">
  <img src="https://img.shields.io/badge/%F0%9F%92%B0_price-%E2%9C%93-green" alt="price">
  <img src="https://img.shields.io/badge/%E2%9C%A8_cleanliness-%E2%9C%93-green" alt="cleanliness">
  <img src="https://img.shields.io/badge/%F0%9F%9A%9A_delivery-%E2%9C%93-green" alt="delivery">
  <img src="https://img.shields.io/badge/%F0%9F%8E%B6_ambiance-%E2%9C%93-green" alt="ambiance">
  <img src="https://img.shields.io/badge/%F0%9F%93%B1_app-%E2%9C%93-green" alt="app">
  <img src="https://img.shields.io/badge/%F0%9F%93%8B_general-%E2%9C%93-green" alt="general">
  <img src="https://img.shields.io/badge/%E2%80%94_none-%E2%9C%93-gray" alt="none">
</p>

---

## 📸 Screenshots

<p align="center">
  <img src="screenshots/hero-landing.jpeg" alt="Hero Landing Page" width="100%">
</p>
<p align="center"><em>Hero landing page with key metrics and project overview</em></p>

<p align="center">
  <img src="screenshots/live-demo.jpeg" alt="Live Demo" width="100%">
</p>
<p align="center"><em>Live demo — real-time Arabic review analysis with confidence scores</em></p>

<p align="center">
  <img src="screenshots/model-architecture.jpeg" alt="Model Architecture" width="100%">
</p>
<p align="center"><em>Model architecture visualization — 4-stage pipeline overview</em></p>

<p align="center">
  <img src="screenshots/model-metrics.jpeg" alt="Model Metrics" width="100%">
</p>
<p align="center"><em>Performance metrics and 3-stage training progression</em></p>

<p align="center">
  <img src="screenshots/prediction-insights.jpeg" alt="Prediction Insights" width="100%">
</p>
<p align="center"><em>Data insights — aspect and sentiment distribution across 500 test predictions</em></p>

<p align="center">
  <img src="screenshots/aspect-taxonomy.jpeg" alt="Aspect Taxonomy" width="100%">
</p>
<p align="center"><em>9 aspect categories with Arabic labels</em></p>

---

## 📊 Results

| Metric | Score |
|:---|:---|
| **Micro F1** | **82.8%** |
| Precision | 81.3% |
| Recall | 84.4% |
| Pillar 1 Score | 24.85 / 30 |

---

## 🏗️ Architecture

```
              ┌────────────────────────┐
              │   MARBERT Encoder      │  Pre-trained on 1B+ dialectal Arabic tweets
              │   (12 layers, 768-dim) │
              └──────────┬─────────────┘
                         │ [CLS] token representation
              ┌──────────▼─────────────┐
              │   Dropout (0.2)        │
              └──────────┬─────────────┘
                         │
              ┌──────────▼─────────────────────────────────┐
              │  9 independent Linear heads (768 → 4 each) │
              │  food, service, price, cleanliness, ...     │
              └──────────┬─────────────────────────────────┘
                         │
              ┌──────────▼─────────────┐
              │  Softmax per head      │  [absent, positive, negative, neutral]
              └────────────────────────┘
```

### Key Design Decisions

- **Per-aspect softmax** instead of sigmoid — a review cannot be BOTH positive and negative for the same aspect
- **Arabic-only filtering** — reviews with <50% Arabic characters or <3 Arabic words are dropped (removed ~12% training data, ~58% unlabeled), improving Recall from 77.6% → 84.4%
- **Semi-supervised pseudo-labeling** — 3-stage pipeline that expands training data from 1,731 → 3,584 samples
- **Inverse-frequency class weights** in CrossEntropyLoss to handle severe class imbalance

### Training Pipeline

```
Stage 1: Supervised         1,731 labeled samples → Val F1 = 78.19%
                                    │
Stage 2: Pseudo-Label       2,972 unlabeled → 1,418 high-confidence (≥90%)
                                    │
Stage 3: Full Retrain       3,584 combined samples → Test F1 = 82.8%
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA GPU recommended (trained on Kaggle T4)

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/deepx-hackathon-2026-absa.git
cd deepx-hackathon-2026-absa

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Model Weights

The trained model weights (`best_model.pt`, ~621 MB) are not included in this repository due to size limits. To obtain them:

1. **Train from scratch** — see [Training](#-training) below
2. **Download** — *(if hosted)* place the file at `saved_model/best_model.pt`

> Without model weights, the web demo runs in **Demo Mode** using keyword-based analysis.

### Dataset

Place the competition datasets in the project root:

| File | Description |
|:---|:---|
| `DeepX_train.xlsx` | Labeled training data |
| `DeepX_validation.xlsx` | Labeled validation data |
| `DeepX_unlabeled.xlsx` | Unlabeled data for pseudo-labeling |
| `DeepX_hidden_test.xlsx` | Hidden test set |

> Datasets are not included as they are part of the DeepX Hackathon 2026 competition.

---

## 🎮 Web Demo

A premium, production-grade web interface for live sentiment analysis is deployed and available at:  
👉 **[deepx-hackathon-2026-absa-dwx2.vercel.app](https://deepx-hackathon-2026-absa-dwx2.vercel.app)**

Or run it locally:
```bash
python app.py
# Open http://localhost:5000
```

**Features:**
- 🔍 Real-time Arabic review analysis
- 📊 Interactive prediction statistics & visualizations
- 🎯 Confidence scores per aspect (in Live mode)
- 📱 Fully responsive dark-mode UI
- ⚡ Example reviews for quick testing
- 📋 JSON output toggle for API inspection

The demo works in two modes:
- **🟢 Live Mode** — uses the trained MARBERTv2 model for real inference
- **🟡 Demo Mode** — uses keyword-based analysis when no model weights are available

### ☁️ Serverless Deployment
This repository is configured for immediate deployment on platforms like **Vercel** via the included `vercel.json`. 

*Note: Due to strict bundle size limits on serverless functions, the `requirements.txt` is purposefully configured to install the lightweight CPU-only version of PyTorch. When deployed on Vercel without the model weights, the server intelligently falls back to the Demo Mode.*

---

## 🏋️ Training

### Stage 1 — Supervised Training

```bash
python -m src.train
```

Trains on Arabic-filtered labeled data with early stopping on validation F1.

### Stage 2 + 3 — Pseudo-Labeling & Retrain

```bash
python -m src.pseudo_label
```

Generates high-confidence pseudo-labels from unlabeled data, merges with training + validation data, and retrains from scratch.

### Generate Predictions

```bash
python -m src.predict
```

Outputs `predictions.json` in the exact competition submission format.

### Evaluate

```bash
python -m src.evaluate
```

Computes Micro-F1 and per-aspect breakdowns on the validation set.

---

## 📁 Project Structure

```
deepx-hackathon-2026-absa/
│
├── app.py                        # Flask web server & API endpoints
├── requirements.txt              # Python dependencies
├── predictions.json              # Final submission (500 test predictions)
├── sample_submission.json        # Competition format example
│
├── src/                          # Core ML pipeline
│   ├── __init__.py
│   ├── config.py                 # All hyperparameters & paths
│   ├── preprocess.py             # Arabic cleaning, filtering, label encoding
│   ├── dataset.py                # PyTorch Dataset classes
│   ├── model.py                  # MARBERT + 9 classification heads
│   ├── train.py                  # Training loop with early stopping
│   ├── evaluate.py               # F1 metrics & per-aspect breakdown
│   ├── predict.py                # Inference → JSON submission
│   └── pseudo_label.py           # Semi-supervised pipeline
│
├── templates/
│   └── index.html                # Web UI template (RTL Arabic support)
│
├── static/
│   ├── css/style.css             # Premium dark-mode stylesheet
│   └── js/app.js                 # Frontend logic & visualizations
│
├── notebooks/
│   ├── deepx.ipynb               # Kaggle training notebook
│   └── training_meta.json        # Training run metadata
│
├── saved_model/                  # Model weights directory
│   └── .gitkeep                  # (weights excluded — see README)
│
├── screenshots/                  # Web UI screenshots
│   ├── hero-landing.jpeg
│   ├── live-demo.jpeg
│   ├── model-architecture.jpeg
│   ├── model-metrics.jpeg
│   ├── prediction-insights.jpeg
│   └── aspect-taxonomy.jpeg
│
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🔧 API Reference

The Flask server exposes the following endpoints:

| Method | Endpoint | Description |
|:---|:---|:---|
| `GET` | `/` | Serve the web UI |
| `POST` | `/api/analyze` | Analyze a single review |
| `POST` | `/api/batch` | Analyze multiple reviews |
| `GET` | `/api/health` | System status & mode |
| `GET` | `/api/stats` | Prediction statistics |

### Example — Analyze a Review

```bash
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "الأكل كان ممتاز بس الخدمة بطيئة"}'
```

**Response:**
```json
{
  "text": "الأكل كان ممتاز بس الخدمة بطيئة",
  "mode": "live",
  "aspects": ["food", "service"],
  "aspect_sentiments": {
    "food": "positive",
    "service": "negative"
  },
  "confidences": {
    "food": 97.3,
    "service": 92.1
  }
}
```

---

## 🏷️ Aspect Taxonomy

| Aspect | Arabic | Icon | Examples |
|:---|:---|:---|:---|
| `food` | الطعام | 🍽️ | Quality, taste, freshness, portions |
| `service` | الخدمة | 🤝 | Staff attitude, speed, professionalism |
| `price` | السعر | 💰 | Value for money, pricing fairness |
| `cleanliness` | النظافة | ✨ | Hygiene, tidiness |
| `delivery` | التوصيل | 🚚 | Speed, packaging, accuracy |
| `ambiance` | الأجواء | 🎶 | Atmosphere, decor, noise |
| `app_experience` | التطبيق | 📱 | App usability, bugs |
| `general` | عام | 📋 | Overall impression |
| `none` | لا يوجد | — | No specific aspect |

Each aspect is classified as one of: **positive** · **negative** · **neutral** · **not_mentioned**

---

## 🛠️ Tech Stack

| Component | Technology |
|:---|:---|
| Language Model | [UBC-NLP/MARBERT](https://huggingface.co/UBC-NLP/MARBERT) |
| Deep Learning | PyTorch 2.10 |
| NLP Framework | Hugging Face Transformers 5.0 |
| Web Backend | Flask 3.1 |
| Data Processing | pandas, NumPy, scikit-learn |
| Training Environment | Kaggle (T4 GPU) |

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <strong>Built with ❤️ for the DeepX Hackathon 2026</strong><br>
  <sub>Arabic Aspect-Based Sentiment Analysis Challenge</sub>
</p>

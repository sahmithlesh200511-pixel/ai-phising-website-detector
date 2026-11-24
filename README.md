# AI Phishing Website Detector

A machine learning system that predicts whether a URL is **legitimate** or **phishing**, purely from the structure of the URL itself — no external threat-database lookups, no page fetching, fully offline.

Built as a minor project for **Engineering Project Work – I (CO443)**, Central Department of Computer Engineering, Mid-West University.

---

## Overview

- **Model:** Random Forest classifier (scikit-learn), 200 estimators
- **Training data:** [PhiUSIIL Phishing URL Dataset](https://archive.ics.uci.edu/dataset/967/phiusil-phishing-url-dataset) (UCI), augmented with real domains from the [Tranco](https://tranco-list.eu/) top-sites ranking and synthetic realistic URL patterns
- **Accuracy:** 98.93% on held-out test data
- **Features:** 23 lexical features extracted from the URL text (length, HTTPS usage, subdomain depth, entropy, and more)
- **Interface:** Flask backend + browser-based frontend, returning a trust score (0–100) and an explainable signal breakdown

## Project Structure

```
AI-Phising-Website-detector/
├── backend/
│   └── app.py                  # Flask server, /predict API endpoint
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── ml/
│   ├── feature_extraction.py   # Shared feature-extraction function
│   ├── train.py                 # Trains and saves the model
│   ├── predict.py               # Loads model, makes predictions
│   ├── augment_dataset.py       # Format-variant augmentation
│   ├── merge_tranco.py          # Merges Tranco domains into training data
│   ├── augment_slugs.py         # Hyphenated slug-path augmentation
│   ├── augment_ids.py           # Numeric-ID path augmentation
│   ├── dataset/                 # CSV datasets (not committed — see .gitignore)
│   └── model/                   # Trained model .pkl (not committed)
├── test/                        # Unit tests
├── requirements.txt
└── README.md
```

## Setup

**1. Clone the repository and create a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add the dataset**

Download the [PhiUSIIL Phishing URL Dataset](https://archive.ics.uci.edu/dataset/967/phiusil-phishing-url-dataset) and place it at `ml/dataset/phishing_dataset.csv` (dataset files are excluded from version control due to size — see `.gitignore`).

**4. Train the model**
```bash
python ml/train.py
```
This produces `ml/model/phishing_model.pkl`.

**5. Run the web app**
```bash
python backend/app.py
```
Open **http://127.0.0.1:5000** in your browser.

## Running Tests

```bash
python -m pytest test/
```

## How It Works

1. A URL is submitted through the web interface.
2. The backend extracts 23 structural features from the URL text (`ml/feature_extraction.py`).
3. The trained Random Forest model predicts Legitimate/Phishing with a confidence score.
4. The frontend displays a trust score and a breakdown of heuristic signals (HTTPS, TLD risk, shortener use, subdomain depth, and more).

## Known Limitations

- Analyzes URL structure only — no live threat-database checks, domain-age lookups, or page-content analysis.
- Can be less confident on structurally ambiguous URLs (e.g. a personal blog on a free hosting platform can resemble a phishing site hosted the same way).
- A "Legitimate" verdict is one signal, not a safety guarantee.

## Team

| Name | Roll No. |
|---|---|
| Abhishek Kumar Sah | 1 |
| Alin Singh | 2 |
| Mithlesh Sah | 12 |
| Rhytham B.C. | 15 |

Central Department of Computer Engineering, Mid-West University, Surkhet, Nepal.
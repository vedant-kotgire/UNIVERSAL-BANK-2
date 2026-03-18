# 🏦 Universal Bank — Personal Loan Campaign Intelligence

A comprehensive Streamlit dashboard for predicting personal loan acceptance using classification algorithms (Decision Tree, Random Forest, Gradient Boosted Tree).

## Features

- **📊 Executive Summary** — Descriptive analytics with KPIs and distribution charts
- **🔍 Customer Deep Dive** — Diagnostic analytics with correlation heatmaps, box plots, and segmentation
- **🤖 Model Performance** — Side-by-side comparison table, ROC curves, confusion matrices, and feature importance
- **🎯 Campaign Strategy** — Prescriptive analytics with customer scoring, cumulative gain curves, and actionable recommendations
- **📤 Predict New Data** — Upload new customer CSV and download predictions with probability scores

## Quick Start

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/universal-bank-ml.git
cd universal-bank-ml

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set `app.py` as the main file
5. Deploy

## Dataset

- **Source:** `data/UniversalBank.csv` — 5,000 customer records
- **Target:** `Personal Loan` (0 = Rejected, 1 = Accepted)
- **Test File:** `data/Test_UniversalBank.csv` — 50 sample records for upload testing

## Tech Stack

- Python 3.10+
- Streamlit
- Scikit-learn / XGBoost
- Plotly / Seaborn / Matplotlib
- Pandas / NumPy

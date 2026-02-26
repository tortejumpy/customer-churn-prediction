# Customer Churn Prediction

A production-grade machine learning pipeline for predicting customer churn, built on the IBM Telco Customer Churn dataset. The project combines a rigorous Python ML backend with a fully interactive frontend dashboard for result visualization.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [ML Pipeline](#ml-pipeline)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Frontend Dashboard](#frontend-dashboard)
- [Results](#results)

---

## Overview

Customer churn—when a customer stops doing business with a company—is one of the most costly challenges in subscription-based industries. This project builds an end-to-end pipeline that:

- Ingests and cleans raw telecom customer data
- Engineers domain-relevant features
- Benchmarks 8 classifier families with 5-fold cross-validation
- Tunes the top-N models via `RandomizedSearchCV`
- Selects the best model by test ROC-AUC
- Handles class imbalance with SMOTE oversampling
- Saves the trained model and preprocessor for inference
- Serves results through a static HTML/CSS/JS dashboard

---

## Project Structure

```
customer-churn-prediction/
├── src/
│   ├── config.py               # Centralized paths and hyperparameters
│   ├── data_preprocessing.py   # Cleaning, encoding, SMOTE, train/test split
│   ├── feature_engineering.py  # Domain feature construction
│   ├── model_training.py       # Baseline experiments + hyperparameter tuning
│   ├── model_evaluation.py     # Metrics, threshold tuning, and plot generation
│   ├── predict.py              # Inference on new customer records
│   └── utils.py                # Logging, seeding, directory helpers
├── frontend/
│   ├── index.html              # Dashboard HTML
│   ├── css/style.css           # Dashboard styles
│   └── js/                     # Dashboard JavaScript
├── data/                       # Processed CSVs (generated at runtime)
├── models/                     # Saved model artifacts (generated at runtime)
│   └── plots/                  # Evaluation plots (generated at runtime)
├── Customer Churn Prediction.ipynb   # Exploratory analysis notebook
├── train.py                    # Main pipeline entry point
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Layer | Library / Tool |
|---|---|
| Data processing | pandas, numpy |
| ML models | scikit-learn, XGBoost, LightGBM |
| Class imbalance | imbalanced-learn (SMOTE) |
| Hyperparameter search | RandomizedSearchCV (scikit-learn) |
| Visualization | matplotlib, seaborn |
| Frontend | HTML, CSS, JavaScript (vanilla) |
| Notebook | Jupyter |

---

## ML Pipeline

### 1. Data Cleaning

- Drops `customerID` (non-predictive identifier)
- Converts `TotalCharges` to numeric; imputes missing values as `tenure × MonthlyCharges`
- Normalizes `SeniorCitizen` from integer flags to `Yes/No` strings
- Encodes the binary target `Churn` as 0/1

### 2. Feature Engineering

Engineered features added on top of the raw columns:

| Feature | Description |
|---|---|
| `tenure_group` | Binned tenure into loyalty tiers |
| `charges_per_month_ratio` | `TotalCharges / MonthlyCharges` ratio |
| `num_services` | Count of active add-on services per customer |
| `has_online_security_and_backup` | Binary flag for both security services |
| `monthly_charges_group` | Binned monthly charges into spending tiers |

### 3. Preprocessing

- **Numerical features** (`tenure`, `MonthlyCharges`, `TotalCharges`): `StandardScaler`
- **Categorical features** (16 columns): `OneHotEncoder` with `drop="first"`
- 80/20 stratified train/test split
- SMOTE applied to the training set to balance the target class

### 4. Model Experiments

Eight classifiers are benchmarked in parallel using 5-fold stratified cross-validation scored by ROC-AUC:

- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- Extra Trees
- AdaBoost
- MLP (Neural Network)

### 5. Hyperparameter Tuning

The top-N baseline models (default: 3) are tuned using `RandomizedSearchCV` with 50 iterations. The model with the best test ROC-AUC is saved to `models/best_model.pkl`.

### 6. Evaluation

- Optimal classification threshold via Youden's J statistic on the ROC curve
- Metrics: ROC-AUC, Precision, Recall, F1-Score
- Plots saved to `models/plots/`: confusion matrix, ROC curve, PR curve, feature importance, model comparison

---

## Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/tortejumpy/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt
```

### Data

Place the raw CSV file in the project root. The expected default filename is:

```
Customer_data - customer_data.csv
```

This path is configurable in `src/config.py` via the `RAW_DATA_FILE` constant.

---

## Usage

### Run the full training pipeline

```bash
python train.py
```

### Options

```
--top-n INT     Number of top baseline models to tune (default: 3)
--data PATH     Path to the raw CSV data file
--no-plots      Skip generating evaluation plots
```

**Examples:**

```bash
# Tune top 5 models
python train.py --top-n 5

# Use a custom data file
python train.py --data path/to/data.csv

# Run without generating plots
python train.py --no-plots
```

### Predict on a new customer record

```python
from src.predict import predict_churn

record = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.5,
    "TotalCharges": 1026.0
}

result = predict_churn(record)
print(result)
# {"churn_probability": 0.81, "churn_prediction": 1}
```

### Outputs

| Output | Location |
|---|---|
| Best model (pickle) | `models/best_model.pkl` |
| Preprocessor (pickle) | `models/preprocessor.pkl` |
| Model metadata (JSON) | `models/model_metadata.json` |
| Confusion matrix | `models/plots/confusion_matrix.png` |
| ROC curve | `models/plots/roc_curve.png` |
| PR curve | `models/plots/pr_curve.png` |
| Feature importance | `models/plots/feature_importance.png` |
| Model comparison | `models/plots/model_comparison.png` |
| Pipeline log | `pipeline.log` |

---

## Frontend Dashboard

The `frontend/` directory contains a self-contained, static HTML dashboard that visualizes the model results and key business insights. Open it directly in any browser — no server required.

```bash
# Open the dashboard
start frontend/index.html   # Windows
open frontend/index.html    # macOS
```

The dashboard includes:

- Churn rate overview and KPI cards
- Interactive model performance comparison charts
- Feature importance visualization
- Customer segmentation insights

---

## Results

The pipeline consistently achieves strong performance on the Telco Churn dataset:

| Metric | Score |
|---|---|
| ROC-AUC | ~0.85 |
| Recall (Churn class) | ~0.80 |
| Precision | ~0.68 |
| F1-Score | ~0.73 |

> Exact numbers vary by run due to randomized search. Set `RANDOM_STATE = 42` in `src/config.py` for reproducibility.

---

## License

This project is released for educational and portfolio purposes.

"""
Prediction Module for Customer Churn Prediction
Single-record and batch inference utilities.
"""
import logging
import pickle
import numpy as np
import pandas as pd

try:
    from src.config import BEST_MODEL_FILE, PREPROCESSOR_FILE
    from src.data_preprocessing import preprocess_single_record
except ImportError:
    from config import BEST_MODEL_FILE, PREPROCESSOR_FILE
    from data_preprocessing import preprocess_single_record

logger = logging.getLogger(__name__)


def load_artifacts():
    """Load model + preprocessor from disk."""
    with open(BEST_MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    with open(PREPROCESSOR_FILE, "rb") as f:
        preprocessor = pickle.load(f)
    return model, preprocessor


def predict_single(record: dict, model=None, preprocessor=None) -> dict:
    """
    Predict churn probability for a single customer record.

    Parameters
    ----------
    record : dict
        Raw feature values (matching the original dataset columns, minus customerID and Churn).
    model : sklearn estimator (optional, loads from disk if None)
    preprocessor : sklearn ColumnTransformer (optional)

    Returns
    -------
    dict with keys: churn_probability, churn_prediction, risk_level
    """
    if model is None or preprocessor is None:
        model, preprocessor = load_artifacts()

    X = preprocess_single_record(record, preprocessor)
    prob = float(model.predict_proba(X)[0, 1])
    prediction = int(prob >= 0.5)

    if prob < 0.3:
        risk = "Low"
    elif prob < 0.6:
        risk = "Medium"
    else:
        risk = "High"

    return {
        "churn_probability": round(prob, 4),
        "churn_prediction": prediction,  # 1 = will churn, 0 = will not
        "risk_level": risk
    }


def predict_batch(df: pd.DataFrame, model=None, preprocessor=None) -> pd.DataFrame:
    """
    Predict churn for a batch of customers.

    Parameters
    ----------
    df : raw DataFrame (same columns as training data, without Churn column)

    Returns
    -------
    DataFrame with added columns: churn_probability, churn_prediction, risk_level
    """
    if model is None or preprocessor is None:
        model, preprocessor = load_artifacts()

    df_proc = preprocessor.transform(df)
    probs = model.predict_proba(df_proc)[:, 1]
    predictions = (probs >= 0.5).astype(int)

    risk_levels = pd.cut(
        probs,
        bins=[0, 0.3, 0.6, 1.0],
        labels=["Low", "Medium", "High"],
        include_lowest=True
    )

    result_df = df.copy()
    result_df["churn_probability"] = np.round(probs, 4)
    result_df["churn_prediction"] = predictions
    result_df["risk_level"] = risk_levels.astype(str)

    return result_df

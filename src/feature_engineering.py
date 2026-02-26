"""
Feature Engineering Module for Customer Churn Prediction
Creates domain-driven features on top of raw columns.
"""
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def add_tenure_group(df: pd.DataFrame) -> pd.DataFrame:
    """Bin tenure into meaningful lifecycle groups."""
    bins = [0, 6, 12, 24, 48, 72]
    labels = ["0-6mo", "6-12mo", "1-2yr", "2-4yr", "4+yr"]
    df["tenure_group"] = pd.cut(
        df["tenure"], bins=bins, labels=labels, include_lowest=True
    )
    df["tenure_group"] = df["tenure_group"].astype(str)
    return df


def add_charge_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create charge-related ratio features."""
    # Total charges as multiple of monthly (avoids div by zero)
    df["charges_per_month_ratio"] = np.where(
        df["MonthlyCharges"] > 0,
        df["TotalCharges"] / (df["MonthlyCharges"] + 1e-9),
        0.0
    )

    # Monthly charges tier
    df["monthly_charges_group"] = pd.cut(
        df["MonthlyCharges"],
        bins=[0, 35, 65, 95, 200],
        labels=["low", "medium", "high", "very_high"],
        include_lowest=True
    ).astype(str)

    return df


def add_service_count(df: pd.DataFrame) -> pd.DataFrame:
    """Count number of value-add services subscribed."""
    service_cols = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]

    positive_values = {"Yes", "DSL", "Fiber optic"}

    def count_services(row):
        return sum(
            1 for col in service_cols
            if col in row.index and str(row[col]) in positive_values
        )

    df["num_services"] = df.apply(count_services, axis=1)
    return df


def add_security_backup_combo(df: pd.DataFrame) -> pd.DataFrame:
    """Flag customers with both Online Security and Online Backup."""
    df["has_online_security_and_backup"] = (
        (df.get("OnlineSecurity", "No") == "Yes") &
        (df.get("OnlineBackup", "No") == "Yes")
    ).astype(int)
    return df


def add_all_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps in sequence."""
    logger.info("Applying feature engineering...")
    df = add_tenure_group(df)
    df = add_charge_features(df)
    df = add_service_count(df)
    df = add_security_backup_combo(df)
    logger.info(f"Feature engineering complete. Final columns: {df.shape[1]}")
    return df

"""
Data Preprocessing Module for Customer Churn Prediction
Handles loading, cleaning, and transforming raw data into ML-ready format.
"""
import logging
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import pickle

warnings.filterwarnings('ignore')

try:
    from src.config import (
        TARGET_COLUMN, CUSTOMER_ID_COLUMN, NUMERICAL_FEATURES,
        CATEGORICAL_FEATURES, RANDOM_STATE, TEST_SIZE,
        PREPROCESSOR_FILE, USE_SMOTE, SMOTE_SAMPLING_STRATEGY
    )
except ImportError:
    from config import (
        TARGET_COLUMN, CUSTOMER_ID_COLUMN, NUMERICAL_FEATURES,
        CATEGORICAL_FEATURES, RANDOM_STATE, TEST_SIZE,
        PREPROCESSOR_FILE, USE_SMOTE, SMOTE_SAMPLING_STRATEGY
    )

logger = logging.getLogger(__name__)


def load_data(filepath: str) -> pd.DataFrame:
    """Load raw CSV data."""
    logger.info(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Comprehensive data cleaning:
    - Drop customerID
    - Fix TotalCharges (convert to numeric, fill NaN)
    - Convert SeniorCitizen to Yes/No string
    - Encode target variable
    """
    logger.info("Starting data cleaning...")
    df = df.copy()

    # Drop customer ID
    if CUSTOMER_ID_COLUMN in df.columns:
        df.drop(columns=[CUSTOMER_ID_COLUMN], inplace=True)
        logger.info(f"Dropped column: {CUSTOMER_ID_COLUMN}")

    # Fix TotalCharges - convert to numeric (blanks become NaN)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Fill NaN TotalCharges with tenure * MonthlyCharges (logical imputation)
    nan_mask = df["TotalCharges"].isna()
    df.loc[nan_mask, "TotalCharges"] = (
        df.loc[nan_mask, "tenure"] * df.loc[nan_mask, "MonthlyCharges"]
    )
    logger.info(f"Fixed {nan_mask.sum()} missing TotalCharges values.")

    # Convert SeniorCitizen from int (0/1) to string (No/Yes)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    # Encode target variable: Yes -> 1, No -> 0
    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = (df[TARGET_COLUMN] == "Yes").astype(int)
        logger.info(f"Encoded {TARGET_COLUMN}: Yes=1, No=0")

    logger.info(f"Data cleaning completed. Shape: {df.shape}")
    return df


def get_preprocessor() -> ColumnTransformer:
    """
    Build sklearn ColumnTransformer for:
    - Numerical: StandardScaler
    - Categorical: OneHotEncoder
    """
    numerical_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("ohe", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False))
    ])

    all_numerical = NUMERICAL_FEATURES
    all_categorical = CATEGORICAL_FEATURES

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, all_numerical),
            ("cat", categorical_pipeline, all_categorical)
        ],
        remainder="passthrough"
    )

    return preprocessor


def apply_smote(X_train: np.ndarray, y_train: np.ndarray):
    """Apply SMOTE to handle class imbalance."""
    if not USE_SMOTE:
        return X_train, y_train

    logger.info("Applying SMOTE oversampling...")
    original_pos = y_train.sum()
    smote = SMOTE(
        sampling_strategy=SMOTE_SAMPLING_STRATEGY,
        random_state=RANDOM_STATE
    )
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    new_pos = y_resampled.sum()
    logger.info(
        f"SMOTE: {len(y_train)} -> {len(y_resampled)} samples. "
        f"Positive class: {original_pos} -> {new_pos}"
    )
    return X_resampled, y_resampled


def split_and_preprocess(df: pd.DataFrame):
    """
    Full preprocessing workflow:
    1. Split features / target
    2. Train/test split
    3. Fit preprocessor on train, transform both
    4. Apply SMOTE to train set
    Returns: X_train, X_test, y_train, y_test, preprocessor, feature_names
    """
    logger.info("Splitting data into features and target...")

    # Separate features and target
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    logger.info(
        f"Train size: {len(X_train)}, Test size: {len(X_test)}"
    )

    # Build and fit preprocessor
    preprocessor = get_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Get feature names after transformation
    try:
        ohe_features = (
            preprocessor
            .named_transformers_["cat"]
            .named_steps["ohe"]
            .get_feature_names_out(CATEGORICAL_FEATURES)
            .tolist()
        )
    except Exception:
        ohe_features = []

    feature_names = NUMERICAL_FEATURES + ohe_features

    # Save preprocessor
    with open(PREPROCESSOR_FILE, "wb") as f:
        pickle.dump(preprocessor, f)
    logger.info(f"Preprocessor saved to: {PREPROCESSOR_FILE}")

    # Apply SMOTE on training data
    X_train_final, y_train_final = apply_smote(X_train_proc, y_train.values)

    return (
        X_train_final, X_test_proc,
        y_train_final, y_test.values,
        preprocessor, feature_names
    )


def preprocess_single_record(record: dict, preprocessor) -> np.ndarray:
    """
    Preprocess a single customer record for inference.
    record: dict with raw feature values (before encoding/scaling)
    Returns numpy array ready for model prediction.
    """
    df_single = pd.DataFrame([record])

    # Fix SeniorCitizen if numeric
    if "SeniorCitizen" in df_single.columns:
        val = df_single["SeniorCitizen"].iloc[0]
        if isinstance(val, (int, float)):
            df_single["SeniorCitizen"] = "Yes" if val == 1 else "No"

    # Fix TotalCharges
    df_single["TotalCharges"] = pd.to_numeric(
        df_single["TotalCharges"], errors="coerce"
    )
    if df_single["TotalCharges"].isna().any():
        df_single["TotalCharges"] = (
            df_single["tenure"] * df_single["MonthlyCharges"]
        )

    return preprocessor.transform(df_single)

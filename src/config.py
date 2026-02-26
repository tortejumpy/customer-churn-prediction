"""
Configuration Settings for Customer Churn Prediction Pipeline
"""
import os

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
SRC_DIR = os.path.join(BASE_DIR, "src")

# Data
RAW_DATA_FILE = os.path.join(BASE_DIR, "Customer_data - customer_data.csv")
PROCESSED_DATA_FILE = os.path.join(DATA_DIR, "processed_data.csv")
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")

# Models
BEST_MODEL_FILE = os.path.join(MODELS_DIR, "best_model.pkl")
PREPROCESSOR_FILE = os.path.join(MODELS_DIR, "preprocessor.pkl")
MODEL_METADATA_FILE = os.path.join(MODELS_DIR, "model_metadata.json")

# ============================================================
# DATA SETTINGS
# ============================================================
TARGET_COLUMN = "Churn"
CUSTOMER_ID_COLUMN = "customerID"

NUMERICAL_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]

ENGINEERED_FEATURES = [
    "tenure_group",
    "charges_per_month_ratio",
    "num_services",
    "has_online_security_and_backup",
    "monthly_charges_group"
]

# ============================================================
# MODEL SETTINGS
# ============================================================
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
SCORING_METRIC = "roc_auc"

# ============================================================
# SMOTE SETTINGS
# ============================================================
USE_SMOTE = True
SMOTE_SAMPLING_STRATEGY = "auto"

# ============================================================
# HYPERPARAMETER SEARCH SETTINGS
# ============================================================
N_ITER_RANDOMIZED = 50  # For RandomizedSearchCV
N_JOBS = -1
VERBOSE = 1

# ============================================================
# LOGGING
# ============================================================
LOG_FILE = os.path.join(BASE_DIR, "pipeline.log")
LOG_LEVEL = "INFO"

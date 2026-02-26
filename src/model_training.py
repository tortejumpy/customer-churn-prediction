"""
Model Training Module for Customer Churn Prediction
Experiments with multiple classifiers, hyperparameter tuning, and selects best model.
"""
import logging
import time
import warnings
import json
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import (
    RandomizedSearchCV, StratifiedKFold, cross_val_score
)
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')

try:
    from src.config import (
        RANDOM_STATE, CV_FOLDS, SCORING_METRIC, N_ITER_RANDOMIZED,
        N_JOBS, BEST_MODEL_FILE, MODEL_METADATA_FILE, VERBOSE
    )
except ImportError:
    from config import (
        RANDOM_STATE, CV_FOLDS, SCORING_METRIC, N_ITER_RANDOMIZED,
        N_JOBS, BEST_MODEL_FILE, MODEL_METADATA_FILE, VERBOSE
    )

logger = logging.getLogger(__name__)


# ============================================================
# MODEL CATALOGUE
# ============================================================
def get_model_catalogue():
    """Return dict of models with their hyperparameter search spaces."""
    catalogue = {
        "LogisticRegression": {
            "model": LogisticRegression(
                random_state=RANDOM_STATE, max_iter=1000, solver="saga"
            ),
            "params": {
                "C": [0.001, 0.01, 0.1, 1, 10, 100],
                "penalty": ["l1", "l2", "elasticnet"],
                "l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0]
            }
        },
        "RandomForest": {
            "model": RandomForestClassifier(
                random_state=RANDOM_STATE, n_jobs=N_JOBS
            ),
            "params": {
                "n_estimators": [100, 200, 300, 500],
                "max_depth": [None, 5, 10, 15, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2", 0.5]
            }
        },
        "GradientBoosting": {
            "model": GradientBoostingClassifier(random_state=RANDOM_STATE),
            "params": {
                "n_estimators": [100, 200, 300, 500],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "max_depth": [3, 4, 5, 6],
                "subsample": [0.7, 0.8, 0.9, 1.0],
                "min_samples_split": [2, 5],
                "max_features": ["sqrt", "log2"]
            }
        },
        "XGBoost": {
            "model": xgb.XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                use_label_encoder=False,
                n_jobs=N_JOBS
            ),
            "params": {
                "n_estimators": [100, 200, 300, 500],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "max_depth": [3, 4, 5, 6, 8],
                "subsample": [0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
                "reg_alpha": [0, 0.1, 1.0],
                "reg_lambda": [0.1, 1.0, 10]
            }
        },
        "LightGBM": {
            "model": lgb.LGBMClassifier(
                random_state=RANDOM_STATE,
                n_jobs=N_JOBS,
                verbose=-1
            ),
            "params": {
                "n_estimators": [100, 200, 300, 500],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "max_depth": [-1, 5, 10, 15],
                "num_leaves": [20, 31, 50, 100],
                "subsample": [0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
                "min_child_samples": [10, 20, 50]
            }
        },
        "ExtraTrees": {
            "model": ExtraTreesClassifier(
                random_state=RANDOM_STATE, n_jobs=N_JOBS
            ),
            "params": {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 5, 10, 15],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            }
        },
        "AdaBoost": {
            "model": AdaBoostClassifier(random_state=RANDOM_STATE),
            "params": {
                "n_estimators": [50, 100, 200, 300],
                "learning_rate": [0.01, 0.1, 0.5, 1.0]
            }
        },
        "MLP": {
            "model": MLPClassifier(
                random_state=RANDOM_STATE, max_iter=500
            ),
            "params": {
                "hidden_layer_sizes": [
                    (64,), (128,), (64, 32), (128, 64), (256, 128, 64)
                ],
                "activation": ["relu", "tanh"],
                "alpha": [0.0001, 0.001, 0.01],
                "learning_rate": ["constant", "adaptive"]
            }
        }
    }
    return catalogue


# ============================================================
# QUICK BASELINE
# ============================================================
def run_baseline_comparison(X_train, y_train, X_test, y_test):
    """
    Run quick baseline comparison (no tuning) for all models.
    Returns sorted results dict.
    """
    logger.info("=" * 60)
    logger.info("BASELINE MODEL COMPARISON (DEFAULT PARAMS)")
    logger.info("=" * 60)

    catalogue = get_model_catalogue()
    results = {}
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for name, config in catalogue.items():
        try:
            model = config["model"]
            t0 = time.time()
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=cv, scoring=SCORING_METRIC, n_jobs=N_JOBS
            )
            elapsed = time.time() - t0

            results[name] = {
                "cv_mean": float(cv_scores.mean()),
                "cv_std": float(cv_scores.std()),
                "train_time": round(elapsed, 2)
            }
            logger.info(
                f"{name:25s} | CV AUC: {cv_scores.mean():.4f} "
                f"(+/-{cv_scores.std():.4f}) | Time: {elapsed:.1f}s"
            )
        except Exception as e:
            logger.warning(f"Skipping {name}: {e}")

    # Sort by cv_mean
    sorted_results = sorted(results.items(), key=lambda x: x[1]["cv_mean"], reverse=True)
    return dict(sorted_results)


# ============================================================
# HYPERPARAMETER TUNING
# ============================================================
def tune_model(name: str, X_train, y_train, n_iter: int = None):
    """
    RandomizedSearchCV tuning for a given model.
    Returns best model, best params, best score.
    """
    if n_iter is None:
        n_iter = N_ITER_RANDOMIZED

    catalogue = get_model_catalogue()
    if name not in catalogue:
        raise ValueError(f"Model '{name}' not in catalogue.")

    config = catalogue[name]
    model = config["model"]
    params = config["params"]

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    logger.info(f"Tuning {name} with {n_iter} iterations...")
    t0 = time.time()

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=params,
        n_iter=n_iter,
        scoring=SCORING_METRIC,
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        verbose=VERBOSE,
        refit=True
    )
    search.fit(X_train, y_train)

    elapsed = time.time() - t0
    logger.info(
        f"{name} tuning done in {elapsed:.1f}s. "
        f"Best CV AUC: {search.best_score_:.4f}"
    )
    logger.info(f"Best params: {search.best_params_}")

    return search.best_estimator_, search.best_params_, search.best_score_


# ============================================================
# FULL EXPERIMENT
# ============================================================
def run_full_experiment(X_train, y_train, X_test, y_test, top_n: int = 3):
    """
    1. Baseline comparison for all models
    2. Tune top N models
    3. Select best model by test AUC
    4. Save best model + metadata
    Returns: best_model, all_results
    """
    logger.info("=" * 60)
    logger.info("STARTING FULL MODEL EXPERIMENT")
    logger.info("=" * 60)

    # Step 1: Baseline
    baseline_results = run_baseline_comparison(X_train, y_train, X_test, y_test)

    # Step 2: Tune top N
    top_models = list(baseline_results.keys())[:top_n]
    logger.info(f"\nTuning top {top_n} models: {top_models}")

    tuned_results = {}
    best_model = None
    best_test_auc = 0.0
    best_name = ""

    for name in top_models:
        try:
            model, params, cv_score = tune_model(name, X_train, y_train)
            model.fit(X_train, y_train)
            y_pred_prob = model.predict_proba(X_test)[:, 1]
            test_auc = roc_auc_score(y_test, y_pred_prob)

            tuned_results[name] = {
                "cv_auc": float(cv_score),
                "test_auc": float(test_auc),
                "best_params": {k: str(v) for k, v in params.items()}
            }

            logger.info(f"{name}: Test AUC = {test_auc:.4f}")

            if test_auc > best_test_auc:
                best_test_auc = test_auc
                best_model = model
                best_name = name

        except Exception as e:
            logger.warning(f"Tuning failed for {name}: {e}")

    # Step 3: Save best model
    all_results = {
        "baseline": baseline_results,
        "tuned": tuned_results,
        "best_model": best_name,
        "best_test_auc": best_test_auc
    }

    with open(BEST_MODEL_FILE, "wb") as f:
        pickle.dump(best_model, f)
    logger.info(f"Best model '{best_name}' saved to: {BEST_MODEL_FILE}")

    with open(MODEL_METADATA_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Model metadata saved to: {MODEL_METADATA_FILE}")

    return best_model, all_results


def load_best_model():
    """Load saved best model from disk."""
    with open(BEST_MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    return model

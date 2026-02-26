"""
Main Training Script — Customer Churn Prediction Pipeline
Run this to execute the complete end-to-end ML pipeline.

Usage:
    python train.py
    python train.py --top-n 3
"""
import argparse
import os
import sys
import logging

# Allow imports from src/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    RAW_DATA_FILE, DATA_DIR, MODELS_DIR, LOG_FILE, LOG_LEVEL, RANDOM_STATE
)
from src.utils import setup_logging, set_random_seed, ensure_dirs, print_section
from src.data_preprocessing import load_data, clean_data, split_and_preprocess
from src.model_training import run_full_experiment
from src.model_evaluation import (
    compute_all_metrics, find_optimal_threshold,
    plot_confusion_matrix, plot_roc_curve,
    plot_precision_recall_curve, plot_feature_importance,
    plot_model_comparison
)


def parse_args():
    parser = argparse.ArgumentParser(description="Customer Churn Prediction — Training Pipeline")
    parser.add_argument("--top-n", type=int, default=3,
                        help="Number of top baseline models to tune (default: 3)")
    parser.add_argument("--data", type=str, default=RAW_DATA_FILE,
                        help="Path to raw CSV data")
    parser.add_argument("--no-plots", action="store_true",
                        help="Suppress evaluation plots")
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup
    logger = setup_logging(LOG_FILE, LOG_LEVEL)
    set_random_seed(RANDOM_STATE)
    ensure_dirs(DATA_DIR, MODELS_DIR)

    print_section("CUSTOMER CHURN PREDICTION — FULL PIPELINE")
    logger.info("Pipeline started.")
    logger.info(f"Data source: {args.data}")

    # --------------------------------------------------------
    # STEP 1: Load & Clean Data
    # --------------------------------------------------------
    print_section("STEP 1: Data Loading & Cleaning")
    df_raw = load_data(args.data)
    df_clean = clean_data(df_raw)
    logger.info(f"Cleaned data shape: {df_clean.shape}")
    logger.info(f"Churn distribution:\n{df_clean['Churn'].value_counts()}")

    # --------------------------------------------------------
    # STEP 2: Preprocessing & Feature Engineering
    # --------------------------------------------------------
    print_section("STEP 2: Preprocessing & Feature Engineering")
    (
        X_train, X_test,
        y_train, y_test,
        preprocessor, feature_names
    ) = split_and_preprocess(df_clean)

    logger.info(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    # --------------------------------------------------------
    # STEP 3: Model Experiments + Hyperparameter Tuning
    # --------------------------------------------------------
    print_section("STEP 3: Model Experiments & Tuning")
    best_model, all_results = run_full_experiment(
        X_train, y_train,
        X_test, y_test,
        top_n=args.top_n
    )

    logger.info(f"\nBest model: {all_results['best_model']}")
    logger.info(f"Best Test AUC: {all_results['best_test_auc']:.4f}")

    # --------------------------------------------------------
    # STEP 4: Evaluation
    # --------------------------------------------------------
    print_section("STEP 4: Final Evaluation")
    y_pred_prob = best_model.predict_proba(X_test)[:, 1]

    # Optimal threshold
    optimal_threshold = find_optimal_threshold(y_test, y_pred_prob)
    metrics = compute_all_metrics(y_test, y_pred_prob, threshold=optimal_threshold)

    if not args.no_plots:
        plots_dir = os.path.join(MODELS_DIR, "plots")
        ensure_dirs(plots_dir)

        plot_confusion_matrix(
            y_test, y_pred_prob,
            threshold=optimal_threshold,
            save_path=os.path.join(plots_dir, "confusion_matrix.png")
        )
        plot_roc_curve(
            y_test, y_pred_prob,
            model_name=all_results["best_model"],
            save_path=os.path.join(plots_dir, "roc_curve.png")
        )
        plot_precision_recall_curve(
            y_test, y_pred_prob,
            model_name=all_results["best_model"],
            save_path=os.path.join(plots_dir, "pr_curve.png")
        )
        plot_feature_importance(
            best_model, feature_names,
            top_n=20,
            save_path=os.path.join(plots_dir, "feature_importance.png")
        )
        # Model comparison
        plot_model_comparison(
            all_results["baseline"],
            save_path=os.path.join(plots_dir, "model_comparison.png")
        )

    # --------------------------------------------------------
    # DONE
    # --------------------------------------------------------
    print_section("PIPELINE COMPLETE")
    logger.info(f"Best Model:          {all_results['best_model']}")
    logger.info(f"Test ROC-AUC:        {metrics['roc_auc']:.4f}")
    logger.info(f"Test Recall (Churn): {metrics['recall']:.4f}")
    logger.info(f"Test Precision:      {metrics['precision']:.4f}")
    logger.info(f"Test F1-Score:       {metrics['f1_score']:.4f}")
    logger.info("Models saved to: models/ directory")
    print("\n✅ Training pipeline completed successfully!\n")


if __name__ == "__main__":
    main()

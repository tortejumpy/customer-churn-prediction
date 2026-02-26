"""
Model Evaluation Module for Customer Churn Prediction
Comprehensive evaluation metrics, plots, and reporting.
"""
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score, accuracy_score
)

logger = logging.getLogger(__name__)


def compute_all_metrics(y_true, y_pred_prob, threshold: float = 0.5) -> dict:
    """Compute a comprehensive set of evaluation metrics."""
    y_pred = (y_pred_prob >= threshold).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_pred_prob)),
        "average_precision": float(average_precision_score(y_true, y_pred_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold_used": threshold
    }

    # Per-class metrics
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics["class_report"] = report

    logger.info("\n" + "=" * 50)
    logger.info("EVALUATION METRICS")
    logger.info("=" * 50)
    for k, v in metrics.items():
        if k != "class_report":
            logger.info(f"  {k:25s}: {v:.4f}")
    logger.info("=" * 50)

    return metrics


def find_optimal_threshold(y_true, y_pred_prob):
    """Find optimal threshold maximising F1-score."""
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_f1, best_threshold = 0, 0.5
    for t in thresholds:
        y_pred = (y_pred_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
    logger.info(f"Optimal threshold: {best_threshold:.2f} (F1={best_f1:.4f})")
    return best_threshold


def plot_confusion_matrix(y_true, y_pred_prob, threshold=0.5, save_path=None):
    """Plot styled confusion matrix."""
    y_pred = (y_pred_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Not Churn", "Churn"],
        yticklabels=["Not Churn", "Churn"],
        ax=ax, linewidths=0.5
    )
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_xlabel("Predicted", fontsize=11)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info(f"Confusion matrix saved to: {save_path}")
    plt.show()
    plt.close()
    return cm


def plot_roc_curve(y_true, y_pred_prob, model_name="Model", save_path=None):
    """Plot ROC curve with AUC annotation."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    auc = roc_auc_score(y_true, y_pred_prob)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, lw=2, color="#3498DB", label=f"{model_name} (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], lw=1.5, linestyle="--", color="gray", label="Random baseline")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#3498DB")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="lower right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info(f"ROC curve saved to: {save_path}")
    plt.show()
    plt.close()


def plot_precision_recall_curve(y_true, y_pred_prob, model_name="Model", save_path=None):
    """Plot Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    ap = average_precision_score(y_true, y_pred_prob)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, lw=2, color="#E74C3C", label=f"{model_name} (AP={ap:.4f})")
    ax.fill_between(recall, precision, alpha=0.1, color="#E74C3C")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="upper right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info(f"PR curve saved to: {save_path}")
    plt.show()
    plt.close()


def plot_feature_importance(model, feature_names, top_n=20, save_path=None):
    """
    Plot feature importance (works for tree-based models).
    Falls back gracefully for other model types.
    """
    try:
        importances = model.feature_importances_
    except AttributeError:
        logger.warning("Model does not support feature_importances_. Skipping plot.")
        return None

    # Sort and select top N
    idx = np.argsort(importances)[::-1][:top_n]
    top_importances = importances[idx]
    top_names = [feature_names[i] if i < len(feature_names) else f"feat_{i}" for i in idx]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n // 2)))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, top_n))
    bars = ax.barh(range(len(top_names)), top_importances[::-1], color=colors[::-1])
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names[::-1], fontsize=10)
    ax.set_xlabel("Feature Importance", fontsize=11)
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Feature importance saved to: {save_path}")
    plt.show()
    plt.close()

    return list(zip(top_names, top_importances))


def plot_model_comparison(results: dict, save_path=None):
    """Bar chart comparing all models by CV AUC."""
    model_names = list(results.keys())
    cv_aucs = [results[n]["cv_mean"] for n in model_names]
    cv_stds = [results[n]["cv_std"] for n in model_names]

    sorted_idx = np.argsort(cv_aucs)[::-1]
    model_names = [model_names[i] for i in sorted_idx]
    cv_aucs = [cv_aucs[i] for i in sorted_idx]
    cv_stds = [cv_stds[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2ECC71" if i == 0 else "#3498DB" for i in range(len(model_names))]
    bars = ax.bar(model_names, cv_aucs, yerr=cv_stds, capsize=5,
                  color=colors, edgecolor="white", linewidth=1.2)
    ax.set_ylabel("CV ROC-AUC", fontsize=11)
    ax.set_title("Model Comparison — Cross-Validation AUC", fontsize=13, fontweight="bold")
    ax.set_ylim(bottom=max(0, min(cv_aucs) - 0.05))
    plt.xticks(rotation=20, ha="right")

    # Annotate bars
    for bar, val in zip(bars, cv_aucs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{val:.4f}", ha="center", va="bottom", fontsize=9
        )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Model comparison plot saved to: {save_path}")
    plt.show()
    plt.close()

"""
Utility Functions for Customer Churn Prediction
Logging setup, plotting helpers, reproducibility.
"""
import os
import logging
import random
import numpy as np


def setup_logging(log_file: str = None, level: str = "INFO"):
    """Configure logging to stdout and optionally a file."""
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="a"))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True
    )
    return logging.getLogger("churn")


def set_random_seed(seed: int = 42):
    """Ensure reproducibility across numpy and python random."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dirs(*paths):
    """Create directories if they don't exist."""
    for path in paths:
        os.makedirs(path, exist_ok=True)


def print_section(title: str, width: int = 60):
    """Pretty-print a section header."""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)

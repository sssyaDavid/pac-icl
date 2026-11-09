"""
Evaluation metrics
"""
import numpy as np
from typing import List, Dict


def accuracy(predictions: np.ndarray, labels: List[int]) -> float:
    labels = np.array(labels)
    return float(np.mean(predictions == labels))


def summarize_sigma(sigma_stars: np.ndarray) -> Dict:
    return {
        "mean": float(np.mean(sigma_stars)),
        "median": float(np.median(sigma_stars)),
        "min": float(np.min(sigma_stars)),
        "max": float(np.max(sigma_stars)),
        "std": float(np.std(sigma_stars)),
    }


def margin_vs_sigma(
    votes: np.ndarray,
    sigma_stars: np.ndarray,
) -> Dict:
    """Compute the correlation between vote margin and σ*."""
    from src.aggregation.pac import vote_margin
    margins = np.array([vote_margin(v) for v in votes])
    corr = float(np.corrcoef(margins, sigma_stars)[0, 1])
    return {"margins": margins.tolist(), "sigma_stars": sigma_stars.tolist(), "correlation": corr}
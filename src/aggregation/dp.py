"""
DP aggregation baseline: Report Noisy Max + Gaussian mechanism
--------------------------------------------------------------
The standard PATE aggregation used in the literature, satisfying (ε, δ)-differential privacy.
"""

import numpy as np
from typing import Tuple


def gaussian_sigma(epsilon: float, delta: float, sensitivity: float = np.sqrt(2)) -> float:
    """
    Noise standard deviation of the Gaussian mechanism.
    σ = sensitivity × √(2 ln(1.25/δ)) / ε

    Global sensitivity Δ = √2 (one teacher changing its vote shifts two counts by ±1,
    so the L2 norm is √2).
    """
    return sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon


def dp_aggregate_single(
    vote: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> Tuple[int, np.ndarray]:
    """
    Run DP Report Noisy Max on a single vote histogram.

    Returns:
        predicted_label : int
        noisy_votes     : np.ndarray
    """
    noisy = vote + rng.standard_normal(len(vote)) * sigma
    return int(np.argmax(noisy)), noisy


def dp_aggregate_batch(
    votes: np.ndarray,
    epsilon: float,
    delta: float,
    seed: int = 42,
) -> Tuple[np.ndarray, float]:
    """
    Run DP aggregation on a batch of votes.

    Args:
        votes   : shape (N, C)
        epsilon : privacy budget ε
        delta   : privacy parameter δ

    Returns:
        predictions : shape (N,), predicted label for each sample
        sigma       : noise standard deviation used (fixed)
    """
    sigma = gaussian_sigma(epsilon, delta)
    rng = np.random.default_rng(seed)

    N = len(votes)
    predictions = np.zeros(N, dtype=int)

    for i in range(N):
        predictions[i], _ = dp_aggregate_single(votes[i], sigma, rng)

    return predictions, sigma

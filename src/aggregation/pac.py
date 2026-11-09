"""
PAC-private aggregation: adaptive noise calibration
---------------------------------------------------
Idea:
  1. For each vote histogram v, estimate I(v; v + N(0, σ²I)) with Monte Carlo
  2. Binary-search the smallest σ* such that I ≤ η (target privacy budget)
  3. Add noise and return argmax

Advantage: the more decisive the vote (larger margin), the smaller σ*, and the higher the accuracy.
"""

import numpy as np
from typing import Tuple, List
from tqdm import tqdm

from src.privacy.mi_estimator import estimate_mi_for_vote


def calibrate_sigma(
    vote: np.ndarray,
    vote_pool: np.ndarray,
    eta: float,
    mc_samples: int = 500,
    sigma_low: float = 0.01,
    sigma_high: float = 30.0,
    tol: float = 0.05,
    rng: np.random.Generator = None,
) -> float:
    """
    Binary-search the smallest σ* such that I(vote; M_σ(vote)) ≤ η.

    If sigma_low already meets the privacy requirement, the vote is safe enough
    on its own, so return sigma_low (very little noise).
    """
    if rng is None:
        rng = np.random.default_rng()

    # Check whether the minimum noise already suffices
    mi_at_low = estimate_mi_for_vote(vote, vote_pool, sigma_low, mc_samples, rng)
    if mi_at_low <= eta:
        return sigma_low

    # Check whether the maximum noise is enough
    mi_at_high = estimate_mi_for_vote(vote, vote_pool, sigma_high, mc_samples, rng)
    if mi_at_high > eta:
        # Extreme case: even large noise is not enough, so return sigma_high
        return sigma_high

    # Binary search
    lo, hi = sigma_low, sigma_high
    while hi - lo > tol:
        mid = (lo + hi) / 2.0
        mi = estimate_mi_for_vote(vote, vote_pool, mid, mc_samples, rng)
        if mi <= eta:
            hi = mid
        else:
            lo = mid

    return hi


def pac_aggregate_single(
    vote: np.ndarray,
    vote_pool: np.ndarray,
    eta: float,
    mc_samples: int,
    sigma_low: float,
    sigma_high: float,
    tol: float,
    rng: np.random.Generator,
) -> Tuple[int, float]:
    """
    Run PAC adaptive aggregation on a single vote.

    Returns:
        predicted_label : int
        sigma_star      : float (noise scale actually used)
    """
    sigma_star = calibrate_sigma(
        vote, vote_pool, eta, mc_samples, sigma_low, sigma_high, tol, rng
    )
    noisy = vote + rng.standard_normal(len(vote)) * sigma_star
    return int(np.argmax(noisy)), sigma_star


def pac_aggregate_batch(
    votes: np.ndarray,
    eta: float,
    mc_samples: int = 500,
    sigma_low: float = 0.01,
    sigma_high: float = 30.0,
    tol: float = 0.05,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run PAC aggregation on a batch of votes.

    Args:
        votes      : shape (N, C)
        eta        : target mutual-information upper bound (privacy budget, nats)
        mc_samples : number of Monte Carlo samples
        sigma_low/high : binary search range
        tol        : search tolerance

    Returns:
        predictions  : shape (N,), predicted label for each sample
        sigma_stars  : shape (N,), σ* used for each sample
    """
    rng = np.random.default_rng(seed)
    N = len(votes)
    predictions = np.zeros(N, dtype=int)
    sigma_stars = np.zeros(N)

    for i in tqdm(range(N), desc="PAC adaptive aggregation", leave=False):
        # Drop sample i itself so the vote pool is not contaminated
        pool_idx = np.concatenate([np.arange(i), np.arange(i + 1, N)])
        vote_pool_i = votes[pool_idx]
        predictions[i], sigma_stars[i] = pac_aggregate_single(
            votes[i], vote_pool_i, eta, mc_samples,
            sigma_low, sigma_high, tol, rng
        )

    return predictions, sigma_stars


def vote_margin(vote: np.ndarray) -> int:
    """Vote margin = top count minus second count. Larger means more decisive."""
    sorted_v = np.sort(vote)[::-1]
    return int(sorted_v[0] - sorted_v[1]) if len(sorted_v) > 1 else int(sorted_v[0])

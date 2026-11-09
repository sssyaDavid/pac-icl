"""
Monte Carlo mutual information estimator
----------------------------------------
I(v ; M_σ(v)) ≈ (1/T) Σ ln[ P(v | v^(t)) / P(v) ]

where:
  v        = raw vote histogram
  v^(t)    ~ D̂  (sampled from the historical vote distribution)
  M_σ(v)   = v + N(0, σ²I)  (noisy mechanism)
  P(v|v^(t)) / P(v) is the likelihood ratio
"""

import numpy as np
from typing import Optional


def gaussian_log_likelihood(x: np.ndarray, mean: np.ndarray, sigma: float) -> float:
    """
    Multivariate Gaussian log-likelihood: ln P(x | mean, σ²I)
    """
    d = len(x)
    diff = x - mean
    return -0.5 * d * np.log(2 * np.pi * sigma**2) - 0.5 * np.sum(diff**2) / sigma**2


def estimate_mi_for_vote(
    vote: np.ndarray,
    vote_pool: np.ndarray,
    sigma: float,
    T: int = 500,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Estimate the mutual information of a single vote histogram under noise σ.

    Args:
        vote      : shape (C,), current vote histogram
        vote_pool : shape (N, C), historical vote distribution D̂ (votes of all samples)
        sigma     : noise standard deviation
        T         : number of Monte Carlo samples
        rng       : random number generator

    Returns:
        Mutual information estimate Î (nats)
    """
    if rng is None:
        rng = np.random.default_rng()

    C = len(vote)
    N = len(vote_pool)

    # Cap the reference set at 100 points to limit compute
    if N > 100:
        idx_pool = rng.choice(N, size=100, replace=False)
        vote_pool = vote_pool[idx_pool]
        N = 100

    # Add noise to the vote to obtain M_σ(vote)
    noise = rng.standard_normal((T, C)) * sigma
    m_v = vote[np.newaxis, :] + noise  # (T, C)

    # Compute log P(M_σ(v) | v)
    log_p_given_v = np.array([
        gaussian_log_likelihood(m_v[t], vote, sigma) for t in range(T)
    ])

    # Compute log P(M_σ(v)) = log E_{v'~D̂}[P(M|v')]
    # For each M sample, use every vote in the pool with logsumexp (unbiased marginal)
    from scipy.special import logsumexp

    log_p_marginal = np.zeros(T)
    for t in range(T):
        log_likelihoods = np.array([
            gaussian_log_likelihood(m_v[t], vote_pool[j], sigma)
            for j in range(N)
        ])
        # log P(M) = log(1/N * Σ P(M|v')) = logsumexp(log P(M|v')) - log N
        log_p_marginal[t] = logsumexp(log_likelihoods) - np.log(N)

    # Mutual information ≈ E[ log P(M|v) - log P(M) ]  (unbiased)
    mi = np.mean(log_p_given_v - log_p_marginal)
    return float(mi)


def estimate_mi_batch(
    votes: np.ndarray,
    sigma: float,
    T: int = 500,
    seed: int = 42,
) -> np.ndarray:
    """
    Estimate mutual information for a batch of votes.

    Args:
        votes : shape (N, C)
        sigma : noise standard deviation
        T     : number of Monte Carlo samples

    Returns:
        mi_values : shape (N,), mutual information estimate for each sample
    """
    rng = np.random.default_rng(seed)
    N = len(votes)
    mi_values = np.zeros(N)

    for i in range(N):
        mi_values[i] = estimate_mi_for_vote(
            votes[i], votes, sigma, T, rng
        )

    return mi_values

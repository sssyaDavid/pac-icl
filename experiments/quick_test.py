"""
Quick test (no LLM): verify the PAC vs DP core logic with simulated votes
-------------------------------------------------------------------------
Run:
  python experiments/quick_test.py

Results appear in about 10-30 seconds. Confirm the PAC mechanism before
running a real LLM experiment.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from src.aggregation.dp import dp_aggregate_batch, gaussian_sigma
from src.aggregation.pac import pac_aggregate_batch, vote_margin
from src.evaluation.metrics import accuracy

def simulate_votes(n_samples=200, n_teachers=10, n_classes=2, seed=42):
    """
    Simulate votes for two scenarios:
      - decisive: 70% of teachers vote for the correct class
      - uncertain: 55% of teachers vote for the correct class
    Each scenario covers half of the samples.
    """
    rng = np.random.default_rng(seed)
    votes = np.zeros((n_samples, n_classes), dtype=int)
    true_labels = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        if i < n_samples // 2:  # decisive
            correct_votes = int(n_teachers * 0.7)
        else:                    # uncertain
            correct_votes = int(n_teachers * 0.55)

        label = rng.integers(n_classes)
        true_labels[i] = label
        votes[i, label] = correct_votes
        # Assign the remaining votes to the other classes
        remaining = n_teachers - correct_votes
        for c in range(n_classes):
            if c != label:
                votes[i, c] += remaining // (n_classes - 1)

    return votes, true_labels.tolist()


def main():
    print("=" * 55)
    print("Quick test: PAC vs DP aggregation (simulated votes)")
    print("=" * 55)

    votes, true_labels = simulate_votes(n_samples=200, n_teachers=10)
    n = len(votes)
    decisive_idx = list(range(n // 2))
    uncertain_idx = list(range(n // 2, n))

    # --- DP baseline ---
    epsilon, delta = 1.0, 1e-5
    dp_sigma = gaussian_sigma(epsilon, delta)
    dp_preds, _ = dp_aggregate_batch(votes, epsilon, delta)
    dp_acc = accuracy(dp_preds, true_labels)

    # --- PAC ---
    pac_preds, pac_sigmas = pac_aggregate_batch(
        votes, eta=0.5, mc_samples=300, tol=0.1
    )
    pac_acc = accuracy(pac_preds, true_labels)

    # --- Results ---
    print(f"\nAccuracy:")
    print(f"  DP  : {dp_acc:.3f}")
    print(f"  PAC : {pac_acc:.3f}  (diff {pac_acc - dp_acc:+.3f})")

    print(f"\nMean noise:")
    print(f"  DP σ (fixed)                  : {dp_sigma:.3f}")
    print(f"  PAC σ_mean (decisive samples) : {pac_sigmas[decisive_idx].mean():.3f}")
    print(f"  PAC σ_mean (uncertain samples): {pac_sigmas[uncertain_idx].mean():.3f}")

    margins = [vote_margin(v) for v in votes]
    corr = float(np.corrcoef(margins, pac_sigmas)[0, 1])
    print(f"\nVote margin vs σ* correlation: {corr:.3f}  (expected < 0)")

    print("\n" + "=" * 55)
    if pac_sigmas[decisive_idx].mean() < dp_sigma:
        print("✓ PAC uses less noise on decisive votes")
    if corr < -0.1:
        print("✓ Larger margin means smaller PAC noise (as expected)")
    if pac_acc >= dp_acc - 0.02:
        print("✓ PAC accuracy is comparable to or higher than DP")
    print("=" * 55)


if __name__ == "__main__":
    main()

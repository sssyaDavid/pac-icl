"""
Paired MIA evaluation for PAC-Private ICL.

For each target sample z we run PATE twice with the SAME seed:
  - member run   : z is included in the training pool
  - non-member run: z is removed from the training pool

Attack signal: vote margin = top_vote - second_vote (after PAC noise).
Attack guess:  "member" if member_margin > non_member_margin.

Metrics:
  Attack Accuracy  = fraction of targets guessed correctly
  Attack Advantage = 2 * (Attack Accuracy - 0.5)
    Advantage = 0  -> random guessing, PAC privacy holds
    Advantage = 1  -> perfect attack, privacy fails

Usage
-----
python mia_eval.py --config ../prompting_configs/ollama-pate-sst2.yaml \
                   --n_targets 50 \
                   --n_mia_queries 100 \
                   --eta 2.0
"""

import argparse
import os
import random
import sys
import logging

import numpy as np
import yaml
from scipy.optimize import brentq

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# PAC helpers
# ---------------------------------------------------------------------------

def _mi_upper_bound(eigenvalues: np.ndarray, sigma: float) -> float:
    pos = eigenvalues[eigenvalues > 1e-10]
    if len(pos) == 0:
        return 0.0
    return 0.5 * float(np.sum(np.log(1.0 + pos / sigma ** 2)))


def calibrate_sigma(cov: np.ndarray, eta: float,
                    sigma_min: float = 1e-3,
                    sigma_max: float = 1e4) -> float:
    eigs = np.linalg.eigvalsh(cov)
    eigs = eigs[eigs > 1e-10]
    if len(eigs) == 0:
        return sigma_min
    if _mi_upper_bound(eigs, sigma_min) <= eta:
        return sigma_min
    try:
        return float(brentq(
            lambda s: _mi_upper_bound(eigs, s) - eta,
            sigma_min, sigma_max, xtol=1e-6, maxiter=200))
    except ValueError:
        return sigma_max


# ---------------------------------------------------------------------------
# Single PATE run
# ---------------------------------------------------------------------------

def run_pate_pac(proc, model, shot: int, K: int, eta: float,
                 seed: int,
                 train_pool_indices: list,
                 test_query_indices: list) -> np.ndarray:
    """
    Run one PAC-PATE pass.

    Returns noisy_votes of shape (n_queries, n_classes).
    Both member and non-member runs use the same seed so that
    randomness is controlled — only pool membership differs.
    """
    n_queries = len(test_query_indices)
    labels    = proc.labels
    n_classes = len(labels)
    subsample = shot * K

    # Build icl_list[j][i] = training indices for query j, teacher i
    icl_list = []
    for j in range(n_queries):
        rng_q  = random.Random(seed + j)
        pool   = list(train_pool_indices)
        rng_q.shuffle(pool)
        selected = pool[:subsample]
        icl_list.append([selected[i::K] for i in range(K)])

    # Collect per-teacher votes: shape (K, n_queries, n_classes)
    all_votes = np.zeros((K, n_queries, n_classes), dtype=np.int32)

    for i in range(K):
        train_per_query = [icl_list[j][i] for j in range(n_queries)]
        custom_split = [proc.test_dataset[i] for i in test_query_indices]
        prompts, _ = proc.create_prompts_diff(
            train_per_query,
            split="custom",
            custom_split=custom_split,
        )
        outputs = model.complete_all(prompts)

        for j, out in enumerate(outputs):
            pred = out.completion
            if pred in labels:
                all_votes[i, j, labels.index(pred)] = 1
            else:
                all_votes[i, j, np.random.randint(n_classes)] = 1

    # Add per-query PAC noise
    total_votes = all_votes.sum(axis=0)          # (n_queries, n_classes)
    noisy_votes = np.zeros_like(total_votes, dtype=float)

    for j in range(n_queries):
        teacher_q  = all_votes[:, j, :].astype(float)   # (K, C)
        cov_q      = np.cov(teacher_q.T)                 # (C, C)
        sigma_star = calibrate_sigma(cov_q, eta)
        noisy_votes[j] = (total_votes[j].astype(float)
                          + np.random.normal(0.0, sigma_star, n_classes))

    return noisy_votes   # (n_queries, n_classes)


def vote_margin(noisy_votes: np.ndarray) -> float:
    """Mean (top - second) vote across all queries."""
    sv = np.sort(noisy_votes, axis=1)[:, ::-1]
    return float(np.mean(sv[:, 0] - sv[:, 1]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",        required=True)
    parser.add_argument("--n_targets",     type=int,   default=50,
                        help="Number of MIA target samples (default 50)")
    parser.add_argument("--n_mia_queries", type=int,   default=100,
                        help="Test queries per PATE run (default 100)")
    parser.add_argument("--eta",           type=float, default=2.0,
                        help="PAC budget to evaluate (default 2.0)")
    args = parser.parse_args()

    with open(args.config) as f:
        conf = yaml.safe_load(f)

    seed  = conf.get("seed", 42)
    shot  = (conf.get("shots", [4])[0]
             if isinstance(conf.get("shots"), list)
             else conf.get("shots", 4))
    K     = conf.get("ensemble", 10)

    random.seed(seed)
    np.random.seed(seed)

    # Same construction as run_ollama.py. conf is a plain dict from yaml.safe_load.
    from prompting import MODELS, PROCESSORS

    proc = PROCESSORS[conf["task"]](
        seed=seed,
        **conf.get("processor_kwargs", {}),
    )
    model = MODELS[conf["model"]](
        conf["model_name"],
        **proc.model_kwargs,
        **(conf.get("model_kwargs") or {}),
    )

    n_train = len(proc.train_dataset)
    n_test  = len(proc.test_dataset)

    n_mia_queries = min(args.n_mia_queries, n_test)

    # Keep the pool small enough that a target is often among the 40 few-shot draws.
    pool_size = 200
    all_train = list(range(min(pool_size, n_train)))
    n_targets = min(args.n_targets, len(all_train))
    test_idx  = list(range(n_mia_queries))

    # Sample MIA targets from training pool
    rng = random.Random(seed + 7777)
    targets = rng.sample(all_train, n_targets)

    print(f"\n{'=' * 60}")
    print(f"Paired MIA Evaluation — PAC-Private ICL")
    print(f"  eta            : {args.eta}")
    print(f"  Teachers (K)   : {K}")
    print(f"  Shot           : {shot}")
    print(f"  Train pool     : {len(all_train)}")
    print(f"  Targets        : {n_targets}")
    print(f"  Test queries   : {n_mia_queries}")
    print(f"  Total LLM runs : {n_targets * 2}")
    print(f"{'=' * 60}\n")

    correct = 0   # member_margin > non_member_margin

    for idx, z in enumerate(targets):
        print(f"Target {idx + 1}/{n_targets}  (train_idx={z})")

        # --- member run: z is in the pool, use seed + idx*2 ---
        member_votes = run_pate_pac(
            proc=proc, model=model,
            shot=shot, K=K, eta=args.eta,
            seed=seed + idx * 2,
            train_pool_indices=all_train,
            test_query_indices=test_idx
        )
        m_score = vote_margin(member_votes)

        # --- non-member run: z removed, same seed ---
        pool_no_z = [i for i in all_train if i != z]
        nonmember_votes = run_pate_pac(
            proc=proc, model=model,
            shot=shot, K=K, eta=args.eta,
            seed=seed + idx * 2,          # same seed as member run
            train_pool_indices=pool_no_z,
            test_query_indices=test_idx
        )
        nm_score = vote_margin(nonmember_votes)

        won = m_score > nm_score
        correct += int(won)

        print(f"  member margin={m_score:.4f}  "
              f"non-member margin={nm_score:.4f}  "
              f"attack={'HIT' if won else 'miss'}")

    attack_acc = correct / n_targets
    advantage  = 2.0 * (attack_acc - 0.5)

    print(f"\n{'=' * 60}")
    print(f"Results (eta={args.eta}, n_targets={n_targets})")
    print(f"  Attack Accuracy  : {attack_acc:.4f}  "
          f"(random baseline = 0.5000)")
    print(f"  Attack Advantage : {advantage:+.4f}  "
          f"(0 = no information, 1 = perfect attack)")
    print(f"{'=' * 60}\n")

    # Save
    out_dir = conf.get("output_dir", "./outputs")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"mia_paired_eta{args.eta}_pool{len(all_train)}.txt")
    with open(path, "w") as f:
        f.write(f"eta={args.eta}\n")
        f.write(f"K={K}, shot={shot}, pool={len(all_train)}, "
                f"n_targets={n_targets}, n_mia_queries={n_mia_queries}\n")
        f.write(f"attack_accuracy={attack_acc:.6f}\n")
        f.write(f"attack_advantage={advantage:.6f}\n")
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
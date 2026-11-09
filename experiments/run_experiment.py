"""
Main experiment script
----------------------
Usage:
  python experiments/run_experiment.py --config configs/experiment.yaml

Output:
  results/<timestamp>/
    votes.npy          raw vote histograms
    dp_results.json    DP aggregation results
    pac_results.json   PAC aggregation results
    comparison.png     comparison plots
"""

import os, sys, json, argparse, time
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data.datasets import load_splits, partition_for_teachers, get_label_names
from src.teachers.llm_teacher import create_teachers, collect_votes
from src.aggregation.dp import dp_aggregate_batch
from src.aggregation.pac import pac_aggregate_batch
from src.evaluation.metrics import accuracy, summarize_sigma, margin_vs_sigma


def run(cfg_path: str):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # ---- Output directory ----
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(cfg["output"]["results_dir"], ts)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Results saved to: {out_dir}")

    # ---- Data ----
    print("\n[1/4] Loading dataset...")
    train, test = load_splits(
        cfg["dataset"]["name"],
        num_test=cfg["dataset"]["num_test"],
        seed=cfg["dataset"]["seed"],
    )
    label_names = get_label_names(cfg["dataset"]["name"])
    num_classes = len(label_names)
    true_labels = [ex["label"] for ex in test]

    partitions = partition_for_teachers(
        train,
        num_teachers=cfg["teachers"]["num_teachers"],
        shots_per_teacher=cfg["teachers"]["shots_per_teacher"],
        seed=cfg["dataset"]["seed"],
    )

    # ---- Teacher inference ----
    print(f"\n[2/4] Creating {cfg['teachers']['num_teachers']} teachers and collecting votes...")
    teachers = create_teachers(
        partitions,
        label_names,
        cfg["dataset"]["name"],
        model_name=cfg["model"]["name"],
        device=cfg["model"]["device"],
        max_new_tokens=cfg["model"]["max_new_tokens"],
    )
    votes = collect_votes(teachers, test, num_classes)  # (N, C)

    if cfg["output"]["save_votes"]:
        np.save(os.path.join(out_dir, "votes.npy"), votes)
        print(f"  Votes saved: votes.npy  shape={votes.shape}")

    # ---- DP aggregation ----
    print("\n[3/4] DP aggregation (baseline)...")
    dp_preds, dp_sigma = dp_aggregate_batch(
        votes,
        epsilon=cfg["dp"]["epsilon"],
        delta=cfg["dp"]["delta"],
    )
    dp_acc = accuracy(dp_preds, true_labels)
    dp_results = {
        "accuracy": dp_acc,
        "sigma": dp_sigma,
        "epsilon": cfg["dp"]["epsilon"],
        "delta": cfg["dp"]["delta"],
    }
    print(f"  DP accuracy: {dp_acc:.3f}  σ={dp_sigma:.3f}")

    # ---- PAC aggregation ----
    print("\n[3/4] PAC aggregation (proposed method)...")
    pac_preds, pac_sigmas = pac_aggregate_batch(
        votes,
        eta=cfg["pac"]["eta"],
        mc_samples=cfg["pac"]["mc_samples"],
        sigma_low=cfg["pac"]["sigma_search"]["low"],
        sigma_high=cfg["pac"]["sigma_search"]["high"],
        tol=cfg["pac"]["sigma_search"]["tol"],
    )
    pac_acc = accuracy(pac_preds, true_labels)
    sigma_stats = summarize_sigma(pac_sigmas)
    margin_stats = margin_vs_sigma(votes, pac_sigmas)

    pac_results = {
        "accuracy": pac_acc,
        "eta": cfg["pac"]["eta"],
        "delta": cfg["pac"]["delta"],
        "sigma_stats": sigma_stats,
        "margin_sigma_correlation": margin_stats["correlation"],
    }
    print(f"  PAC accuracy: {pac_acc:.3f}  σ_mean={sigma_stats['mean']:.3f}  σ_median={sigma_stats['median']:.3f}")
    print(f"  Margin-noise correlation: {margin_stats['correlation']:.3f}  (negative = larger margin, smaller noise ✓)")

    # ---- Save results ----
    with open(os.path.join(out_dir, "dp_results.json"), "w") as f:
        json.dump(dp_results, f, indent=2)
    with open(os.path.join(out_dir, "pac_results.json"), "w") as f:
        json.dump(pac_results, f, indent=2)

    print(f"\n[4/4] Generating comparison plots...")
    _plot_comparison(votes, dp_sigma, pac_sigmas, dp_acc, pac_acc, out_dir)

    # ---- Summary ----
    print("\n" + "="*50)
    print("Experiment finished. Key results:")
    print(f"  Accuracy: DP={dp_acc:.3f}  PAC={pac_acc:.3f}  (diff={pac_acc-dp_acc:+.3f})")
    print(f"  Mean noise: DP σ={dp_sigma:.3f}  PAC σ_mean={sigma_stats['mean']:.3f}  (PAC{'<' if sigma_stats['mean']<dp_sigma else '>'}DP ✓)")
    print(f"  Results directory: {out_dir}")
    print("="*50)


def _plot_comparison(votes, dp_sigma, pac_sigmas, dp_acc, pac_acc, out_dir):
    import matplotlib.pyplot as plt
    from src.aggregation.pac import vote_margin

    margins = [vote_margin(v) for v in votes]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Plot 1: σ* distribution
    axes[0].hist(pac_sigmas, bins=20, color="steelblue", alpha=0.7, label="PAC σ*")
    axes[0].axvline(dp_sigma, color="red", linestyle="--", label=f"DP σ={dp_sigma:.2f}")
    axes[0].set_title("Noise distribution")
    axes[0].set_xlabel("σ")
    axes[0].set_ylabel("Number of samples")
    axes[0].legend()

    # Plot 2: margin vs σ*
    axes[1].scatter(margins, pac_sigmas, alpha=0.4, s=10, color="steelblue")
    axes[1].axhline(dp_sigma, color="red", linestyle="--", label=f"DP σ={dp_sigma:.2f}")
    axes[1].set_title("Vote margin vs PAC noise")
    axes[1].set_xlabel("Vote margin")
    axes[1].set_ylabel("σ*")
    axes[1].legend()

    # Plot 3: accuracy comparison
    axes[2].bar(["DP", "PAC"], [dp_acc, pac_acc], color=["tomato", "steelblue"])
    axes[2].set_ylim(max(0, min(dp_acc, pac_acc) - 0.1), 1.0)
    axes[2].set_title("Accuracy comparison")
    axes[2].set_ylabel("Accuracy")
    for i, v in enumerate([dp_acc, pac_acc]):
        axes[2].text(i, v + 0.005, f"{v:.3f}", ha="center")

    plt.tight_layout()
    path = os.path.join(out_dir, "comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Plot saved: comparison.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()
    run(args.config)

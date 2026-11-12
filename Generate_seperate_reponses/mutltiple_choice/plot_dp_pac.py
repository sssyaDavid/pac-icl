#!/usr/bin/env python3
"""Plot DP vs PAC accuracy for SST-2 and AG News.

Run from this directory:

    python plot_dp_pac.py

The script reads the latest DP record (keys eps_*) and the latest PAC record
(keys rho_*) written by:

    python run_ollama.py prompting_configs/ollama-pate-sst2-dp.yaml
    python run_ollama.py prompting_configs/ollama-pate-sst2-pac.yaml
    python run_ollama.py prompting_configs/ollama-pate-agnews-dp.yaml
    python run_ollama.py prompting_configs/ollama-pate-agnews-pac.yaml

Budgets are paired by rho = epsilon^2 / 2, so (1, 0.5), (3, 4.5), (8, 32).
Figures are written to figures/sst2_dp_vs_pac.png and figures/agnews_dp_vs_pac.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent

DATASETS = [
    {
        "title": "SST-2",
        "slug": "sst2",
        "dp_dirs": ["outputs/ollama-pate-sst2-dp", "outputs/ollama-pate-sst2"],
        "pac_dirs": ["outputs/ollama-pate-sst2-pac", "outputs/ollama-pate-sst2"],
    },
    {
        "title": "AG News",
        "slug": "agnews",
        "dp_dirs": ["outputs/ollama-pate-agnews-dp"],
        "pac_dirs": ["outputs/ollama-pate-agnews-pac"],
    },
]


def _load_records(path: Path) -> list[dict]:
    records = []
    if path.suffix == ".jsonl":
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
    else:
        records.append(json.loads(path.read_text()))
    return records


def _result_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    files = sorted(directory.rglob("eval.jsonl"))
    if files:
        return files
    return sorted(directory.rglob("eval.json"))


def _kind(record: dict) -> str | None:
    keys = record.keys()
    has_eps = any(key.startswith("eps_") for key in keys)
    has_rho = any(key.startswith("rho_") for key in keys)
    if has_eps and not has_rho:
        return "dp"
    if has_rho and not has_eps:
        return "pac"
    return None


def latest_record(directories: list[str], kind: str) -> tuple[dict, Path]:
    for relative in directories:
        directory = ROOT / relative
        for path in _result_files(directory):
            matches = [record for record in _load_records(path) if _kind(record) == kind]
            if matches:
                return matches[-1], path
    searched = ", ".join(directories)
    raise FileNotFoundError(f"No {kind.upper()} result under {searched}")


def _budget_points(record: dict, prefix: str) -> list[tuple[float, float]]:
    points = []
    for key, value in record.items():
        if key.startswith(prefix):
            points.append((float(key[len(prefix):]), float(value)))
    return sorted(points)


def align_budgets(dp: dict, pac: dict) -> list[dict]:
    """Pair DP epsilon with PAC rho when rho = epsilon^2 / 2."""
    dp_points = _budget_points(dp, "eps_")
    pac_points = _budget_points(pac, "rho_")
    used_rho = set()
    rows = []
    for epsilon, dp_acc in dp_points:
        target = epsilon ** 2 / 2.0
        match = None
        for index, (rho, pac_acc) in enumerate(pac_points):
            if index not in used_rho and abs(rho - target) < 1e-6:
                match = (index, rho, pac_acc)
                break
        if match is None:
            rows.append({"epsilon": epsilon, "rho": None, "dp": dp_acc, "pac": None})
            continue
        index, rho, pac_acc = match
        used_rho.add(index)
        rows.append({"epsilon": epsilon, "rho": rho, "dp": dp_acc, "pac": pac_acc})
    for index, (rho, pac_acc) in enumerate(pac_points):
        if index not in used_rho:
            rows.append({"epsilon": None, "rho": rho, "dp": None, "pac": pac_acc})
    return rows


def _tick_label(row: dict) -> str:
    if row["epsilon"] is not None and row["rho"] is not None:
        rho = row["rho"]
        rho_text = str(int(rho)) if rho.is_integer() else str(rho)
        eps_text = str(int(row["epsilon"])) if row["epsilon"].is_integer() else str(row["epsilon"])
        return f"ε = {eps_text}\nρ = {rho_text}"
    if row["epsilon"] is not None:
        return f"ε = {row['epsilon']}"
    return f"ρ = {row['rho']}"


def plot_dataset(title: str, dp: dict, pac: dict, out_path: Path) -> None:
    rows = align_budgets(dp, pac)
    labels = [_tick_label(row) for row in rows]
    dp_acc = np.array([row["dp"] if row["dp"] is not None else np.nan for row in rows])
    pac_acc = np.array([row["pac"] if row["pac"] is not None else np.nan for row in rows])
    x = np.arange(len(rows))
    nonprivate_dp = float(dp["non_private"])
    nonprivate_pac = float(pac["non_private"])

    values = [value for value in (*dp_acc, *pac_acc, nonprivate_dp, nonprivate_pac) if not np.isnan(value)]
    low, high = min(values), max(values)
    span = max(high - low, 0.08)
    pad = span * 0.55
    y_min = max(0.0, low - pad * 0.85)
    y_max = min(1.0, high + pad * 0.75)

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.axhline(np.mean([nonprivate_dp, nonprivate_pac]), color="#b0b6be", linestyle="--", linewidth=1, zorder=1, label="Non-private")
    ax.plot(x, dp_acc, marker="o", markersize=7, markerfacecolor="white", markeredgewidth=2, linewidth=2.2, color="#d06a45", label="DP", zorder=3)
    ax.plot(x, pac_acc, marker="o", markersize=7, markerfacecolor="white", markeredgewidth=2, linewidth=2.2, color="#2f6f9f", label="PAC", zorder=3)

    for index, (dp_value, pac_value) in enumerate(zip(dp_acc, pac_acc)):
        separated = np.isnan(dp_value) or np.isnan(pac_value) or abs(dp_value - pac_value) >= 0.03
        if not separated:
            continue
        if not np.isnan(dp_value) and (np.isnan(pac_value) or dp_value < pac_value):
            ax.annotate(f"{dp_value:.1%}", (index, dp_value), textcoords="offset points", xytext=(0, -12), ha="center", va="top", fontsize=9, color="#d06a45")
        if not np.isnan(pac_value) and (np.isnan(dp_value) or pac_value < dp_value):
            ax.annotate(f"{pac_value:.1%}", (index, pac_value), textcoords="offset points", xytext=(0, -12), ha="center", va="top", fontsize=9, color="#2f6f9f")

    ax.set_title(title, loc="left", pad=8)
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x, labels)
    ax.set_xlim(-0.35, len(rows) - 0.65)
    ax.set_ylim(y_min, y_max)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _pos: f"{value:.0%}"))
    ax.yaxis.grid(True, linestyle="-", linewidth=0.6, color="#eceff3")
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#c5c9d0")
    ax.spines["bottom"].set_color("#c5c9d0")
    ax.tick_params(colors="#4a4f57", length=0)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    out_dir = ROOT / "figures"
    for spec in DATASETS:
        dp, dp_path = latest_record(spec["dp_dirs"], "dp")
        pac, pac_path = latest_record(spec["pac_dirs"], "pac")
        out_path = out_dir / f"{spec['slug']}_dp_vs_pac.png"
        plot_dataset(spec["title"], dp, pac, out_path)
        print(f"{spec['title']}")
        print(f"  DP  {dp_path.relative_to(ROOT)}")
        print(f"  PAC {pac_path.relative_to(ROOT)}")
        print(f"  saved {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

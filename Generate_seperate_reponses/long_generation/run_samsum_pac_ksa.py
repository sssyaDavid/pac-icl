"""
PAC-KSA on SAMSum using Ollama (LLaMA 3.1:8b).

Usage:
    python run_samsum_pac_ksa.py                          # PAC-KSA lambda=1
    python run_samsum_pac_ksa.py --lambda_adapt 0.0       # DP-KSA baseline
    python run_samsum_pac_ksa.py --eps_list 3 8           # only eps=3 and eps=8
    python run_samsum_pac_ksa.py --lambda_adapt 0.5 --output_dir ./outputs/ablation
"""
import argparse
import json
import os
import random
import numpy as np
import requests
from datasets import load_dataset
from evaluate import load as load_metric
from tqdm import tqdm

from pac_ksa_aggregator import pac_ksa_aggregate, build_ksa_prompt

# ── reproducibility ────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

# ── CLI args ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--lambda_adapt", type=float, default=1.0,
                    help="0=DP-KSA baseline, >0=PAC-KSA adaptive")
parser.add_argument("--eps_list", type=float, nargs="+", default=[1, 3, 8])
parser.add_argument("--ensemble", type=int, default=10)
parser.add_argument("--ds_size", type=int, default=100)
parser.add_argument("--ice_num", type=int, default=4)
parser.add_argument("--kw_size", type=int, default=10)
parser.add_argument("--output_dir", type=str, default=None)
args_cli = parser.parse_args()

# ── config ─────────────────────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434/api/generate"
MODEL_NAME   = "llama3.1:8b"
ENSEMBLE     = args_cli.ensemble
DS_SIZE      = args_cli.ds_size
ICE_NUM      = args_cli.ice_num
EPS_LIST     = args_cli.eps_list
LAMBDA_ADAPT = args_cli.lambda_adapt
KW_SIZE      = args_cli.kw_size

if args_cli.output_dir:
    OUTPUT_DIR = args_cli.output_dir
elif LAMBDA_ADAPT == 0.0:
    OUTPUT_DIR = "./outputs/dp_ksa_samsum"
else:
    OUTPUT_DIR = f"./outputs/pac_ksa_samsum_lambda{LAMBDA_ADAPT}"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def ollama_generate(prompt, max_tokens=128):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": 0}
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["response"].strip()


def make_teacher_prompt(ice_examples, test_dialogue):
    """Build few-shot prompt for one teacher."""
    prompt = ""
    for ex in ice_examples:
        prompt += f'Dialogue:\n"{ex["dialogue"]}"\nSummarize the above dialogue: {ex["summary"]}\n\n'
    prompt += f'Dialogue:\n"{test_dialogue}"\nSummarize the above dialogue:'
    return prompt


def run_experiment(eps, train_data, test_data, rouge_metric):
    # Split training data into ENSEMBLE disjoint shards
    indices = list(range(len(train_data)))
    random.shuffle(indices)
    shard_size = len(indices) // ENSEMBLE
    shards = [indices[i * shard_size:(i + 1) * shard_size] for i in range(ENSEMBLE)]

    all_preds, all_refs = [], []
    kw_counts, agreement_scores = [], []

    for sample in tqdm(test_data, desc=f"eps={eps}"):
        dialogue = sample["dialogue"]
        reference = sample["summary"]

        # Step 1: each teacher generates a summary
        teacher_outputs = []
        for shard_idx in shards:
            ice = [train_data[j] for j in random.sample(shard_idx, min(ICE_NUM, len(shard_idx)))]
            prompt = make_teacher_prompt(ice, dialogue)
            out = ollama_generate(prompt)
            teacher_outputs.append(out)

        # Step 2: PAC-KSA aggregation
        keywords, agreement = pac_ksa_aggregate(
            teacher_outputs, eps=eps,
            lambda_adapt=LAMBDA_ADAPT, kw_size=KW_SIZE
        )
        kw_counts.append(len(keywords) if keywords else 0)
        agreement_scores.append(agreement)

        # Step 3: final generation with keywords
        final_prompt = build_ksa_prompt(dialogue, keywords)
        final_summary = ollama_generate(final_prompt)

        all_preds.append(final_summary)
        all_refs.append(reference)

    # Evaluate ROUGE
    scores = rouge_metric.compute(predictions=all_preds, references=all_refs)
    result = {
        "eps": eps,
        "lambda_adapt": LAMBDA_ADAPT,
        "rouge1": round(scores["rouge1"], 4),
        "rouge2": round(scores["rouge2"], 4),
        "rougeL": round(scores["rougeL"], 4),
        "avg_keywords": round(float(np.mean(kw_counts)), 2),
        "avg_agreement": round(float(np.mean(agreement_scores)), 4),
    }
    return result, all_preds


def main():
    print(f"Config: lambda_adapt={LAMBDA_ADAPT}, eps_list={EPS_LIST}, "
          f"ensemble={ENSEMBLE}, ds_size={DS_SIZE}, output_dir={OUTPUT_DIR}")

    print("Loading SAMSum...")
    dataset = load_dataset("knkarthick/samsum", cache_dir="./data")
    train_data = list(dataset["train"])
    test_data  = list(dataset["test"])[:DS_SIZE]

    rouge_metric = load_metric("rouge")
    all_results = []

    for eps in EPS_LIST:
        result, preds = run_experiment(eps, train_data, test_data, rouge_metric)
        print(f"\n[eps={eps}] ROUGE-1={result['rouge1']}  ROUGE-2={result['rouge2']}  ROUGE-L={result['rougeL']}")
        print(f"         avg_keywords={result['avg_keywords']}  avg_agreement={result['avg_agreement']}")
        all_results.append(result)

        pred_path = os.path.join(OUTPUT_DIR, f"preds_eps{eps}.json")
        with open(pred_path, "w") as f:
            json.dump(preds, f, indent=2)

    result_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {result_path}")


if __name__ == "__main__":
    main()
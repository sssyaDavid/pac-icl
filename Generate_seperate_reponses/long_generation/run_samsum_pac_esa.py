"""
PAC-ESA on SAMSum using Ollama (LLaMA 3.1:8b).
Usage:
    python run_samsum_pac_esa.py                      # PAC-ESA lambda=1
    python run_samsum_pac_esa.py --lambda_adapt 0.0   # DP-ESA baseline
    python run_samsum_pac_esa.py --eps_list 3 8
"""
import argparse, json, os, random
import numpy as np
import requests
from datasets import load_dataset
from evaluate import load as load_metric
from tqdm import tqdm
from pac_esa_aggregator import pac_esa_aggregate, ollama_embed, find_best_candidate

parser = argparse.ArgumentParser()
parser.add_argument("--lambda_adapt", type=float, default=1.0)
parser.add_argument("--eps_list", type=float, nargs="+", default=[1, 3, 8])
parser.add_argument("--delta", type=float, default=1e-5)
parser.add_argument("--ensemble", type=int, default=10)
parser.add_argument("--ds_size", type=int, default=100)
parser.add_argument("--ice_num", type=int, default=4)
parser.add_argument("--n_candidates", type=int, default=10)
parser.add_argument("--candidate_temp", type=float, default=0.7)
parser.add_argument("--output_dir", type=str, default=None)
args_cli = parser.parse_args()

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1:8b"
ENSEMBLE = args_cli.ensemble
DS_SIZE = args_cli.ds_size
ICE_NUM = args_cli.ice_num
EPS_LIST = args_cli.eps_list
DELTA = args_cli.delta
LAMBDA_ADAPT = args_cli.lambda_adapt
N_CANDIDATES = args_cli.n_candidates
CANDIDATE_TEMP = args_cli.candidate_temp

if args_cli.output_dir:
    OUTPUT_DIR = args_cli.output_dir
elif LAMBDA_ADAPT == 0.0:
    OUTPUT_DIR = "./outputs/dp_esa_samsum"
else:
    OUTPUT_DIR = f"./outputs/pac_esa_samsum_lambda{LAMBDA_ADAPT}"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def ollama_generate(prompt, max_tokens=128, temperature=0.0):
    payload = {
        "model": MODEL_NAME, "prompt": prompt, "stream": False,
        "options": {"num_predict": max_tokens, "temperature": temperature},
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["response"].strip()


def make_teacher_prompt(ice_examples, test_dialogue):
    prompt = ""
    for ex in ice_examples:
        prompt += f'Dialogue:\n"{ex["dialogue"]}"\nSummarize the above dialogue: {ex["summary"]}\n\n'
    prompt += f'Dialogue:\n"{test_dialogue}"\nSummarize the above dialogue:'
    return prompt


def make_zero_shot_prompt(dialogue):
    return (
        f"{dialogue}\n"
        f"Summarize the above dialogue:\n"
        f"The summary is:"
    )


def run_experiment(eps, train_data, test_data, rouge_metric):
    random.seed(42)
    np.random.seed(42)

    indices = list(range(len(train_data)))
    random.shuffle(indices)
    shard_size = len(indices) // ENSEMBLE
    shards = [indices[i * shard_size:(i + 1) * shard_size] for i in range(ENSEMBLE)]

    all_preds, all_refs = [], []
    agreement_scores = []

    for sample in tqdm(test_data, desc=f"eps={eps}"):
        dialogue = sample["dialogue"]
        reference = sample["summary"]

        # Step 1: K teacher outputs
        teacher_outputs = []
        for shard_idx in shards:
            ice = [train_data[j] for j in random.sample(shard_idx, min(ICE_NUM, len(shard_idx)))]
            prompt = make_teacher_prompt(ice, dialogue)
            teacher_outputs.append(ollama_generate(prompt))

        # Step 2: PAC-ESA -> noisy mean embedding
        noisy_mean, confidence = pac_esa_aggregate(
            teacher_outputs, eps=eps, delta=DELTA, lambda_adapt=LAMBDA_ADAPT
        )
        agreement_scores.append(confidence)

        # Step 3: generate zero-shot candidates
        zs_prompt = make_zero_shot_prompt(dialogue)
        candidates = [ollama_generate(zs_prompt, temperature=0.0)]
        for _ in range(N_CANDIDATES - 1):
            candidates.append(ollama_generate(zs_prompt, temperature=CANDIDATE_TEMP))

        # Step 4: embed candidates, pick closest to noisy mean
        cand_embeddings = [ollama_embed(c) for c in candidates]
        if noisy_mean is not None:
            final_summary = find_best_candidate(candidates, cand_embeddings, noisy_mean)
        else:
            final_summary = candidates[0]

        all_preds.append(final_summary)
        all_refs.append(reference)

    scores = rouge_metric.compute(predictions=all_preds, references=all_refs)
    result = {
        "eps": eps, "lambda_adapt": LAMBDA_ADAPT, "delta": DELTA,
        "rouge1": round(scores["rouge1"], 4),
        "rouge2": round(scores["rouge2"], 4),
        "rougeL": round(scores["rougeL"], 4),
        "avg_agreement": round(float(np.mean(agreement_scores)), 4),
    }
    return result, all_preds


def main():
    print(f"Config: lambda_adapt={LAMBDA_ADAPT}, eps_list={EPS_LIST}, n_candidates={N_CANDIDATES}")
    dataset = load_dataset("knkarthick/samsum", cache_dir="./data")
    train_data = list(dataset["train"])
    test_data = list(dataset["test"])[:DS_SIZE]
    rouge_metric = load_metric("rouge")

    all_results = []
    for eps in EPS_LIST:
        result, preds = run_experiment(eps, train_data, test_data, rouge_metric)
        print(f"\n[eps={eps}] R1={result['rouge1']} R2={result['rouge2']} RL={result['rougeL']} agreement={result['avg_agreement']}")
        all_results.append(result)
        with open(os.path.join(OUTPUT_DIR, f"preds_eps{eps}.json"), "w") as f:
            json.dump(preds, f, indent=2)

    result_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {result_path}")


if __name__ == "__main__":
    main()
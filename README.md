# PAC-Private ICL Aggregation

Run DP-ICL experiments with a locally deployed LLaMA, comparing DP (RNM + Gaussian) with PAC Privacy adaptive aggregation.

## Directory layout

```
pac-dp-icl/
├── configs/
│   └── experiment.yaml      # All hyperparameters (edit the config here)
├── src/
│   ├── data/datasets.py     # Dataset loading + teacher splits
│   ├── teachers/
│   │   └── llm_teacher.py   # LLaMA ICL inference (HF and Ollama backends)
│   ├── aggregation/
│   │   ├── dp.py            # DP baseline: RNM + Gaussian
│   │   └── pac.py           # PAC: Monte Carlo adaptive noise
│   ├── privacy/
│   │   └── mi_estimator.py  # Mutual information Monte Carlo estimator
│   └── evaluation/
│       └── metrics.py       # Accuracy and noise statistics
└── experiments/
    ├── quick_test.py        # Fast check without an LLM (run this first)
    └── run_experiment.py    # Full experiment
```

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Check the PAC core logic (no LLM, about 2 minutes)

```bash
python experiments/quick_test.py
```

Expected output:
- Noise used by PAC on decisive votes is smaller than the fixed DP noise
- Vote margin is negatively correlated with σ* (larger margin → smaller noise)

### 3. Configure the LLM backend

**Option A: HuggingFace Transformers (requires downloading the model)**

```yaml
# configs/experiment.yaml
model:
  name: "meta-llama/Llama-3.2-1B-Instruct"
  device: "cpu"   # or "cuda" / "mps"
```

Request Meta LLaMA access and log in to Hugging Face first:
```bash
huggingface-cli login
```

**Option B: Ollama (lower VRAM use, recommended for local tests)**

```bash
# Install Ollama first: https://ollama.ai
ollama pull llama3.2:1b
```

```yaml
# configs/experiment.yaml
model:
  name: "ollama:llama3.2:1b"
```

### 4. Run the full experiment

```bash
python experiments/run_experiment.py --config configs/experiment.yaml
```

Results are saved under `results/<timestamp>/` and include:
- `votes.npy`: raw vote histograms
- `dp_results.json`: DP aggregation results
- `pac_results.json`: PAC aggregation results
- `comparison.png`: three comparison plots

## Core algorithms

### DP aggregation (baseline)

```
σ = √(2 ln(1.25/δ)) / ε    [fixed, worst case]
ṽᵢ = vᵢ + N(0, σ²)
Output: argmax ṽ
Privacy guarantee: (ε, δ)-DP
```

### PAC aggregation (proposed method)

```
Step 1: Estimate Î = (1/T) Σ ln[ P(M|v) / P(M|v') ]   [Monte Carlo]
Step 2: σ* = min{σ : Î ≤ η}                             [binary search]
Step 3: ṽᵢ = vᵢ + N(0, σ*²),  output: argmax ṽ
Privacy guarantee: (η, δ)-PAC Privacy
```

**Key advantage**: when the vote margin is large (for example [8, 2]), σ* << DP σ and accuracy is higher.

## Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `teachers.num_teachers` | Number of teachers | 10 |
| `teachers.shots_per_teacher` | ICL examples per teacher | 4 |
| `dp.epsilon` | DP privacy budget ε | 1.0 |
| `pac.eta` | PAC target mutual-information upper bound (nats) | 0.5 |
| `pac.mc_samples` | Number of Monte Carlo samples | 500 |

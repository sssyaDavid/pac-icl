# Run log

## 2026-11-09 21:07–21:43 — AG News, DP

Command:

```bash
python run_ollama.py prompting_configs/ollama-pate-agnews-dp.yaml
```

| Setting | Value |
| --- | --- |
| Task | AG News (`fancyzhx/ag_news`) |
| Model | Ollama `llama3.1:8b` |
| Strategy | PATE, 4-shot |
| Teachers | 10 |
| Train / test | 8000 / 1000 |
| Seed | 42 |
| Privacy | DP, δ = 1e-5, ε ∈ {1, 3, 8} |

| Setting | Accuracy |
| --- | --- |
| Non-private (ε = ∞) | 26.0% |
| DP (ε = 1) | 26.1% |
| DP (ε = 3) | 25.9% |
| DP (ε = 8) | 26.0% |

Raw result: `outputs/ollama-pate-agnews-dp/4-shot/eval.json` and `outputs/ollama-pate-agnews-dp/eval.jsonl`.

```json
{"non_private": 0.26, "eps_1": 0.261, "eps_3": 0.259, "eps_8": 0.26}
```

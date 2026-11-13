"""
PAC-ESA aggregator: embed K teacher outputs in Ollama embedding space,
compute adaptive Gaussian noise based on semantic clustering confidence.
"""
import numpy as np
import requests
from typing import List, Tuple, Optional

OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"


def ollama_embed(text: str, model: str = "nomic-embed-text") -> np.ndarray:
    r = requests.post(
        OLLAMA_EMBED_URL,
        json={"model": model, "prompt": text},
        timeout=60,
    )
    r.raise_for_status()
    emb = np.array(r.json()["embedding"], dtype=np.float64)
    norm = np.linalg.norm(emb)
    if norm > 1e-10:
        emb = emb / norm
    return emb


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def pac_esa_aggregate(
    teacher_outputs: List[str],
    eps: float,
    delta: float = 1e-5,
    lambda_adapt: float = 1.0,
    model: str = "nomic-embed-text",
) -> Tuple[Optional[np.ndarray], float]:
    """
    1. Embed K teacher outputs (unit-normalized).
    2. Mean embedding as centroid.
    3. Confidence = mean cosine similarity to centroid (high = teachers agree).
    4. Adaptive Gaussian noise:
         sigma = (2/K * sqrt(2*ln(1.25/delta)) / eps) / (1 + lambda * confidence)
       lambda=0 -> DP-ESA baseline; lambda>0 -> PAC-ESA.
    5. Return (noisy mean embedding, confidence).
    """
    K = len(teacher_outputs)
    if K == 0:
        return None, 0.0

    embeddings = []
    for text in teacher_outputs:
        emb = ollama_embed(text, model)
        embeddings.append(emb)
    embeddings = np.array(embeddings)  # K x D

    mean_emb = np.mean(embeddings, axis=0)

    sims = [cosine_sim(emb, mean_emb) for emb in embeddings]
    confidence = float(np.clip(np.mean(sims), 0.0, 1.0))

    # L2 sensitivity of mean = 2/K (unit sphere)
    sensitivity = 2.0 / K
    sigma_base = sensitivity * np.sqrt(2.0 * np.log(1.25 / delta)) / eps
    sigma = sigma_base / (1.0 + lambda_adapt * confidence)

    noise = np.random.normal(0.0, sigma, mean_emb.shape)
    noisy_mean = mean_emb + noise

    return noisy_mean, confidence


def find_best_candidate(
    candidates: List[str],
    candidate_embeddings: List[np.ndarray],
    target_embedding: np.ndarray,
) -> str:
    if not candidates:
        return ""
    sims = [cosine_sim(emb, target_embedding) for emb in candidate_embeddings]
    return candidates[int(np.argmax(sims))]
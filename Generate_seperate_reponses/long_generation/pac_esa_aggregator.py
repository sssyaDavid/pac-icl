"""
PAC-ESA aggregator: embed K teacher outputs in Ollama embedding space,
compute adaptive Gaussian noise based on semantic clustering confidence.

Supported confidence modes:
  cosine_centroid  - mean cosine similarity to centroid (original, tends to saturate)
  pairwise_entropy - softmax entropy of pairwise cosine similarities;
                     captures spread of the K embeddings more sensitively
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


def _pairwise_entropy_confidence(embeddings: np.ndarray) -> float:
    """
    Compute confidence from pairwise cosine similarity entropy.

    Steps:
      1. Compute K x K cosine similarity matrix (embeddings are unit-normalized,
         so dot product = cosine similarity).
      2. For each teacher i, treat its similarities to all other teachers as
         a raw score vector, apply softmax to get a probability distribution.
      3. Compute Shannon entropy of each distribution; average across K teachers.
      4. Normalize by log(K-1) so entropy is in [0, 1].
      5. confidence = 1 - normalized_entropy
         (all teachers identical -> entropy=0 -> confidence=1;
          all teachers maximally different -> entropy=1 -> confidence=0)

    Why this is better than cosine_centroid:
      cosine_centroid measures each teacher's similarity to the mean, which
      saturates when embeddings cluster near the same region of the unit sphere.
      Pairwise entropy amplifies small inter-teacher differences, giving a
      more discriminative signal between easy (uniform) and hard (diverse) samples.
    """
    K = len(embeddings)
    if K <= 1:
        return 1.0

    # K x K similarity matrix; diagonal = 1.0 (self-similarity)
    sim_matrix = embeddings @ embeddings.T  # already unit-normalized

    entropies = []
    for i in range(K):
        # Exclude self-similarity (diagonal)
        scores = np.concatenate([sim_matrix[i, :i], sim_matrix[i, i+1:]])  # K-1 values

        # Softmax to convert scores to a probability distribution
        scores = scores - scores.max()  # numerical stability
        exp_scores = np.exp(scores)
        probs = exp_scores / (exp_scores.sum() + 1e-12)

        # Shannon entropy
        h = -np.sum(probs * np.log(probs + 1e-12))
        entropies.append(h)

    mean_entropy = float(np.mean(entropies))
    max_entropy = np.log(K - 1) if K > 2 else 1.0  # uniform distribution over K-1 others
    normalized_entropy = mean_entropy / (max_entropy + 1e-12)

    confidence = float(np.clip(1.0 - normalized_entropy, 0.0, 1.0))
    return confidence


def pac_esa_aggregate(
    teacher_outputs: List[str],
    eps: float,
    delta: float = 1e-5,
    lambda_adapt: float = 1.0,
    model: str = "nomic-embed-text",
    confidence_mode: str = "cosine_centroid",
) -> Tuple[Optional[np.ndarray], float]:
    """
    1. Embed K teacher outputs (unit-normalized).
    2. Mean embedding as centroid.
    3. Confidence computed by confidence_mode:
         cosine_centroid  - mean cosine similarity to centroid (original)
         pairwise_entropy - 1 - normalized softmax entropy of pairwise sims
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

    # Confidence
    if confidence_mode == "pairwise_entropy":
        confidence = _pairwise_entropy_confidence(embeddings)
    else:  # cosine_centroid (original)
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
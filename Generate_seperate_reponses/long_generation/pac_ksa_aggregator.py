"""
PAC-KSA aggregator: aggregate K teacher text outputs
using adaptive PAC noise.
Confidence = 1 - normalized entropy of keyword frequency distribution.
Low entropy (votes concentrated) -> high confidence -> less noise.
"""
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


def pac_ksa_aggregate(teacher_outputs, eps, lambda_adapt=1.0, kw_size=10):
    stopword_set = set(stopwords.words('english'))
    all_tokens = {}
    for sentence in teacher_outputs:
        tokens = nltk.word_tokenize(sentence)
        onegrams = set(ngrams(tokens, 1))
        for token in onegrams:
            word = token[0]
            if (not all(ch in string.punctuation for ch in word)
                    and word not in stopword_set
                    and len(word) > 1):
                all_tokens[word] = all_tokens.get(word, 0) + 1

    if not all_tokens:
        return None, 0.0

    sorted_tokens = sorted(all_tokens.items(), key=lambda x: x[1], reverse=True)

    if len(sorted_tokens) < 2:
        return [t[0] for t in sorted_tokens[:kw_size]], 1.0

    # --- Entropy-based confidence ---
    # Low entropy = votes concentrated on few keywords = teachers agree = high confidence
    freqs_arr = np.array([v for _, v in all_tokens.items()], dtype=float)
    freqs_norm = freqs_arr / freqs_arr.sum()
    entropy = -np.sum(freqs_norm * np.log(freqs_norm + 1e-10))
    max_entropy = np.log(max(len(freqs_arr), 2))
    confidence = float(1.0 - entropy / max_entropy)   # in [0, 1]

    # --- Gap list (same as before) ---
    limit = min(len(sorted_tokens) - 1, 30)
    gap_lst = []
    for k in range(limit):
        gap = sorted_tokens[k][1] - sorted_tokens[k + 1][1]
        gap_lst.append(float(gap))
    gap_lst.append(float(sorted_tokens[limit][1]))
    gap_lst = np.array(gap_lst)

    # --- Adaptive noise ---
    base_scale  = 2.0 / eps
    noise_scale = base_scale / (1.0 + lambda_adapt * confidence)
    noisy_gaps  = gap_lst + np.random.gumbel(0, noise_scale, len(gap_lst))
    kstar       = int(np.argmax(noisy_gaps))

    keywords = [t[0] for t in sorted_tokens[: kstar + 1]]
    keywords = keywords[:kw_size]
    return keywords, confidence


def build_ksa_prompt(dialogue, keywords):
    if not keywords:
        return (
            f"{dialogue}\n"
            f"Summarize the above dialogue:\n"
            f"The summary is:"
        )
    kw_str = ", ".join(keywords)
    return (
        f"{dialogue}\n"
        f"Summarize the above dialogue with the following word suggestions "
        f"ranked by their frequency from high to low: [{kw_str}]\n"
        f"The summary is:"
    )
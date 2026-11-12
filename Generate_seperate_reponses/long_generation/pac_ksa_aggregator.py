"""
PAC-KSA aggregator: 把 K 个 teacher 的文本输出
用自适应 PAC 噪声聚合成关键词列表。
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
    """
    Args:
        teacher_outputs: list of K strings (each teacher's generated summary)
        eps: privacy budget
        lambda_adapt: adaptivity strength (0 = fixed noise, same as DP-ICL)
        kw_size: max number of keywords to return
    Returns:
        keywords: list of keyword strings (None if no keywords found)
        agreement: float, teacher agreement score (for logging)
    """
    stopword_set = set(stopwords.words('english'))

    # Step 1: build word frequency histogram across all teachers
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

    # Sort by frequency descending
    sorted_tokens = sorted(all_tokens.items(), key=lambda x: x[1], reverse=True)

    if len(sorted_tokens) < 2:
        return [t[0] for t in sorted_tokens[:kw_size]], 1.0

    # Step 2: compute gaps between consecutive frequencies
    limit = min(len(sorted_tokens) - 1, 30)
    gap_lst = []
    for k in range(limit):
        gap = sorted_tokens[k][1] - sorted_tokens[k + 1][1]
        gap_lst.append(float(gap))
    gap_lst.append(float(sorted_tokens[limit][1]))
    gap_lst = np.array(gap_lst)

    # Step 3: adaptive PAC noise
    # confidence = how much teachers agree on the top word (normalized)
    max_freq = float(sorted_tokens[0][1])
    K = float(len(teacher_outputs))
    confidence = max_freq / K  # 和分类任务完全一致：最高频词出现在几分之几的teacher里

    # Higher confidence → smaller noise scale
    base_scale = 2.0 / eps
    noise_scale = base_scale / (1.0 + lambda_adapt * confidence)
    noisy_gaps = gap_lst + np.random.gumbel(0, noise_scale, len(gap_lst))
    kstar = int(np.argmax(noisy_gaps))

    # Step 4: return top-kstar+1 keywords
    keywords = [t[0] for t in sorted_tokens[: kstar + 1]]
    keywords = keywords[:kw_size]

    return keywords, confidence


def build_ksa_prompt(dialogue, keywords):
    """
    Build the final generation prompt.
    If keywords is None/empty, fall back to zero-shot.
    """
    if not keywords:
        return (
            f"Summarize the above dialogue:\n{dialogue}\nThe summary is:"
        )
    kw_str = ", ".join(keywords)
    return (
        f"{dialogue}\n"
        f"Summarize the above dialogue with the following word suggestions "
        f"ranked by their frequency from high to low: [{kw_str}]\n"
        f"The summary is:"
    )
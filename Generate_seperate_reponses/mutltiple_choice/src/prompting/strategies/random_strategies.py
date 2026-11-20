import logging
import math
import random
from collections import defaultdict
from itertools import permutations

import numpy as np
from scipy.optimize import brentq

from prompting import BaseProcessor, GPT2Wrapper
from prompting.strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

from tqdm import tqdm


# ---------------------------------------------------------------------------
# PAC Privacy helpers  (PAC Privacy paper, Ghazi et al., Theorem 3)
# ---------------------------------------------------------------------------

def _mi_upper_bound(eigenvalues: np.ndarray, sigma: float) -> float:
    """
    Compute the mutual information upper bound from PAC Theorem 3.

    I(X; M(X) + N(0, sigma^2 I))  <=  0.5 * sum_j log(1 + lambda_j / sigma^2)

    Parameters
    ----------
    eigenvalues : positive eigenvalues of the output covariance matrix Sigma_out
    sigma       : noise standard deviation
    """
    pos = eigenvalues[eigenvalues > 1e-10]
    if len(pos) == 0:
        return 0.0
    return 0.5 * float(np.sum(np.log(1.0 + pos / (sigma ** 2))))


def calibrate_sigma_from_covariance(cov: np.ndarray, eta: float,
                                     sigma_min: float = 1e-3,
                                     sigma_max: float = 1e4) -> float:
    """
    Find the smallest sigma such that the PAC MI upper bound <= eta.

    Uses eigenvalue decomposition of the output covariance matrix and
    binary search (brentq) on the closed-form bound from Theorem 3.

    Parameters
    ----------
    cov       : (C, C) sample covariance of teacher votes for one query
    eta       : PAC privacy budget (mutual information bound)
    sigma_min : lower bound for binary search
    sigma_max : upper bound for binary search

    Returns
    -------
    sigma*  : minimum noise level satisfying the PAC guarantee
    """
    eigenvalues = np.linalg.eigvalsh(cov)          # sorted ascending
    eigenvalues = eigenvalues[eigenvalues > 1e-10]  # discard numerical zeros

    if len(eigenvalues) == 0:
        # All teachers agree perfectly; tiny noise is sufficient.
        return sigma_min

    # If even sigma_min already satisfies the budget, return it immediately.
    if _mi_upper_bound(eigenvalues, sigma_min) <= eta:
        return sigma_min

    # Binary search: find root of  MI_upper_bound(sigma) - eta = 0
    try:
        sigma_star = brentq(
            lambda s: _mi_upper_bound(eigenvalues, s) - eta,
            sigma_min, sigma_max,
            xtol=1e-6, maxiter=200
        )
    except ValueError:
        # MI upper bound is still > eta at sigma_max; clamp to sigma_max.
        logger.warning(
            "calibrate_sigma: MI bound still exceeds eta=%.4f at sigma_max=%.1f; "
            "clamping to sigma_max.", eta, sigma_max
        )
        sigma_star = sigma_max

    return float(sigma_star)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

class RandomStrategy(BaseStrategy):
    def run_strategy(self, proc: BaseProcessor, model: GPT2Wrapper, shot: int):
        logger.info(f"RandomStrategy - shot={shot}")
        train_indices = random.sample(range(len(proc.train_dataset)), k=shot)
        prompts, cali_prompts = proc.create_prompts(train_indices)
        outputs = model.complete_all(prompts, calibration_prompts=cali_prompts,
                                     file_name=self.output_dir)
        eval_result = proc.extract_predictions(outputs)
        self.write_result(eval_result, shot)
        return {"shot": shot, "train_indices": train_indices, "acc": eval_result["acc"]}


class PATEStrategy(BaseStrategy):
    """
    PATE-style private aggregation supporting two privacy modes:

    DP mode  : Gaussian RNM with fixed sigma derived from (epsilon, delta).
    PAC mode : Adaptive Gaussian noise whose sigma is calibrated per query
               from the eigenvalues of the teacher-vote covariance matrix,
               following PAC Privacy Theorem 3 (Ghazi et al.).

    Key data structure
    ------------------
    all_votes : ndarray of shape (K, n_test, n_classes)
        all_votes[i, j, c] = 1 if teacher i predicted class c for query j.
        This preserves per-teacher information needed to estimate Sigma_out(q).
    """

    # ------------------------------------------------------------------
    # DP helper
    # ------------------------------------------------------------------
    @staticmethod
    def _rnm_sigma(epsilon: float, delta: float = 1e-5) -> float:
        """
        Gaussian RNM noise level for DP.
        L2-sensitivity of a vote vector = sqrt(2).
        sigma = sqrt(2) * sqrt(2 * ln(1.25 / delta)) / epsilon
        """
        return math.sqrt(2) * math.sqrt(2 * math.log(1.25 / delta)) / epsilon

    # ------------------------------------------------------------------
    # Main strategy
    # ------------------------------------------------------------------
    def run_strategy(self, proc: BaseProcessor, model: GPT2Wrapper, shot: int):
        privacy_mode = self.conf.get("privacy", "dp").lower()
        logger.info(f"PATEStrategy - shot={shot}, privacy={privacy_mode}")

        K             = self.conf.ensemble          # number of teachers
        subsample_num = shot * K
        n_test        = len(proc.test_dataset)
        labels        = proc.labels
        n_classes     = len(labels)

        print(f"Train size: {len(proc.train_dataset)},  "
              f"Test size: {n_test},  Teachers: {K}")
        print(f"Privacy mode: {privacy_mode.upper()}")

        # ------------------------------------------------------------------
        # Step 1: assign training indices to each (query, teacher) pair
        # ------------------------------------------------------------------
        # icl_list[j][i] = list of training indices for query j, teacher i
        # Each query uses a fresh random seed so its K teacher subsets are
        # independent of other queries.
        icl_list = []
        for j in range(n_test):
            rng_q = random.Random(self.conf.seed + j)
            pool  = list(range(len(proc.train_dataset)))
            rng_q.shuffle(pool)
            selected = pool[:subsample_num]
            # Stride split: teacher i gets indices [i, i+K, i+2K, ...]
            icl_list.append([selected[i::K] for i in range(K)])

        # ------------------------------------------------------------------
        # Step 2: collect per-teacher votes
        # shape: (K, n_test, n_classes)
        # ------------------------------------------------------------------
        all_votes = np.zeros((K, n_test, n_classes), dtype=np.int32)

        print(f"\nCollecting votes from {K} teachers x {n_test} queries ...")
        for i in range(K):
            print(f"  Teacher {i + 1}/{K}", flush=True)
            # For teacher i: each query j uses icl_list[j][i] as context
            train_indices_per_query = [icl_list[j][i] for j in range(n_test)]
            prompts, _ = proc.create_prompts_diff(train_indices_per_query)
            outputs    = model.complete_all(prompts)

            for j, out in enumerate(outputs):
                pred_label = out.completion
                if pred_label in labels:
                    all_votes[i, j, labels.index(pred_label)] = 1
                else:
                    # Uniform random fallback for unrecognised responses
                    all_votes[i, j, np.random.randint(n_classes)] = 1

        # ------------------------------------------------------------------
        # Step 3: ground-truth labels
        # ------------------------------------------------------------------
        gt_labels = []
        for j in range(n_test):
            ex     = proc.test_dataset[j]
            fields = proc.convert_example_to_template_fields(ex)
            gt_labels.append(
                fields.get("label_text", labels[ex["label"]])
            )

        # ------------------------------------------------------------------
        # Step 4: non-private baseline (argmax of summed votes)
        # ------------------------------------------------------------------
        total_votes      = all_votes.sum(axis=0)   # (n_test, n_classes)
        preds_nonprivate = [labels[int(np.argmax(total_votes[j]))]
                            for j in range(n_test)]
        acc_nonprivate   = sum(p == g for p, g in zip(preds_nonprivate, gt_labels)) / n_test

        print(f"\n{'=' * 55}")
        print(f"{'Setting':<35}  {'Accuracy':>10}")
        print(f"{'-' * 55}")
        print(f"{'Non-private (eps=inf)':<35}  {acc_nonprivate:>9.1%}")

        results = {"non_private": acc_nonprivate}

        # ------------------------------------------------------------------
        # Step 5a: PAC aggregation
        # ------------------------------------------------------------------
        if privacy_mode == "pac":
            eta_list = self.conf.get("etas", [0.5, 2.0, 8.0])

            for eta in eta_list:
                preds    = []
                sigma_log = []

                for j in range(n_test):
                    # teacher_votes_q: (K, C) — K independent samples of M(X) for query j
                    teacher_votes_q = all_votes[:, j, :].astype(float)

                    # Estimate output covariance Sigma_out(q) using sample covariance
                    cov_q = np.cov(teacher_votes_q.T)   # (C, C)

                    # Calibrate sigma* via PAC Theorem 3 eigenvalue formula
                    sigma_star = calibrate_sigma_from_covariance(cov_q, eta)
                    sigma_log.append(sigma_star)

                    # Add Gaussian noise to aggregated vote and predict
                    noisy = total_votes[j].astype(float) + \
                            np.random.normal(0.0, sigma_star, n_classes)
                    preds.append(labels[int(np.argmax(noisy))])

                acc = sum(p == g for p, g in zip(preds, gt_labels)) / n_test
                mean_sigma = float(np.mean(sigma_log))
                tag = f"PAC (eta={eta})"
                print(f"{tag:<35}  {acc:>9.1%}   mean_sigma={mean_sigma:.3f}")
                results[f"eta_{eta}"] = {"acc": acc, "mean_sigma": mean_sigma}

        # ------------------------------------------------------------------
        # Step 5b: DP aggregation
        # ------------------------------------------------------------------
        else:
            delta    = self.conf.get("delta", 1e-5)
            eps_list = self.conf.get("epsilons", [1, 3, 8])

            for eps in eps_list:
                sigma = self._rnm_sigma(eps, delta)
                preds = []
                for j in range(n_test):
                    noisy = total_votes[j].astype(float) + \
                            np.random.normal(0.0, sigma, n_classes)
                    preds.append(labels[int(np.argmax(noisy))])

                acc = sum(p == g for p, g in zip(preds, gt_labels)) / n_test
                tag = f"DP (eps={eps})"
                print(f"{tag:<35}  {acc:>9.1%}   sigma={sigma:.3f}")
                results[f"eps_{eps}"] = {"acc": acc, "sigma": sigma}

        print(f"{'=' * 55}\n")
        return results


class RandomClassBalancedStrategy(BaseStrategy):
    def run_strategy(self, proc: BaseProcessor, model: GPT2Wrapper, shot: int):
        logger.info(f"RandomClassBalancedStrategy - shot={shot}")
        train_indices_by_class = defaultdict(list)

        for i, example in enumerate(proc.train_dataset):
            label = proc.convert_example_to_template_fields(example)["label_text"]
            train_indices_by_class[label].append(i)

        for indices in train_indices_by_class.values():
            random.shuffle(indices)

        label_list = list(train_indices_by_class.keys())
        random.shuffle(label_list)

        train_indices = []
        for i in range(shot):
            label = label_list[i % len(label_list)]
            train_indices.append(train_indices_by_class[label].pop())

        random.shuffle(train_indices)

        prompts, cali_prompts = proc.create_prompts(train_indices)
        outputs    = model.complete_all(prompts, calibration_prompts=cali_prompts)
        eval_result = proc.extract_predictions(outputs)
        self.write_result(eval_result, shot)
        return {"shot": shot, "train_indices": train_indices, "acc": eval_result["acc"]}


class BestPermStrategy(BaseStrategy):
    def run_strategy(self, proc: BaseProcessor, model: GPT2Wrapper, shot: int):
        logger.info(f"BestPermStrategy - shot={shot}")
        perm_accs      = []
        train_original = random.sample(range(len(proc.train_dataset)), k=shot)

        for train_indices in permutations(train_original):
            prompts, cali_prompts = proc.create_prompts(train_indices, split="val")
            outputs    = model.complete_all(prompts, calibration_prompts=cali_prompts)
            eval_result = proc.extract_predictions(outputs, split="val")
            perm_accs.append((train_indices, eval_result["acc"]))

        best_indices = max(perm_accs, key=lambda t: t[1])[0]
        prompts, cali_prompts = proc.create_prompts(best_indices)
        outputs    = model.complete_all(prompts, calibration_prompts=cali_prompts)
        eval_result = proc.extract_predictions(outputs)
        self.write_result(eval_result, shot)
        return {
            "shot": shot,
            "train_indices": best_indices,
            "acc": eval_result["acc"],
            "perm_accs": perm_accs,
        }


class BestOfKStrategy(BaseStrategy):
    def run_strategy(self, proc: BaseProcessor, model: GPT2Wrapper, shot: int):
        logger.info(f"BestOfKStrategy - shot={shot}")
        K            = 10
        best_indices = None
        best_val_acc = -1.0

        for _ in range(K):
            train_indices = random.sample(range(len(proc.train_dataset)), k=shot)
            prompts, cali_prompts = proc.create_prompts(train_indices, split="val")
            outputs    = model.complete_all(prompts, calibration_prompts=cali_prompts)
            eval_result = proc.extract_predictions(outputs, split="val")
            if eval_result["acc"] > best_val_acc:
                best_val_acc = eval_result["acc"]
                best_indices = train_indices

        prompts, cali_prompts = proc.create_prompts(best_indices)
        outputs    = model.complete_all(prompts, calibration_prompts=cali_prompts)
        eval_result = proc.extract_predictions(outputs)
        self.write_result(eval_result, shot)
        return {"shot": shot, "train_indices": best_indices, "acc": eval_result["acc"]}
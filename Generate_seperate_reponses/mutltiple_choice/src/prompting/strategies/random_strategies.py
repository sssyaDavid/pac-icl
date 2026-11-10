import logging
import random
from collections import defaultdict
from itertools import permutations

from prompting import BaseProcessor, GPT2Wrapper
from prompting.strategies.base_strategy import BaseStrategy
import random
import numpy as np

logger = logging.getLogger(__name__)

from tqdm import tqdm

class RandomStrategy(BaseStrategy):
    def run_strategy(self, proc: BaseProcessor, model: GPT2Wrapper, shot: int):
        logger.info(f"RandomStrategy - shot={shot}")
        train_indices = random.sample(range(len(proc.train_dataset)), k=shot)
        prompts, cali_prompts = proc.create_prompts(train_indices)
        outputs = model.complete_all(prompts, calibration_prompts=cali_prompts, file_name=self.output_dir)
        eval_result = proc.extract_predictions(outputs)
        self.write_result(eval_result, shot)

        simple_result = {
            "shot": shot,
            "train_indices": train_indices,
            "acc": eval_result["acc"],
        }
        return simple_result


class PATEStrategy(BaseStrategy):

    @staticmethod
    def _rnm_sigma(epsilon: float, delta: float = 1e-5) -> float:
        """
        DP noise level for Gaussian RNM.
        sigma = sqrt(2) * sqrt(2*ln(1.25/delta)) / epsilon
        L2-sensitivity of the vote vector = sqrt(2)
        """
        import math
        return math.sqrt(2) * math.sqrt(2 * math.log(1.25 / delta)) / epsilon

    @staticmethod
    def _pac_sigma(rho: float) -> float:
        """
        PAC-Private noise level for Gaussian RNM.
        For Gaussian mechanism with L2 sensitivity sqrt(2):
          I(X; M(X)) <= Delta_2^2 / (2*sigma^2) = 1/sigma^2 = rho
        So sigma = 1/sqrt(rho).

        Compare to DP: sigma_DP = sqrt(2)*sqrt(2*ln(1.25/delta))/epsilon
        Setting rho = epsilon^2/2 gives sigma_PAC << sigma_DP,
        reflecting that PAC is an average-case (mutual-information) bound
        rather than DP's worst-case guarantee.
        """
        import math
        return 1.0 / math.sqrt(rho)

    def run_strategy(self, proc: BaseProcessor, model: GPT2Wrapper, shot: int):
        # Determine privacy mode from config: "dp" (default) or "pac"
        privacy_mode = self.conf.get("privacy", "dp").lower()
        logger.info(f"PATEStrategy - shot={shot}, privacy={privacy_mode}")
        print(f"Train size: {len(proc.train_dataset)},  Test size: {len(proc.test_dataset)}")
        print(f"Privacy mode: {privacy_mode.upper()}")

        Teacher       = self.conf.ensemble
        subsample_num = shot * Teacher
        n_test        = len(proc.test_dataset)
        labels        = proc.labels
        n_classes     = len(labels)

        # Build per-query teacher ICL index lists
        icl_list = []
        for j in range(n_test):
            random.seed(self.conf.seed + j)
            rand = random.Random()
            index_list = list(range(len(proc.train_dataset)))
            rand.shuffle(index_list)
            selected_list = index_list[:subsample_num]
            icl_list.append([selected_list[i::Teacher] for i in range(Teacher)])

        # votes[j, c] = number of teachers predicting class c for query j
        votes = np.zeros((n_test, n_classes), dtype=int)

        print(f"\nCollecting votes from {Teacher} teachers x {n_test} queries ...")
        for i in range(Teacher):
            print(f"  Teacher {i+1}/{Teacher}", flush=True)
            train_indices_per_query = [icl_list[j][i] for j in range(n_test)]
            prompts, _ = proc.create_prompts_diff(train_indices_per_query)
            outputs    = model.complete_all(prompts)
            for j, out in enumerate(outputs):
                pred_label = out.completion
                if pred_label in labels:
                    votes[j, labels.index(pred_label)] += 1
                else:
                    votes[j, np.random.randint(n_classes)] += 1

        # Ground-truth labels
        gt_labels = []
        for j in range(n_test):
            ex     = proc.test_dataset[j]
            fields = proc.convert_example_to_template_fields(ex)
            if "label_text" in fields:
                gt_labels.append(fields["label_text"])
            else:
                gt_labels.append(labels[ex["label"]])

        # Non-private baseline
        preds_nonprivate = [labels[int(np.argmax(votes[j]))] for j in range(n_test)]
        acc_nonprivate   = sum(p == g for p, g in zip(preds_nonprivate, gt_labels)) / n_test

        print(f"\n{'='*55}")
        print(f"{'Setting':<25}  {'Accuracy':>10}")
        print(f"{'-'*55}")
        print(f"{'Non-private (eps=inf)':<25}  {acc_nonprivate:>9.1%}")

        results = {"non_private": acc_nonprivate}

        if privacy_mode == "pac":
            # PAC aggregation: Gaussian RNM with sigma = 1/sqrt(rho)
            # rho is the mutual-information privacy budget
            for rho in self.conf.get("rhos", [0.5, 4.5, 32.0]):
                sigma = self._pac_sigma(rho)
                preds = []
                for j in range(n_test):
                    noisy = votes[j].astype(float) + np.random.normal(0, sigma, n_classes)
                    preds.append(labels[int(np.argmax(noisy))])
                acc = sum(p == g for p, g in zip(preds, gt_labels)) / n_test
                print(f"{'PAC (rho='+str(rho)+')':<25}  {acc:>9.1%}")
                results[f"rho_{rho}"] = acc

        else:
            # DP aggregation: Gaussian RNM with sigma from (epsilon, delta)
            delta = self.conf.get("delta", 1e-5)
            for eps in self.conf.get("epsilons", [1, 3, 8]):
                sigma = self._rnm_sigma(eps, delta)
                preds = []
                for j in range(n_test):
                    noisy = votes[j].astype(float) + np.random.normal(0, sigma, n_classes)
                    preds.append(labels[int(np.argmax(noisy))])
                acc = sum(p == g for p, g in zip(preds, gt_labels)) / n_test
                print(f"{'DP (eps='+str(eps)+')':<25}  {acc:>9.1%}")
                results[f"eps_{eps}"] = acc

        print(f"{'='*55}\n")
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
        outputs = model.complete_all(prompts, calibration_prompts=cali_prompts)
        eval_result = proc.extract_predictions(outputs)
        self.write_result(eval_result, shot)

        simple_result = {
            "shot": shot,
            "train_indices": train_indices,
            "acc": eval_result["acc"],
        }
        return simple_result


class BestPermStrategy(BaseStrategy):
    def run_strategy(self, proc: BaseProcessor, model: GPT2Wrapper, shot: int):
        logger.info(f"BestPermStrategy - shot={shot}")

        perm_accs = []
        train_original = random.sample(range(len(proc.train_dataset)), k=shot)

        for train_indices in permutations(train_original):
            prompts, cali_prompts = proc.create_prompts(train_indices, split="val")
            outputs = model.complete_all(prompts, calibration_prompts=cali_prompts)
            eval_result = proc.extract_predictions(outputs, split="val")
            perm_accs.append((train_indices, eval_result["acc"]))

        best_indices = max(perm_accs, key=lambda t: t[1])[0]
        prompts, cali_prompts = proc.create_prompts(best_indices)
        outputs = model.complete_all(prompts, calibration_prompts=cali_prompts)
        eval_result = proc.extract_predictions(outputs)
        self.write_result(eval_result, shot)

        simple_result = {
            "shot": shot,
            "train_indices": best_indices,
            "acc": eval_result["acc"],
            "perm_accs": perm_accs,
        }
        return simple_result


class BestOfKStrategy(BaseStrategy):
    def run_strategy(self, proc: BaseProcessor, model: GPT2Wrapper, shot: int):
        logger.info(f"BestOfKStrategy - shot={shot}")
        K = 10
        best_indices = None
        best_val_acc = -1.0

        for i in range(K):
            train_indices = random.sample(range(len(proc.train_dataset)), k=shot)
            prompts, cali_prompts = proc.create_prompts(train_indices, split="val")
            outputs = model.complete_all(prompts, calibration_prompts=cali_prompts)
            eval_result = proc.extract_predictions(outputs, split="val")
            if eval_result["acc"] > best_val_acc:
                best_val_acc = eval_result["acc"]
                best_indices = train_indices

        prompts, cali_prompts = proc.create_prompts(best_indices)
        outputs = model.complete_all(prompts, calibration_prompts=cali_prompts)
        eval_result = proc.extract_predictions(outputs)
        self.write_result(eval_result, shot)

        simple_result = {
            "shot": shot,
            "train_indices": best_indices,
            "acc": eval_result["acc"],
        }
        return simple_result
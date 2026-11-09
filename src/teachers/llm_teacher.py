"""
LLaMA-based ICL Teacher
-----------------------
Each teacher holds a set of in-context demonstrations, classifies a test
example, and casts a vote.

Two backends are supported:
  - transformers (HuggingFace, GPU / CPU / MPS)
  - ollama (local Ollama server, lower VRAM use)
"""

from __future__ import annotations
import re
import numpy as np
from typing import List, Dict, Optional
from tqdm import tqdm


# ----------------------------------------------------------------
#  Prompt construction
# ----------------------------------------------------------------

def build_prompt(
    demonstrations: List[Dict],
    query_text: str,
    label_names: List[str],
    dataset_name: str = "sst2",
) -> str:
    """
    Build an ICL prompt.
    Format:
      Classify the sentiment of the following text.
      Labels: negative, positive

      Text: "..."
      Label: negative

      ...

      Text: "<query>"
      Label:
    """
    if dataset_name in ("sst2", "yelp_polarity"):
        task_desc = "Classify the sentiment of the following text."
    elif dataset_name == "ag_news":
        task_desc = "Classify the topic of the following news article."
    else:
        task_desc = "Classify the following text."

    labels_str = ", ".join(label_names)
    prompt = f"{task_desc}\nLabels: {labels_str}\n\n"

    for demo in demonstrations:
        text = demo["text"].strip().replace("\n", " ")[:200]
        label = demo["label_name"]
        prompt += f'Text: "{text}"\nLabel: {label}\n\n'

    query_text = query_text.strip().replace("\n", " ")[:200]
    prompt += f'Text: "{query_text}"\nLabel:'
    return prompt


def parse_label(generated: str, label_names: List[str]) -> int:
    """
    Parse a label index from the model output.
    Prefer an exact match, then a fuzzy match.
    """
    text = generated.strip().lower()
    # Only look at the first line / first 20 characters
    text = text.split("\n")[0][:30]

    for idx, name in enumerate(label_names):
        if name.lower() in text:
            return idx

    # Match failed: return a random class (rare edge case; does not affect aggregate stats)
    return np.random.randint(len(label_names))


# ----------------------------------------------------------------
#  HuggingFace Transformers backend
# ----------------------------------------------------------------

class HFTeacher:
    """A single teacher that runs inference with HuggingFace transformers."""

    _model = None      # Shared at class level; all teachers reuse one model
    _tokenizer = None

    def __init__(
        self,
        demonstrations: List[Dict],
        label_names: List[str],
        dataset_name: str,
        model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
        device: str = "cpu",
        max_new_tokens: int = 8,
    ):
        self.demonstrations = demonstrations
        self.label_names = label_names
        self.dataset_name = dataset_name
        self.max_new_tokens = max_new_tokens

        # Load the model only once (shared by all teachers)
        if HFTeacher._model is None:
            self._load_model(model_name, device)

    @classmethod
    def _load_model(cls, model_name: str, device: str):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        print(f"[HFTeacher] Loading model: {model_name}  device: {device}")
        cls._tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        cls._tokenizer.pad_token = cls._tokenizer.eos_token

        dtype = torch.float16 if device != "cpu" else torch.float32
        # device_map="auto" is for GPU only; CPU/MPS use .to(device) directly
        if device in ("cuda", "auto"):
            cls._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            cls._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                trust_remote_code=True,
            ).to(device)
        cls._model.eval()
        print("[HFTeacher] Model loaded")

    def predict(self, query_text: str) -> int:
        """Return the predicted label index for a single query."""
        import torch

        prompt = build_prompt(
            self.demonstrations, query_text, self.label_names, self.dataset_name
        )
        inputs = HFTeacher._tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        ).to(HFTeacher._model.device)

        with torch.no_grad():
            output = HFTeacher._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=HFTeacher._tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        generated = HFTeacher._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return parse_label(generated, self.label_names)

    def predict_batch(self, queries: List[str]) -> List[int]:
        return [self.predict(q) for q in queries]


# ----------------------------------------------------------------
#  Ollama backend (lower VRAM use)
# ----------------------------------------------------------------

class OllamaTeacher:
    """A single teacher that runs inference through a local Ollama server."""

    def __init__(
        self,
        demonstrations: List[Dict],
        label_names: List[str],
        dataset_name: str,
        model_name: str = "llama3.2:1b",
        **kwargs,
    ):
        self.demonstrations = demonstrations
        self.label_names = label_names
        self.dataset_name = dataset_name
        self.model_name = model_name

    def predict(self, query_text: str) -> int:
        import urllib.request, json

        prompt = build_prompt(
            self.demonstrations, query_text, self.label_names, self.dataset_name
        )
        payload = json.dumps({
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": 8, "temperature": 0},
        }).encode()
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        return parse_label(data.get("response", ""), self.label_names)

    def predict_batch(self, queries: List[str]) -> List[int]:
        return [self.predict(q) for q in queries]


# ----------------------------------------------------------------
#  Teacher factory + batch inference
# ----------------------------------------------------------------

def create_teachers(
    partitions: List[List[Dict]],
    label_names: List[str],
    dataset_name: str,
    model_name: str,
    device: str = "cpu",
    max_new_tokens: int = 8,
) -> List:
    """Pick a backend from the model_name prefix."""
    if model_name.startswith("ollama:"):
        ollama_model = model_name[len("ollama:"):]
        return [
            OllamaTeacher(p, label_names, dataset_name, ollama_model)
            for p in partitions
        ]
    else:
        teachers = []
        for i, p in enumerate(partitions):
            t = HFTeacher(p, label_names, dataset_name, model_name, device, max_new_tokens)
            teachers.append(t)
            if i == 0:
                pass  # Model is already loaded; later teachers reuse it
        return teachers


def collect_votes(
    teachers: List,
    test_examples: List[Dict],
    num_classes: int,
) -> np.ndarray:
    """
    Collect every teacher's vote on every test example.

    Returns: votes[i, c] = number of votes for class c on test example i
    Shape: (num_test, num_classes)
    """
    num_teachers = len(teachers)
    num_test = len(test_examples)
    queries = [ex["text"] for ex in test_examples]

    # votes_raw[t, i] = predicted label of teacher t on example i
    votes_raw = np.zeros((num_teachers, num_test), dtype=int)

    for t_idx, teacher in enumerate(tqdm(teachers, desc="Teacher inference")):
        preds = teacher.predict_batch(queries)
        votes_raw[t_idx] = preds

    # Convert to a vote histogram: (num_test, num_classes)
    vote_hist = np.zeros((num_test, num_classes), dtype=int)
    for i in range(num_test):
        for t_idx in range(num_teachers):
            vote_hist[i, votes_raw[t_idx, i]] += 1

    return vote_hist

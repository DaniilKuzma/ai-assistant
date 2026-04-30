"""Optional RuBERT context reranker for ambiguous correction candidates."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Sequence


@dataclass(frozen=True)
class ContextScore:
    text: str
    score: float
    token_count: int = 0
    available: bool = True
    reason: str = "ok"


class ContextReranker:
    """Score candidate substitutions with a masked language model.

    The class is deliberately fail-open: missing torch/transformers/model files
    never break correction. Runtime code can inspect ``available``/``reason``
    and simply keep the neural edit model's original decision.
    """

    def __init__(
        self,
        model_name: str = "DeepPavlov/rubert-base-cased",
        *,
        enabled: bool = True,
        window: int = 12,
        device: str | None = None,
    ):
        self.model_name = model_name
        self.enabled = enabled
        self.window = max(1, int(window))
        self.device = device
        self.cache: dict[tuple[str, str, tuple[str, ...]], list[ContextScore]] = {}
        self._load_attempted = False
        self._available = False
        self.disabled_reason = "not_loaded"
        self.tokenizer = None
        self.model = None
        self.torch = None

    @property
    def available(self) -> bool:
        return self._ensure_loaded()

    def score_candidates(
        self,
        words: Sequence[str],
        index: int,
        source: str,
        candidates: Sequence[str],
    ) -> list[ContextScore]:
        unique_candidates = list(dict.fromkeys(str(candidate) for candidate in candidates if str(candidate)))
        cache_key = (self._context_hash(words, index), str(source), tuple(unique_candidates))
        if cache_key in self.cache:
            return self.cache[cache_key]

        if not self._ensure_loaded():
            result = [
                ContextScore(text=candidate, score=0.0, available=False, reason=self.disabled_reason)
                for candidate in unique_candidates
            ]
            self.cache[cache_key] = result
            return result

        result = [self._score_one(words, index, candidate) for candidate in unique_candidates]
        self.cache[cache_key] = result
        return result

    def _ensure_loaded(self) -> bool:
        if self._available:
            return True
        if not self.enabled:
            self.disabled_reason = "disabled"
            return False
        if self._load_attempted:
            return False

        self._load_attempted = True
        try:
            import torch
            from transformers import AutoModelForMaskedLM, AutoTokenizer

            self.torch = torch
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self.model.eval()
            if not getattr(self.tokenizer, "mask_token", None):
                self.disabled_reason = "missing_mask_token"
                return False
            self._available = True
            self.disabled_reason = "ok"
            return True
        except Exception as exc:
            self.disabled_reason = f"unavailable:{type(exc).__name__}"
            self._available = False
            return False

    def _score_one(self, words: Sequence[str], index: int, candidate: str) -> ContextScore:
        assert self.tokenizer is not None
        assert self.model is not None
        assert self.torch is not None

        try:
            candidate_ids = self.tokenizer(candidate, add_special_tokens=False)["input_ids"]
            if not candidate_ids:
                return ContextScore(candidate, -math.inf, 0, True, "empty_candidate_tokens")

            start = max(0, index - self.window)
            end = min(len(words), index + self.window + 1)
            left = [str(word) for word in words[start:index]]
            right = [str(word) for word in words[index + 1 : end]]
            mask_tokens = [self.tokenizer.mask_token] * len(candidate_ids)
            masked_text = " ".join(left + mask_tokens + right)
            encoded = self.tokenizer(masked_text, return_tensors="pt")
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            mask_positions = (encoded["input_ids"][0] == self.tokenizer.mask_token_id).nonzero(as_tuple=False).flatten()
            if len(mask_positions) != len(candidate_ids):
                return ContextScore(candidate, -math.inf, len(candidate_ids), True, "mask_count_mismatch")

            with self.torch.no_grad():
                logits = self.model(**encoded).logits[0]
                log_probs = self.torch.log_softmax(logits, dim=-1)

            total = 0.0
            for position, token_id in zip(mask_positions.tolist(), candidate_ids):
                total += float(log_probs[position, token_id].detach().cpu())
            score = total / max(1, len(candidate_ids))
            return ContextScore(candidate, score, len(candidate_ids), True, "ok")
        except Exception as exc:
            return ContextScore(candidate, -math.inf, 0, True, f"score_error:{type(exc).__name__}")

    def _context_hash(self, words: Sequence[str], index: int) -> str:
        start = max(0, int(index) - self.window)
        end = min(len(words), int(index) + self.window + 1)
        context = "\u241f".join(str(word) for word in words[start:end])
        return hashlib.sha1(context.encode("utf-8")).hexdigest()

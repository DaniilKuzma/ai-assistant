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
        device: str | None = "cpu",
        batch_size: int = 16,
    ):
        self.model_name = model_name
        self.enabled = enabled
        self.window = max(1, int(window))
        self.device = device or "cpu"
        self.batch_size = max(1, int(batch_size))
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

        result = self._score_batch(words, index, unique_candidates)
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
            from transformers.utils import logging as hf_logging

            self.torch = torch
            hf_logging.set_verbosity_error()
            self._prefer_stable_torch_attention(torch)
            try:
                from huggingface_hub.utils import logging as hub_logging

                hub_logging.set_verbosity_error()
            except Exception:
                pass
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)
            resolved_device = self._resolve_device(torch)
            try:
                self.model.to(resolved_device)
                self.device = resolved_device
            except Exception as exc:
                if resolved_device == "cpu":
                    raise
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                self.model.to("cpu")
                self.device = "cpu"
                self.disabled_reason = self._exception_reason("cuda_fallback_cpu", exc)
            self.model.eval()
            if not getattr(self.tokenizer, "mask_token", None):
                self.disabled_reason = "missing_mask_token"
                return False
            self._available = True
            if not str(self.disabled_reason).startswith("cuda_fallback_cpu"):
                self.disabled_reason = "ok"
            return True
        except Exception as exc:
            self.disabled_reason = self._exception_reason("unavailable", exc)
            self._available = False
            return False

    def _resolve_device(self, torch_module) -> str:
        requested = str(self.device or "cpu").lower()
        if requested == "auto":
            return "cuda" if torch_module.cuda.is_available() else "cpu"
        if requested == "cuda":
            return "cuda" if torch_module.cuda.is_available() else "cpu"
        if requested == "cpu":
            return requested
        return "cpu"

    def _score_batch(self, words: Sequence[str], index: int, candidates: Sequence[str]) -> list[ContextScore]:
        assert self.tokenizer is not None
        assert self.model is not None
        assert self.torch is not None

        prepared = []
        immediate: dict[str, ContextScore] = {}
        for candidate in candidates:
            try:
                candidate_ids = self.tokenizer(candidate, add_special_tokens=False)["input_ids"]
                if not candidate_ids:
                    immediate[candidate] = ContextScore(candidate, -math.inf, 0, True, "empty_candidate_tokens")
                    continue
                prepared.append((candidate, candidate_ids, self._masked_text(words, index, len(candidate_ids))))
            except Exception as exc:
                immediate[candidate] = ContextScore(
                    candidate,
                    -math.inf,
                    0,
                    True,
                    self._exception_reason("tokenize_error", exc),
                )

        result_by_text: dict[str, ContextScore] = dict(immediate)
        for start in range(0, len(prepared), self.batch_size):
            chunk = prepared[start : start + self.batch_size]
            masked_texts = [item[2] for item in chunk]
            try:
                encoded = self.tokenizer(masked_texts, return_tensors="pt", padding=True)
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                with self.torch.inference_mode():
                    logits = self.model(**encoded).logits
                    log_probs = self.torch.log_softmax(logits, dim=-1)

                for row, (candidate, candidate_ids, _) in enumerate(chunk):
                    mask_positions = (
                        encoded["input_ids"][row] == self.tokenizer.mask_token_id
                    ).nonzero(as_tuple=False).flatten()
                    if len(mask_positions) != len(candidate_ids):
                        result_by_text[candidate] = ContextScore(
                            candidate,
                            -math.inf,
                            len(candidate_ids),
                            True,
                            "mask_count_mismatch",
                        )
                        continue
                    total = 0.0
                    for position, token_id in zip(mask_positions.tolist(), candidate_ids):
                        total += float(log_probs[row, position, token_id].detach().cpu())
                    result_by_text[candidate] = ContextScore(
                        candidate,
                        total / max(1, len(candidate_ids)),
                        len(candidate_ids),
                        True,
                        "ok",
                    )
            except Exception as exc:
                if self.device != "cpu" and self._fallback_to_cpu():
                    return self._score_batch(words, index, candidates)
                reason = self._exception_reason("score_error", exc)
                for candidate, candidate_ids, _ in chunk:
                    result_by_text[candidate] = ContextScore(candidate, -math.inf, len(candidate_ids), True, reason)

        return [result_by_text[candidate] for candidate in candidates]

    def _score_one(self, words: Sequence[str], index: int, candidate: str) -> ContextScore:
        assert self.tokenizer is not None
        assert self.model is not None
        assert self.torch is not None

        try:
            candidate_ids = self.tokenizer(candidate, add_special_tokens=False)["input_ids"]
            if not candidate_ids:
                return ContextScore(candidate, -math.inf, 0, True, "empty_candidate_tokens")

            masked_text = self._masked_text(words, index, len(candidate_ids))
            encoded = self.tokenizer(masked_text, return_tensors="pt")
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            mask_positions = (encoded["input_ids"][0] == self.tokenizer.mask_token_id).nonzero(as_tuple=False).flatten()
            if len(mask_positions) != len(candidate_ids):
                return ContextScore(candidate, -math.inf, len(candidate_ids), True, "mask_count_mismatch")

            with self.torch.inference_mode():
                logits = self.model(**encoded).logits[0]
                log_probs = self.torch.log_softmax(logits, dim=-1)

            total = 0.0
            for position, token_id in zip(mask_positions.tolist(), candidate_ids):
                total += float(log_probs[position, token_id].detach().cpu())
            score = total / max(1, len(candidate_ids))
            return ContextScore(candidate, score, len(candidate_ids), True, "ok")
        except Exception as exc:
            if self.device != "cpu" and self._fallback_to_cpu():
                return self._score_one(words, index, candidate)
            return ContextScore(candidate, -math.inf, 0, True, self._exception_reason("score_error", exc))

    def _masked_text(self, words: Sequence[str], index: int, mask_count: int) -> str:
        assert self.tokenizer is not None
        start = max(0, index - self.window)
        end = min(len(words), index + self.window + 1)
        left = [str(word) for word in words[start:index]]
        right = [str(word) for word in words[index + 1 : end]]
        mask_tokens = [self.tokenizer.mask_token] * max(1, int(mask_count))
        return " ".join(left + mask_tokens + right)

    @staticmethod
    def _exception_reason(prefix: str, exc: Exception, max_len: int = 180) -> str:
        message = " ".join(str(exc).split())
        if len(message) > max_len:
            message = message[: max_len - 3] + "..."
        if message:
            return f"{prefix}:{type(exc).__name__}:{message}"
        return f"{prefix}:{type(exc).__name__}"

    def _context_hash(self, words: Sequence[str], index: int) -> str:
        start = max(0, int(index) - self.window)
        end = min(len(words), int(index) + self.window + 1)
        context = "\u241f".join(str(word) for word in words[start:end])
        return hashlib.sha1(context.encode("utf-8")).hexdigest()

    @staticmethod
    def _prefer_stable_torch_attention(torch_module) -> None:
        try:
            cuda_backend = torch_module.backends.cuda
            cuda_backend.enable_flash_sdp(False)
            cuda_backend.enable_mem_efficient_sdp(False)
            cuda_backend.enable_math_sdp(True)
        except Exception:
            pass

    def _fallback_to_cpu(self) -> bool:
        if self.model is None or self.torch is None:
            return False
        try:
            try:
                self.torch.cuda.empty_cache()
            except Exception:
                pass
            self.model.to("cpu")
            self.device = "cpu"
            return True
        except Exception:
            return False

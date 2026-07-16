"""Reference-model token surprisal: the black-box detector's instrument.

ReferenceModel is the interface; MockReferenceModel is a deterministic stand-in for
TDD; LlamaCppReferenceModel (below) is the real local-logprob backend.
Surprisal s_t = -log2 P(token | context); higher mean = less fluent ('stilted')."""
from __future__ import annotations
import math
from abc import ABC, abstractmethod

# A tiny unigram frequency table standing in for a real LM's word likelihoods.
# Common words -> low surprisal; rare/ornate words -> high surprisal.
_COMMON = {"the": 1, "a": 1, "of": 1, "on": 2, "in": 2, "sat": 3, "cat": 4, "mat": 4,
           "warm": 4, "and": 1, "to": 1, "was": 2, "is": 2}
_DEFAULT_RANK = 12  # unknown word -> treated as moderately rare


class ReferenceModel(ABC):
    @abstractmethod
    def surprisals(self, text: str) -> list[float]:
        ...

    def mean_surprisal(self, text: str) -> float:
        s = self.surprisals(text)
        return sum(s) / len(s) if s else 0.0

    def token_stats(self, text: str) -> tuple[list[float], list[int] | None]:
        """Per-token (surprisals, ranks). Default: ranks unavailable -> None.
        Subclasses with logit access override to supply ranks (rank 0 = argmax)."""
        return self.surprisals(text), None


class MockReferenceModel(ReferenceModel):
    """Deterministic surprisal from a unigram-rarity proxy. No model, no randomness.
    default_rank tunes the unknown-word rarity so two distinct mock models can be
    compared (the cross-perplexity baseline needs two reference models)."""

    def __init__(self, default_rank: int = _DEFAULT_RANK):
        self._default_rank = default_rank

    def surprisals(self, text: str) -> list[float]:
        out = []
        for tok in text.split():
            w = tok.strip(".,;:!?").lower()
            rank = _COMMON.get(w, self._default_rank + (len(w) // 3))  # longer/ornate -> rarer
            out.append(math.log2(rank + 1))
        return out

    def token_stats(self, text: str) -> tuple[list[float], list[int]]:
        surprisals, ranks = [], []
        for tok in text.split():
            w = tok.strip(".,;:!?").lower()
            rarity = _COMMON.get(w, self._default_rank + (len(w) // 3))
            surprisals.append(math.log2(rarity + 1))
            ranks.append(rarity - 1)  # common (rarity 1) -> rank 0 (argmax)
        return surprisals, ranks


class LlamaCppReferenceModel(ReferenceModel):
    """Local open-weight backend via llama-cpp-python (full per-token logprobs).
    Pi 5 / Umbrel: a 1-3B GGUF at Q8. Generation-free extraction — NOT create_completion
    (echo=True generates a continuation that contaminates the measurement, verified)."""

    def __init__(self, model_path: str, n_ctx: int = 4096):
        from llama_cpp import Llama  # lazy import so the package works without the optional dep
        self._llm = Llama(model_path=model_path, n_ctx=n_ctx, logits_all=True, verbose=False)

    def surprisals(self, text: str) -> list[float]:
        import numpy as np
        toks = self._llm.tokenize(text.encode("utf-8"), add_bos=True)
        self._llm.reset()
        self._llm.eval(toks)
        out = []
        for i in range(len(toks) - 1):                       # scores[i] predicts toks[i+1]
            logits = np.array(self._llm.scores[i], dtype=np.float64)
            m = logits.max()
            lse = m + math.log(float(np.exp(logits - m).sum()))
            out.append(-(logits[toks[i + 1]] - lse) / math.log(2))  # natural log -> bits
        return out

    def token_stats(self, text: str) -> tuple[list[float], list[int]]:
        # Same generation-free eval as surprisals(); additionally compute the rank of
        # the realized token = count of vocab entries with logit strictly greater.
        import numpy as np
        toks = self._llm.tokenize(text.encode("utf-8"), add_bos=True)
        self._llm.reset()
        self._llm.eval(toks)
        surprisals, ranks = [], []
        for i in range(len(toks) - 1):                       # scores[i] predicts toks[i+1]
            logits = np.array(self._llm.scores[i], dtype=np.float64)
            m = logits.max()
            lse = m + math.log(float(np.exp(logits - m).sum()))
            realized = toks[i + 1]
            surprisals.append(-(logits[realized] - lse) / math.log(2))   # bits
            ranks.append(int((logits > logits[realized]).sum()))          # 0 = argmax
        return surprisals, ranks

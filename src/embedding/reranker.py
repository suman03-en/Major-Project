"""
Cross-Encoder Re-ranker
=======================
Loads BAAI/bge-reranker-v2-m3 via HuggingFace ``transformers`` and
re-scores candidate chunks retrieved by the hybrid Qdrant search.
The cross-encoder jointly processes each (query, document) pair,
producing a deeply contextual relevance score that is significantly
more accurate than the bi-encoder cosine / RRF scores alone.

Uses ``AutoModelForSequenceClassification`` + ``AutoTokenizer`` directly
instead of the ``FlagReranker`` wrapper to avoid tokenizer version
incompatibilities.

Usage:
    from src.embedding.reranker import CrossEncoderReranker

    reranker = CrossEncoderReranker()
    reranked = reranker.rerank(query="कम्पनी दर्ता", candidates=results, top_k=5)
"""

import os
import logging
from typing import Dict, Any, List, Optional

# Suppress noisy HuggingFace Hub logs during model download.
os.environ.setdefault("HF_HUB_VERBOSITY", "error")

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default model – BAAI/bge-reranker-v2-m3: multilingual cross-encoder.
# ---------------------------------------------------------------------------
DEFAULT_RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
MAX_LENGTH: int = 512


class CrossEncoderReranker:
    """Wraps a HuggingFace cross-encoder for two-stage retrieval re-ranking."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        device: Optional[str] = None,
        use_fp16: Optional[bool] = None,
        max_length: int = MAX_LENGTH,
    ) -> None:
        """
        Parameters
        ----------
        model_name : str
            HuggingFace model identifier or local path for the reranker.
        device : str | None
            Torch device string (``"cpu"``, ``"cuda"``).  ``None`` = auto-detect.
        use_fp16 : bool | None
            Enable half-precision inference.  ``None`` = auto (True on CUDA).
        max_length : int
            Maximum token length for tokenizer truncation.
        """
        # Auto-detect CUDA GPU when no device is specified.
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Reranker auto-detected device: %s", device)

        if use_fp16 is None:
            use_fp16 = True if device == "cuda" else False

        self._device = torch.device(device)
        self._max_length = max_length

        logger.info("Loading cross-encoder reranker: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

        if use_fp16:
            self.model.half()

        self.model.to(self._device)

        logger.info(
            "Reranker loaded  ▸ model=%s  ▸ device=%s  ▸ fp16=%s",
            model_name,
            device,
            use_fp16,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_scores(
        self,
        query: str,
        texts: List[str],
        normalize: bool = True,
    ) -> List[float]:
        """
        Compute cross-encoder relevance scores for (query, text) pairs.

        Parameters
        ----------
        query : str
            The user query string.
        texts : list[str]
            Document texts to score against the query.
        normalize : bool
            If True, scores are sigmoid-normalized to [0, 1].

        Returns
        -------
        list[float]
            One relevance score per input text.
        """
        if not texts:
            return []

        pairs = [[query, text] for text in texts]
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self._max_length,
        ).to(self._device)

        logits = self.model(**inputs, return_dict=True).logits.view(-1).float()

        if normalize:
            logits = torch.sigmoid(logits)

        return logits.cpu().tolist()

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5,
        normalize: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Re-rank retrieval candidates using cross-encoder scores.

        Each candidate dict is expected to contain at least a ``"text"``
        key (the raw chunk text used for scoring).

        Parameters
        ----------
        query : str
            The user query string.
        candidates : list[dict]
            Candidate result dicts from ``QdrantVectorStore.search()``.
            Each must have a ``"text"`` key.
        top_k : int
            Number of top results to return after re-ranking.
        normalize : bool
            If True, cross-encoder scores are sigmoid-normalized to [0, 1].

        Returns
        -------
        list[dict]
            The top-k candidates sorted by cross-encoder score (descending).
            Each dict gains two additional keys:
            - ``"rerank_score"``: the cross-encoder relevance score.
            - ``"original_score"``: the original retrieval score preserved
              from the ``"score"`` field.
        """
        if not candidates:
            return []

        texts = [c["text"] for c in candidates]
        scores = self.compute_scores(query, texts, normalize=normalize)

        # Attach scores and sort by cross-encoder relevance.
        reranked: List[Dict[str, Any]] = []
        for candidate, ce_score in zip(candidates, scores):
            entry = dict(candidate)  # Shallow copy to avoid mutating input.
            entry["rerank_score"] = ce_score
            entry["original_score"] = candidate.get("score", 0.0)
            reranked.append(entry)

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

        logger.info(
            "Re-ranked %d candidates → top %d returned.",
            len(candidates),
            min(top_k, len(reranked)),
        )

        return reranked[:top_k]

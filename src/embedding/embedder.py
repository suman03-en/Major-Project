"""
Embedder Module
===============
Loads BAAI/bge-m3 via SentenceTransformer and generates dense vector
embeddings for structured legal-text chunks produced by the extraction
pipeline.

Usage:
    from src.embedding.embedder import LegalChunkEmbedder

    embedder = LegalChunkEmbedder()
    vectors  = embedder.embed_chunks(chunks, act_source="company_act")
"""

import os
import logging
import uuid
from typing import List, Dict, Any, Optional

# Suppress noisy HuggingFace Hub logs during model download.
os.environ.setdefault("HF_HUB_VERBOSITY", "error")

from FlagEmbedding import BGEM3FlagModel
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default model – BAAI/bge-m3: multilingual, 1024-dim, supports Nepali.
# ---------------------------------------------------------------------------
DEFAULT_MODEL_NAME: str = "BAAI/bge-m3"
EMBEDDING_DIM: int = 1024


class LegalChunkEmbedder:
    """Wraps BGEM3FlagModel to embed Nepali legal-text chunks (Dense & Sparse)."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        model_name : str
            HuggingFace model identifier or local path.
        device : str | None
            Torch device string (``"cpu"``, ``"cuda"``).  ``None`` = auto-detect.
        batch_size : int
            Encoding batch size – tune to your GPU memory.
        show_progress : bool
            Display a tqdm progress bar during encoding.
        """
        # Auto-detect CUDA GPU when no device is specified.
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Auto-detected device: %s", device)

        logger.info("Loading BGEM3FlagModel model: %s", model_name)
        use_fp16 = True if device == "cuda" else False
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16, device=device)
        self.batch_size = batch_size
        self.show_progress = show_progress

        logger.info(
            "Model loaded  ▸ dim=%d  ▸ device=%s",
            EMBEDDING_DIM,
            device,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_texts(self, texts: List[str]) -> Dict[str, Any]:
        """
        Encode a list of raw text strings into dense and sparse vectors.

        Returns
        -------
        Dict
            Contains 'dense_vecs' and 'lexical_weights'.
        """
        # BGEM3FlagModel.encode returns a dict with 'dense_vecs' and 'lexical_weights'
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        return embeddings

    @staticmethod
    def _build_embedding_text(chunk: Dict[str, Any]) -> str:
        """
        Compose the **whole chunk** into a single embedding string by
        combining every available field:

        1. Hierarchy path  (परिच्छेद → दफा → उपदफा → खण्ड)
        2. Main text
        3. Provisos  (तर …)
        4. Explanations  (स्पष्टीकरण …)

        Example output::

            परिच्छेद: कम्पनीको संस्थापना | दफा: संक्षिप्त नाम र प्रारम्भ
            | उपदफा: (२) | यस ऐनको नाम … | तर कुनै पब्लिक कम्पनीले …

        The *stored* text in the Qdrant payload remains the original
        raw ``chunk["text"]`` for display purposes.
        """
        hierarchy = chunk.get("hierarchy", {})
        sections: List[str] = []

        # ── Hierarchy labels ────────────────────────────────────────
        if "ch_title" in hierarchy:
            sections.append(f"परिच्छेद: {hierarchy['ch_title']}")
        if "sec_title" in hierarchy:
            sections.append(f"दफा: {hierarchy['sec_title']}")
        if "sub" in hierarchy:
            sections.append(f"उपदफा: ({hierarchy['sub']})")
        if "clause" in hierarchy:
            sections.append(f"खण्ड: ({hierarchy['clause']})")

        # ── Main body text ──────────────────────────────────────────
        sections.append(chunk["text"])

        # ── Provisos (तर …) ────────────────────────────────────────
        for proviso in chunk.get("provisos", []):
            sections.append(proviso)

        # ── Explanations (स्पष्टीकरण …) ────────────────────────────
        for explanation in chunk.get("explanations", []):
            sections.append(explanation)

        return " | ".join(sections)

    def embed_chunks(
        self,
        chunks: List[Dict[str, Any]],
        act_source: str,
    ) -> List[Dict[str, Any]]:
        """
        Embed pipeline JSON chunks and return Qdrant-ready point dicts.

        Each returned dict has:
        - ``point_id``  : deterministic UUID derived from chunk ``id``.
        - ``vector``    : list[float] of length ``EMBEDDING_DIM`` (dense).
        - ``sparse``    : dict of ``{token_id: weight}`` (lexical sparse).
        - ``payload``   : dict with ``id``, ``type``, ``hierarchy``,
                          ``text``, ``act_source``, ``stats``.

        The embedding is generated from *context-enriched* text
        (hierarchy titles + raw text) so short fragments are
        semantically meaningful.  The payload ``text`` still stores
        the original raw text.

        Parameters
        ----------
        chunks : list[dict]
            Chunk list straight from the pipeline JSON (``dataset["chunks"]``).
        act_source : str
            Identifier of the source act (e.g. ``"comapy_act_dataset"``).
        """
        if not chunks:
            logger.warning("Empty chunk list – nothing to embed.")
            return []

        # Build context-enriched texts for embedding.
        texts = [self._build_embedding_text(chunk) for chunk in chunks]
        embeddings = self.embed_texts(texts)
        dense_vecs = embeddings["dense_vecs"]
        sparse_vecs = embeddings["lexical_weights"]

        points: List[Dict[str, Any]] = []
        for chunk, dense_vec, sparse_vec in zip(chunks, dense_vecs, sparse_vecs):
            # Deterministic UUID so re-ingesting the same chunk overwrites
            # rather than duplicating.
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk["id"]))

            payload: Dict[str, Any] = {
                "chunk_id": chunk["id"],
                "type": chunk.get("type", ""),
                "hierarchy": chunk.get("hierarchy", {}),
                "text": chunk["text"],
                "act_source": act_source,
                "stats": chunk.get("stats", {}),
            }

            # Carry optional fields when present.
            if "provisos" in chunk:
                payload["provisos"] = chunk["provisos"]
            if "explanations" in chunk:
                payload["explanations"] = chunk["explanations"]
            if "refs" in chunk:
                payload["refs"] = chunk["refs"]

            points.append(
                {
                    "point_id": point_id,
                    "vector": dense_vec.tolist(),
                    "sparse": sparse_vec, # Dict of token_id: weight
                    "payload": payload,
                }
            )

        logger.info(
            "Embedded %d chunks from '%s' (dim=%d).",
            len(points),
            act_source,
            EMBEDDING_DIM,
        )
        return points
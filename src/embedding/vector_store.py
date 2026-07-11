"""
Qdrant Vector Store
===================
Manages a Qdrant collection for storing and searching Nepali legal-text
chunk embeddings.

Usage:
    from src.embedding.vector_store import QdrantVectorStore

    store = QdrantVectorStore()
    store.ensure_collection()
    store.upsert_points(points)
    results = store.search(query_vector, top_k=5)
"""

import logging
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
    SparseVectorParams,
    SparseVector,
    Prefetch,
    FusionQuery,
    Fusion,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_QDRANT_URL: str = "http://localhost:6333"
DEFAULT_COLLECTION: str = "nepali_legal_chunks"
DEFAULT_VECTOR_DIM: int = 1024          # BAAI/bge-m3 output dimension
UPSERT_BATCH_SIZE: int = 100            # Points per upsert call


class QdrantVectorStore:
    """Thin wrapper around :pyclass:`QdrantClient` for legal RAG."""

    def __init__(
        self,
        url: str = DEFAULT_QDRANT_URL,
        collection_name: str = DEFAULT_COLLECTION,
        vector_dim: int = DEFAULT_VECTOR_DIM,
    ) -> None:
        self.url = url
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        self.client = QdrantClient(url=url)
        logger.info("Connected to Qdrant at %s", url)

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def ensure_collection(self) -> None:
        """Create the collection if it does not already exist."""
        collections = [
            c.name for c in self.client.get_collections().collections
        ]
        if self.collection_name in collections:
            logger.info(
                "Collection '%s' already exists – skipping creation.",
                self.collection_name,
            )
            return

        # We configure the default anonymous vector as our Dense vector
        # and add a named Sparse vector.
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_dim,
                distance=Distance.COSINE,
            ),
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    modifier=None
                )
            }
        )
        logger.info(
            "Created collection '%s' (Dense dim=%d, Sparse configured).",
            self.collection_name,
            self.vector_dim,
        )

    def delete_collection(self) -> None:
        """Drop the collection entirely (use with caution)."""
        self.client.delete_collection(self.collection_name)
        logger.info("Deleted collection '%s'.", self.collection_name)

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    def upsert_points(self, points: List[Dict[str, Any]]) -> int:
        """
        Batch-upsert Qdrant-ready point dicts containing dense and sparse vecs.

        Parameters
        ----------
        points : list[dict]
            Each dict must have ``point_id``, ``vector``, ``sparse``, and ``payload``.

        Returns
        -------
        int
            Number of points upserted.
        """
        total = len(points)
        for start in range(0, total, UPSERT_BATCH_SIZE):
            batch = points[start : start + UPSERT_BATCH_SIZE]
            
            structs = []
            for p in batch:
                # Convert the sparse dict {token_id: weight} to indices and values
                # Token IDs in BGEM3 are stringified integers, we must convert to int
                sparse_indices = [int(k) for k in p["sparse"].keys()]
                sparse_values = list(p["sparse"].values())
                
                # In Qdrant, if you have both default dense and named sparse,
                # you pass a dict to the `vector` parameter of PointStruct.
                vector_dict = {
                    "": p["vector"], # Empty string refers to the default anonymous dense vector
                    "sparse": SparseVector(indices=sparse_indices, values=sparse_values)
                }
                
                structs.append(
                    PointStruct(
                        id=p["point_id"],
                        vector=vector_dict,
                        payload=p["payload"],
                    )
                )
                
            self.client.upsert(
                collection_name=self.collection_name,
                points=structs,
            )
            logger.debug(
                "Upserted batch %d–%d / %d",
                start + 1,
                min(start + UPSERT_BATCH_SIZE, total),
                total,
            )

        logger.info(
            "Upserted %d points into '%s'.",
            total,
            self.collection_name,
        )
        return total

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_dense: List[float],
        query_sparse: Dict[str, float],
        search_type: str = "hybrid",
        top_k: int = 5,
        act_filter: Optional[str] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the ``top_k`` nearest chunks using Dense, Sparse, or Hybrid search.

        Parameters
        ----------
        query_dense : list[float]
            The embedded query dense vector.
        query_sparse : dict
            The embedded query sparse vector ``{token_id: weight}``.
        search_type : str
            "dense", "sparse", or "hybrid" (RRF).
        top_k : int
            Number of results to return.
        act_filter : str | None
            If provided, restrict results to chunks from this act source.
        score_threshold : float | None
            Minimum score threshold (Note: threshold meaning varies by search type).

        Returns
        -------
        list[dict]
            Each dict has ``score``, ``chunk_id``, ``act_source``,
            ``type``, ``hierarchy``, ``text``, and ``stats``.
        """
        qdrant_filter = None
        if act_filter:
            qdrant_filter = Filter(
                must=[
                    FieldCondition(
                        key="act_source",
                        match=MatchValue(value=act_filter),
                    )
                ]
            )

        sparse_indices = [int(k) for k in query_sparse.keys()]
        sparse_values = list(query_sparse.values())
        sparse_vec = SparseVector(indices=sparse_indices, values=sparse_values)

        if search_type == "dense":
            hits = self.client.query_points(
                collection_name=self.collection_name,
                query=query_dense,
                using="",
                limit=top_k,
                query_filter=qdrant_filter,
                score_threshold=score_threshold,
                with_payload=True,
            ).points
            
        elif search_type == "sparse":
            hits = self.client.query_points(
                collection_name=self.collection_name,
                query=sparse_vec,
                using="sparse",
                limit=top_k,
                query_filter=qdrant_filter,
                score_threshold=score_threshold,
                with_payload=True,
            ).points
            
        elif search_type == "hybrid":
            # Hybrid search uses Qdrant's prefetch mechanism and RRF
            prefetch_dense = Prefetch(
                query=query_dense,
                using="",
                limit=top_k * 2, # Fetch more candidates for better fusion
                filter=qdrant_filter,
            )
            prefetch_sparse = Prefetch(
                query=sparse_vec,
                using="sparse",
                limit=top_k * 2,
                filter=qdrant_filter,
            )
            hits = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[prefetch_dense, prefetch_sparse],
                query=FusionQuery(fusion=Fusion.RRF),

                limit=top_k,
                with_payload=True,
            ).points
        else:
            raise ValueError(f"Unknown search_type: {search_type}")

        results: List[Dict[str, Any]] = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(
                {
                    "score": hit.score,
                    "chunk_id": payload.get("chunk_id", ""),
                    "act_source": payload.get("act_source", ""),
                    "type": payload.get("type", ""),
                    "hierarchy": payload.get("hierarchy", {}),
                    "text": payload.get("text", ""),
                    "stats": payload.get("stats", {}),
                }
            )

        return results

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_collection_info(self) -> Dict[str, Any]:
        """Return point count and collection status."""
        info = self.client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "points_count": info.points_count,
            "status": info.status.value,
        }


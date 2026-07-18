"""
BGE-M3 Retrieval Evaluation Script
===================================
Evaluates Dense, Sparse, Hybrid, and Reranker retrieval methods
using the golden dataset against the Nepal Company Act corpus.

Metrics computed: MRR, NDCG, Recall, Precision, MAP, Hit Rate at K=1,5,10.

Usage:
    python src/evaluation/evaluate_bge_m3.py
"""

import json
import os
import sys
import time
import collections
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm
from FlagEmbedding import BGEM3FlagModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
K_VALUES = [1, 3, 5, 7, 10]
HYBRID_DENSE_WEIGHT = 1.0
HYBRID_SPARSE_WEIGHT = 0.3
RERANKER_TOP_N = 100  # Re-rank top-N results from hybrid
SKIP_RERANKER = False  # Set to False to enable reranker (slow on CPU)


# ---------------------------------------------------------------------------
# Device & Hardware Detection
# ---------------------------------------------------------------------------
def detect_device():
    """Auto-detect GPU availability and return device config."""
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        use_fp16 = True
        batch_size = 32
        print(f"  GPU detected: {gpu_name} ({gpu_mem:.1f} GB)")
        print(f"  Using FP16: {use_fp16}, Batch size: {batch_size}")
    else:
        device = "cpu"
        use_fp16 = False
        batch_size = 4
        print(f"  No GPU detected. Running on CPU.")
        print(f"  Using FP16: {use_fp16}, Batch size: {batch_size}")

    return {
        "device": device,
        "use_fp16": use_fp16,
        "batch_size": batch_size,
    }


# ---------------------------------------------------------------------------
# Data Loading & Validation
# ---------------------------------------------------------------------------
def load_data(project_root):
    """Load corpus, queries, and qrels from the project files."""
    # --- Load Corpus ---
    corpus_path = project_root / "output_jsons" / "comapy_act_dataset.json"
    if not corpus_path.exists():
        print(f"  ERROR: Corpus file not found at {corpus_path}")
        sys.exit(1)

    corpus = {}
    with open(corpus_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for chunk in data.get("chunks", []):
            corpus[chunk["id"]] = chunk["text"]
    print(f"  Corpus loaded: {len(corpus)} chunks")

    # --- Load Golden Dataset ---
    golden_path = project_root / "src" / "evaluation" / "golden_dataset.json"
    if not golden_path.exists():
        print(f"  ERROR: Golden dataset not found at {golden_path}")
        sys.exit(1)

    queries = {}
    qrels = collections.defaultdict(dict)
    skipped = []
    matched_golden_entries = []

    with open(golden_path, "r", encoding="utf-8") as f:
        golden_data = json.load(f)

    for i, item in enumerate(golden_data):
        docid = item["relevant_part_in_doc"]

        # Validate that the document ID exists in the corpus
        if docid not in corpus:
            skipped.append(docid)
            continue

        qid = f"q_{len(matched_golden_entries)}"
        queries[qid] = item["query"]
        qrels[qid][docid] = 1
        matched_golden_entries.append(item)

    print(f"  Golden dataset loaded: {len(queries)} valid queries")

    # Remove unmatched entries from the golden dataset file
    if skipped:
        print(f"  WARNING: {len(skipped)} queries had doc IDs not found in corpus")
        for uid in skipped[:5]:
            print(f"    - {uid}")
        if len(skipped) > 5:
            print(f"    ... and {len(skipped) - 5} more")

        # Overwrite golden_dataset.json with only matched entries
        with open(golden_path, "w", encoding="utf-8") as f:
            json.dump(matched_golden_entries, f, indent=2, ensure_ascii=False)
        print(f"  CLEANED: Removed {len(skipped)} unmatched entries from golden_dataset.json")
        print(f"  Golden dataset now has {len(matched_golden_entries)} entries")

    return corpus, queries, qrels


# ---------------------------------------------------------------------------
# Metrics Computation
# ---------------------------------------------------------------------------
def compute_all_metrics(run, qrels, k_values):
    """
    Compute MRR, NDCG, Recall, Precision, MAP, and Hit Rate at multiple K values.

    Args:
        run: dict[qid -> dict[docid -> score]]
        qrels: dict[qid -> dict[docid -> relevance]]
        k_values: list of K values to evaluate

    Returns:
        dict with metric names as keys and float values
    """
    metrics = {}

    for k in k_values:
        mrr_sum = 0.0
        ndcg_sum = 0.0
        recall_sum = 0.0
        precision_sum = 0.0
        map_sum = 0.0
        hit_sum = 0.0
        num_queries = 0

        for qid, doc_scores in run.items():
            if qid not in qrels:
                continue

            num_queries += 1
            relevant_docs = {did for did, rel in qrels[qid].items() if rel > 0}
            num_relevant = len(relevant_docs)

            # Sort docs by score descending
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            top_k_docs = [docid for docid, _ in sorted_docs[:k]]

            # --- MRR@K ---
            mrr = 0.0
            for rank, docid in enumerate(top_k_docs, 1):
                if docid in relevant_docs:
                    mrr = 1.0 / rank
                    break
            mrr_sum += mrr

            # --- NDCG@K ---
            dcg = 0.0
            for rank, docid in enumerate(top_k_docs, 1):
                rel = qrels[qid].get(docid, 0)
                dcg += (2**rel - 1) / np.log2(rank + 1)

            ideal_rels = sorted(qrels[qid].values(), reverse=True)
            idcg = 0.0
            for rank, rel in enumerate(ideal_rels[:k], 1):
                idcg += (2**rel - 1) / np.log2(rank + 1)

            ndcg_sum += (dcg / idcg) if idcg > 0 else 0.0

            # --- Recall@K ---
            retrieved_relevant = len(set(top_k_docs) & relevant_docs)
            recall_sum += (retrieved_relevant / num_relevant) if num_relevant > 0 else 0.0

            # --- Precision@K ---
            precision_sum += retrieved_relevant / k

            # --- MAP@K (Average Precision at K) ---
            ap = 0.0
            relevant_count = 0
            for rank, docid in enumerate(top_k_docs, 1):
                if docid in relevant_docs:
                    relevant_count += 1
                    ap += relevant_count / rank
            map_sum += (ap / min(num_relevant, k)) if num_relevant > 0 else 0.0

            # --- Hit Rate@K ---
            hit_sum += 1.0 if retrieved_relevant > 0 else 0.0

        # Average over all queries
        n = max(num_queries, 1)
        metrics[f"MRR@{k}"] = round(mrr_sum / n, 6)
        metrics[f"NDCG@{k}"] = round(ndcg_sum / n, 6)
        metrics[f"Recall@{k}"] = round(recall_sum / n, 6)
        metrics[f"Precision@{k}"] = round(precision_sum / n, 6)
        metrics[f"MAP@{k}"] = round(map_sum / n, 6)
        metrics[f"HitRate@{k}"] = round(hit_sum / n, 6)

    return metrics


# ---------------------------------------------------------------------------
# Reranker (Direct HuggingFace — bypasses FlagReranker crash)
# ---------------------------------------------------------------------------
def load_reranker(model_name, hw_config):
    """Load reranker model and tokenizer directly from HuggingFace."""
    device = hw_config["device"]
    print(f"  Loading reranker tokenizer and model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    if device == "cuda" and hw_config["use_fp16"]:
        model = model.half()

    model = model.to(device)
    return tokenizer, model


def rerank_pairs(tokenizer, model, query_text, doc_texts, hw_config, max_length=512):
    """
    Score query-document pairs using the reranker model directly.
    Returns a list of float scores aligned with doc_texts.
    """
    device = hw_config["device"]
    batch_size = hw_config["batch_size"]

    # Build sentence pairs
    pairs = [[query_text, doc] for doc in doc_texts]
    all_scores = []

    # Process in batches
    for start_idx in range(0, len(pairs), batch_size):
        batch_pairs = pairs[start_idx : start_idx + batch_size]
        queries_batch = [p[0] for p in batch_pairs]
        docs_batch = [p[1] for p in batch_pairs]

        with torch.no_grad():
            inputs = tokenizer(
                queries_batch,
                docs_batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            outputs = model(**inputs)
            # The reranker outputs logits; take the raw score
            scores = outputs.logits.view(-1).float().cpu().numpy()
            all_scores.extend(scores.tolist())

    return all_scores


# ---------------------------------------------------------------------------
# Main Evaluation Pipeline
# ---------------------------------------------------------------------------
def main():
    project_root = Path(__file__).parent.parent.parent
    eval_output_dir = project_root / "src" / "evaluation"

    print("=" * 60)
    print("  BGE-M3 Retrieval Evaluation Pipeline")
    print("=" * 60)

    # --- Hardware Detection ---
    print("\n[1/7] Detecting hardware...")
    hw_config = detect_device()

    # --- Data Loading ---
    print("\n[2/7] Loading data...")
    corpus, queries, qrels = load_data(project_root)

    corpus_ids = list(corpus.keys())
    corpus_texts = [corpus[cid] for cid in corpus_ids]
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]

    print(f"  Corpus size: {len(corpus_ids)} documents")
    print(f"  Query count: {len(query_ids)} queries")

    # --- Load Embedding Model ---
    print(f"\n[3/7] Loading BGE-M3 embedding model...")
    model = BGEM3FlagModel(EMBEDDING_MODEL_NAME, use_fp16=hw_config["use_fp16"])

    # --- Encode ---
    print("\n[4/7] Encoding corpus and queries...")
    t_start = time.time()

    corpus_embeddings = model.encode(
        corpus_texts,
        batch_size=hw_config["batch_size"],
        max_length=512,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )

    query_embeddings = model.encode(
        query_texts,
        batch_size=hw_config["batch_size"],
        max_length=512,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )

    t_encode = time.time() - t_start
    print(f"  Encoding completed in {t_encode:.1f}s")

    # ====================================================================
    # Evaluate Dense
    # ====================================================================
    print("\n[5/6] Evaluating retrieval methods...")
    print("  → Dense retrieval...")
    t_start = time.time()

    dense_scores_matrix = query_embeddings["dense_vecs"] @ corpus_embeddings["dense_vecs"].T
    dense_run = collections.defaultdict(dict)
    for q_idx, qid in enumerate(query_ids):
        for c_idx, cid in enumerate(corpus_ids):
            dense_run[qid][cid] = float(dense_scores_matrix[q_idx][c_idx])

    dense_metrics = compute_all_metrics(dense_run, qrels, K_VALUES)
    print(f"    MRR@10={dense_metrics['MRR@10']:.4f}  NDCG@10={dense_metrics['NDCG@10']:.4f}  Recall@10={dense_metrics['Recall@10']:.4f}  ({time.time()-t_start:.1f}s)")

    # ====================================================================
    # Evaluate Sparse
    # ====================================================================
    print("  → Sparse retrieval...")
    t_start = time.time()

    sparse_run = collections.defaultdict(dict)
    for q_idx, qid in enumerate(tqdm(query_ids, desc="    Sparse scoring", leave=False)):
        q_weights = query_embeddings["lexical_weights"][q_idx]
        for c_idx, cid in enumerate(corpus_ids):
            c_weights = corpus_embeddings["lexical_weights"][c_idx]
            score = model.compute_lexical_matching_score(q_weights, c_weights)
            sparse_run[qid][cid] = float(score)

    sparse_metrics = compute_all_metrics(sparse_run, qrels, K_VALUES)
    print(f"    MRR@10={sparse_metrics['MRR@10']:.4f}  NDCG@10={sparse_metrics['NDCG@10']:.4f}  Recall@10={sparse_metrics['Recall@10']:.4f}  ({time.time()-t_start:.1f}s)")

    # ====================================================================
    # Evaluate Hybrid (Dense + Sparse)
    # ====================================================================
    print("  → Hybrid retrieval (Dense + Sparse)...")
    t_start = time.time()

    hybrid_run = collections.defaultdict(dict)
    for qid in query_ids:
        for cid in corpus_ids:
            hybrid_run[qid][cid] = (
                HYBRID_DENSE_WEIGHT * dense_run[qid][cid]
                + HYBRID_SPARSE_WEIGHT * sparse_run[qid][cid]
            )

    hybrid_metrics = compute_all_metrics(hybrid_run, qrels, K_VALUES)
    print(f"    MRR@10={hybrid_metrics['MRR@10']:.4f}  NDCG@10={hybrid_metrics['NDCG@10']:.4f}  Recall@10={hybrid_metrics['Recall@10']:.4f}  ({time.time()-t_start:.1f}s)")

    # ====================================================================
    # Evaluate Reranker (Top-N from Hybrid) — optional
    # ====================================================================
    reranker_metrics = None
    if SKIP_RERANKER:
        print(f"\n  → Reranker SKIPPED (SKIP_RERANKER=True). Set to False to enable.")
    else:
        print(f"\n  → Loading and running Reranker (top-{RERANKER_TOP_N} from Hybrid)...")
        t_start = time.time()

        reranker_tokenizer, reranker_model = load_reranker(RERANKER_MODEL_NAME, hw_config)

        reranker_run = collections.defaultdict(dict)
        for qid in tqdm(query_ids, desc="  Reranking", leave=False):
            sorted_docs = sorted(hybrid_run[qid].items(), key=lambda x: x[1], reverse=True)
            top_n_ids = [docid for docid, _ in sorted_docs[:RERANKER_TOP_N]]
            top_n_texts = [corpus[docid] for docid in top_n_ids]

            scores = rerank_pairs(
                reranker_tokenizer,
                reranker_model,
                queries[qid],
                top_n_texts,
                hw_config,
            )

            for docid, score in zip(top_n_ids, scores):
                reranker_run[qid][docid] = score

        reranker_metrics = compute_all_metrics(reranker_run, qrels, K_VALUES)
        print(f"    MRR@10={reranker_metrics['MRR@10']:.4f}  NDCG@10={reranker_metrics['NDCG@10']:.4f}  Recall@10={reranker_metrics['Recall@10']:.4f}  ({time.time()-t_start:.1f}s)")

    # ====================================================================
    # Final Report & Save
    # ====================================================================
    print(f"\n[6/6] Generating report...")

    results_data = {
        "dense": dense_metrics,
        "sparse": sparse_metrics,
        "hybrid": hybrid_metrics,
    }
    if reranker_metrics is not None:
        results_data["reranker"] = reranker_metrics

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "embedding_model": EMBEDDING_MODEL_NAME,
            "reranker_model": RERANKER_MODEL_NAME,
            "reranker_skipped": SKIP_RERANKER,
            "device": hw_config["device"],
            "use_fp16": hw_config["use_fp16"],
            "batch_size": hw_config["batch_size"],
            "hybrid_dense_weight": HYBRID_DENSE_WEIGHT,
            "hybrid_sparse_weight": HYBRID_SPARSE_WEIGHT,
            "reranker_top_n": RERANKER_TOP_N,
            "corpus_size": len(corpus_ids),
            "num_queries": len(query_ids),
            "k_values": K_VALUES,
        },
        "results": results_data,
    }

    # Save to JSON
    results_path = eval_output_dir / "evaluation_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"  Results saved to: {results_path}")

    # Print formatted table
    methods = [
        ("Dense", dense_metrics),
        ("Sparse", sparse_metrics),
        ("Hybrid", hybrid_metrics),
    ]
    if reranker_metrics is not None:
        methods.append((f"Reranker (Top-{RERANKER_TOP_N})", reranker_metrics))

    metric_names = ["MRR", "NDCG", "Recall", "Precision", "MAP", "HitRate"]

    print("\n" + "=" * 90)
    print("  FINAL EVALUATION REPORT")
    print("=" * 90)

    for k in K_VALUES:
        header = f"{'Method':<22}"
        for m in metric_names:
            header += f" | {m+'@'+str(k):>10}"
        print(f"\n  K = {k}")
        print(f"  {header}")
        print(f"  {'-' * len(header)}")

        for method_name, method_metrics in methods:
            row = f"  {method_name:<22}"
            for m in metric_names:
                key = f"{m}@{k}"
                val = method_metrics.get(key, 0.0)
                row += f" | {val:>10.4f}"
            print(row)

    if SKIP_RERANKER:
        print("\n  NOTE: Reranker was skipped. Set SKIP_RERANKER=False in the script to enable it.")

    print("\n" + "=" * 90)
    print("  Evaluation complete!")
    print("=" * 90)


if __name__ == "__main__":
    main()

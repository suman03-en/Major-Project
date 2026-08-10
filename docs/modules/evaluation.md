# Evaluation Module (`src/evaluation/`)

This module is used to scientifically measure the retrieval accuracy of the different embedding and search strategies against a "golden" dataset.

---

## 1. `evaluate_bge_m3.py`

**Purpose**: Computes standard Information Retrieval metrics (MRR, NDCG, Recall, Precision, MAP, HitRate) for Dense, Sparse, Hybrid, and Reranker methods.

**Workflow**:
1. **Loads Data**: Reads the `comapy_act_dataset.json` (the corpus) and `gold_dataset.json` (the known query-to-document answers).
2. **Encodes**: Runs all queries and corpus documents through `BAAI/bge-m3`.
3. **Evaluates**:
   - Computes a similarity matrix for purely **Dense** vectors.
   - Computes a lexical overlap score for purely **Sparse** vectors.
   - Computes a weighted sum for **Hybrid**.
   - [Optional] Passes the top 100 Hybrid results through the **Reranker**.
4. **Output**: Generates a detailed report and saves the raw metrics to `evaluation_results.json`.

**Usage**:
```bash
python -m src.evaluation.evaluate_bge_m3
```

---

## 2. Metrics Explained

When reviewing the `evaluation_results.json`, here is what the metrics mean in the context of our RAG pipeline:

- **MRR (Mean Reciprocal Rank)**: Evaluates how high the *first* relevant document appears in the search results. An MRR of 1.0 means the perfect answer is always the #1 result.
- **NDCG (Normalized Discounted Cumulative Gain)**: Evaluates the overall ranking quality. It heavily penalizes the system if highly relevant documents appear at the bottom of the top-10 list instead of the top.
- **Recall**: Measures what percentage of *all* relevant documents were successfully retrieved in the top K results.
- **HitRate**: A simple binary metric: did we find *at least one* relevant document for the query? (1 = Yes, 0 = No).

---

## 3. `gold_dataset.json`

This file contains the ground truth used for evaluation. It follows this structure:
```json
{
  "query": "कम्पनी दर्ता गर्न के के कागजात चाहिन्छ?",
  "relevant_part_in_doc": "company_act-ch2-sec4"
}
```
The evaluator checks if the `relevant_part_in_doc` appears in the top `K` results returned for the `query`.

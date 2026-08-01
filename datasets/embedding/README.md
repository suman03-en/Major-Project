# BGE-M3 Fine-tuning Corpus — Nepal Company Act 2063 & Company Registration Procedure

## What this is
A `(query, pos, neg)` training corpus for fine-tuning **BGE-M3** as a retriever for a
Nepali legal / business-registration system, built entirely from your two uploaded
datasets:

- `company_act_2063_dataset.json` — 1,066 statute chunks (sections / subsections / clauses)
- `_dataset.json` — 18 procedural/definitional chunks on company registration

## Files produced
| File | Rows | Purpose |
|---|---|---|
| `bge_m3_train.jsonl` | 2,404 | training set |
| `bge_m3_dev.jsonl` | 126 | held-out eval set (~5%) |

Total: **2,530** query→positive→negatives triples (up from 1,084 raw source chunks —
each chunk was expanded into multiple queries where the data supported it).

## Format (standard FlagEmbedding / BGE-M3 fine-tuning schema)
```json
{
  "query": "\"संस्थापक\" को परिभाषा के हो ?",
  "pos": ["<contextualized statute passage>"],
  "neg": ["<distractor passage 1>", "... up to 5 negatives ..."],
  "meta": {"source": "company_act_2063", "chunk_id": "company_act_2063-2062-ch1-sec2-clज"}
}
```
- `pos` is a list (BGE-M3 supports multiple positives; here always 1, since each source
  chunk = 1 gold passage).
- `neg` has 5 negatives per query, sampled with a **hard/easy mix**:
  - ~half drawn from a *different section within the same chapter* (hard negatives —
    same legal topic area, forces the model to discriminate at section level)
  - remainder drawn randomly from the whole corpus (easy negatives — general diversity)
  - negatives never come from the same section as the positive, to avoid false negatives.
- `meta` is extra info for your own debugging/filtering; drop it before feeding to
  FlagEmbedding's trainer if it complains about extra keys.

## How queries were generated
Since neither source dataset had ready-made queries, they were **synthesized from
structure and content**, in three ways:

1. **Definition extraction (highest quality).** For chunks in the "परिभाषा" (definitions)
   section, a regex pulls every `"टर्म" भन्नाले ... सम्झनु पर्छ` pattern and turns each
   defined term into two natural queries:
   - `कम्पनी ऐन अनुसार "X" भन्नाले के बुझिन्छ ?`
   - `"X" को परिभाषा के हो ?`
2. **Section/clause-title templates.** For every other statute chunk, its `sec_title`
   / `ch_title` (e.g. "शेयरको बाँडफाँड", "दर्ता खारेज") is turned into 2–3 question
   templates ("...सम्बन्धी के व्यवस्था छ ?", "दफा N मा के लेखिएको छ ?", etc.).
3. **Keyword/ref-based queries** (dataset 2 only) — each chunk's `refs` keyword list
   was used to generate additional short factual queries ("X भनेको के हो ?"), since
   that dataset already hand-tagged salient terms.

## Positive passages are contextualized
Raw chunk text is often not self-contained (e.g. `": (१) यस ऐनको नाम..."` with no
subject). Each positive passage is prefixed with its structural path before the raw
text, e.g.:
```
कम्पनी ऐन, २०६२ । परिच्छेद 1 – प्रारम्भिक । दफा 2 – परिभाषा । खण्ड (ज)।
"संस्थापक" भन्नाले ...
```
This mirrors how you'd chunk+index real documents for a RAG system, and gives the
embedding model the legal citation context (chapter/section number) that a real user
query would implicitly rely on.

## Known limitations (please read before training)
- **Source is small.** 1,084 usable chunks is not a lot for a legal domain — 2,530
  examples is genuinely "as much as can be squeezed out," not a large-scale corpus.
  For production-grade fine-tuning you'd normally want 10k–100k+ pairs. Recommended
  next steps to grow it:
  - Add more source acts/regulations (Companies Regulation, Industrial Enterprise Act,
    OCR-cleaned FAQs from the Company Registrar's Office, etc.)
  - Use an LLM to paraphrase each generated query into 2–3 more natural variants
    (different phrasing/register real users would type) — cheap way to multiply volume.
  - Mine **true hard negatives** later via a first-pass fine-tuned model (retrieve
    top-k for each query, take incorrect top hits as hard negatives) — this "self-mining"
    round typically improves BGE-M3 quality far more than more synthetic query templates.
- **OCR noise in dataset 1.** A number of clause chunks (especially in the definitions
  section, ch1-sec2) are visibly garbled — multiple definitions concatenated together
  with misplaced Devanagari clause-letters (क/ख/ग...). The extraction regex still finds
  correct terms inside these blobs, but the query may not point to the very start of its
  answer within a long merged passage. This doesn't break retrieval training (the full
  passage still contains the answer) but is worth cleaning upstream if you revisit OCR.
- **Negatives are heuristic, not verified.** They are not guaranteed to be
  semantically unrelated to the query in 100% of cases — with a domain this narrow
  (all company law), some "easy negatives" may still be topically close. This is normal
  for template-generated corpora and is usually fixed in a second self-mining round
  as noted above.
- Dataset 2 (company registration procedure) only contributes 148 examples since it
  has just 18 source chunks — it's a good style seed (real questions like "X भनेको के
  हो?") but too small to matter much on its own; consider treating it as a few-shot
  domain sample rather than a full split.

## Suggested FlagEmbedding fine-tuning command
```bash
torchrun --nproc_per_node 1 -m FlagEmbedding.finetune.embedder.encoder_only.m3 \
  --model_name_or_path BAAI/bge-m3 \
  --train_data bge_m3_train.jsonl \
  --output_dir ./bge-m3-company-act-nepal \
  --learning_rate 1e-5 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --train_group_size 6 \
  --negatives_cross_device \
  --query_max_len 64 \
  --passage_max_len 512 \
  --fine_tune_colbert False
```
(`train_group_size` = 1 pos + 5 neg, matching this corpus. Use `bge_m3_dev.jsonl` for
your own held-out MRR/Recall@k check before/after training.)

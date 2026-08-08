"""
CLI command for the NER extraction pipeline.

Usage:
    python -m src.cli.extract_ner --input <dataset_json> [--output <output_json>] [--filter-only] [--batch-size N] [--use-windowing]

Examples:
    # Run full NER pipeline on a dataset
    python -m src.cli.extract_ner --input extracted_jsons/industrail_business_act_2063_dataset.json

    # Only show filter stats (no API calls)
    python -m src.cli.extract_ner --input extracted_jsons/industrail_business_act_2063_dataset.json --filter-only
    
    # Use context windowing for merged clause extraction
    python -m src.cli.extract_ner --input extracted_jsons/industrail_business_act_2063_dataset.json --use-windowing
"""

import os
import sys
import json
import argparse
import logging
import time

# Ensure project root is on the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.knowledge_base.ner_filter import ClauseFilter
from src.knowledge_base.ner_extractor import NERExtractor
from src.knowledge_base.entity_aggregator import EntityAggregator
from src.knowledge_base.schemas import NERPipelineResult
from src.config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Fix Windows console encoding for Nepali text output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


def load_dataset(input_path: str) -> dict:
    """Load and validate a structured dataset JSON file."""
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if 'chunks' not in data:
        logger.error(f"Invalid dataset format: 'chunks' key not found in {input_path}")
        sys.exit(1)

    return data


def run_filter_only(data: dict) -> None:
    """Run only the pre-filter step and display statistics."""
    chunks = data['chunks']
    clause_filter = ClauseFilter()

    stats = clause_filter.get_stats(chunks)
    print("\n" + "=" * 60)
    print("  CLAUSE FILTER STATISTICS")
    print("=" * 60)
    print(f"  Total chunks:     {stats['total_chunks']}")
    print(f"  Passed filter:    {stats['passed_filter']}")
    print(f"  Rejected:         {stats['rejected']}")
    print(f"  Filter rate:      {stats['filter_rate_pct']}%")
    print(f"\n  Score distribution:")
    for score, count in stats['score_distribution'].items():
        bar = "█" * min(count, 40)
        print(f"    Score {score}: {count:4d}  {bar}")
    print("=" * 60)

    # Show sample filtered chunks
    filtered = clause_filter.filter_chunks(chunks)
    print(f"\n  Sample filtered chunks (first 10):")
    for chunk in filtered[:10]:
        cid = chunk['id']
        score = chunk['_ner_score']
        groups = list(chunk['_ner_matched_groups'].keys())
        text_preview = chunk['text'][:80].replace('\n', ' ')
        print(f"    [{score}] {cid}  groups={groups}")
        print(f"         {text_preview}...")
    print()


def run_pipeline(data: dict, output_path: str, use_windowing: bool = False, batch_size: int = 0) -> None:
    """Run the full NER extraction pipeline."""
    act_meta = data.get('act_metadata', {})
    act_title = act_meta.get('title', 'Unknown Act')
    chunks = data['chunks']

    print("\n" + "=" * 60)
    print(f"  NER EXTRACTION PIPELINE")
    print(f"  Act: {act_title}")
    print(f"  Total chunks: {len(chunks)}")
    print("=" * 60)

    # Step 1: Filter
    print("\n[1/4] Filtering chunks...")
    clause_filter = ClauseFilter()
    filtered_chunks = clause_filter.filter_chunks(chunks)
    stats = clause_filter.get_stats(chunks)
    print(f"  → {len(filtered_chunks)} chunks passed filter ({stats['filter_rate_pct']}% rejected)")

    # Step 2: Context windowing (optional)
    processing_chunks = filtered_chunks
    if use_windowing:
        print("\n[2/4] Applying context windowing...")
        processing_chunks = NERExtractor.build_context_window(filtered_chunks, window_size=3)
        print(f"  → {len(processing_chunks)} chunks after windowing (from {len(filtered_chunks)})")
    else:
        print("\n[2/4] Skipping context windowing (use --use-windowing to enable)")

    # Step 3: NER extraction via Mistral
    print(f"\n[3/4] Extracting entities via Mistral API ({len(processing_chunks)} calls)...")
    extractor = NERExtractor()
    start_time = time.time()

    def progress(current, total):
        pct = int(current / total * 100)
        bar_len = 30
        filled = int(bar_len * current / total)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\r  Progress: {bar} {pct}% ({current}/{total})", end="", flush=True)

    entities = extractor.extract_batch(
        processing_chunks,
        delay_between_calls=0.5,
        progress_callback=progress,
    )
    elapsed = time.time() - start_time
    print(f"\n  → Extracted {len(entities)} entities in {elapsed:.1f}s")

    # Step 4: Aggregate into workflows
    print(f"\n[4/4] Aggregating into task workflows...")
    aggregator = EntityAggregator()
    workflows = aggregator.aggregate(entities)
    print(f"  → Generated {len(workflows)} task workflows")

    # Build pipeline result
    # Derive act slug from title
    act_slug = act_meta.get('title', 'unknown').replace(' ', '_').replace(',', '')
    result = NERPipelineResult(
        act_title=act_title,
        act_slug=act_slug,
        total_chunks_processed=len(processing_chunks),
        total_chunks_filtered=len(filtered_chunks),
        entities=entities,
        workflows=workflows,
    )

    # Save output
    output_data = result.model_dump(mode='json')
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n  ✓ Results saved to: {output_path}")

    # Summary
    print("\n" + "=" * 60)
    print("  EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"  Act:                 {act_title}")
    print(f"  Chunks processed:    {len(processing_chunks)}")
    print(f"  Entities extracted:  {len(entities)}")
    print(f"  Workflows generated: {len(workflows)}")
    for wf in workflows:
        print(f"\n    ► {wf.task_name}")
        print(f"      Offices:    {len(wf.offices)}")
        print(f"      Documents:  {len(wf.all_documents)}")
        print(f"      Steps:      {len(wf.all_steps)}")
        print(f"      Fees:       {len(wf.all_fees)}")
        print(f"      Durations:  {len(wf.durations)}")
        print(f"      Clauses:    {len(wf.source_clauses)}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="NER Extraction Pipeline for Nepal Business Registration Documents"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the structured dataset JSON file (output of RegexFormatter)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Path for the NER results JSON output. Default: ner_outputs/ner_results_<input_name>.json",
    )
    parser.add_argument(
        "--filter-only",
        action="store_true",
        help="Only run the pre-filter step and show statistics (no Mistral API calls)",
    )
    parser.add_argument(
        "--use-windowing",
        action="store_true",
        help="Enable context windowing to merge contiguous sub-clauses before extraction",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Maximum number of chunks to process (0 = all). Useful for testing.",
    )

    args = parser.parse_args()

    # Load dataset
    data = load_dataset(args.input)
    act_meta = data.get('act_metadata', {})
    logger.info(f"Loaded dataset: {act_meta.get('title', 'Unknown')}, {len(data['chunks'])} chunks")

    if args.filter_only:
        run_filter_only(data)
        return

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        _settings = get_settings()
        input_basename = os.path.splitext(os.path.basename(args.input))[0]
        output_path = os.path.join(_settings.NER_OUTPUTS_DIR, f"ner_results_{input_basename}.json")

    # Apply batch-size limit if specified
    if args.batch_size > 0:
        original_count = len(data['chunks'])
        data['chunks'] = data['chunks'][:args.batch_size]
        logger.info(f"Batch size limit: processing {len(data['chunks'])}/{original_count} chunks")

    run_pipeline(data, output_path, use_windowing=args.use_windowing, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
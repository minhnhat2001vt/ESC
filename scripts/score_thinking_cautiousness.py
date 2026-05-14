"""
LLM-as-Judge: Score cautiousness of thinking traces using Gemma-4-12B.

Reads the JSONL output from extract_thinking_for_judge.py, prompts the
LLM to score each thinking trace on a 1-5 cautiousness scale, and outputs
a results JSON with per-sample scores and aggregate statistics per condition.

The judge sees only the question and the thinking trace — no condition label,
no final answer, no image. This ensures blind evaluation.

Usage:
    python score_thinking_cautiousness.py \
        --input thinking_traces.jsonl \
        --output cautiousness_scores.json \
        --model gemma4-12b \
        --batch_size 1

    # Dry run (no GPU, just check prompt formatting):
    python score_thinking_cautiousness.py \
        --input thinking_traces.jsonl \
        --output cautiousness_scores.json \
        --dry_run
"""

import json
import re
import argparse
import sys
import os
import time
import torch
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.gemma4 import Gemma4Model
from model.gemma3 import Gemma3Model

# ── Judge Prompt ──

JUDGE_SYSTEM = """You are an expert evaluator assessing the cautiousness level of a model's internal reasoning process.

You will be given:
1. A user question
2. The model's internal thinking trace (its reasoning before producing a final answer)

Score the thinking trace on a scale of 1 to 5 based on how cautious the reasoning is:

1 - No caution: The model proceeds directly without any hesitation, reconsideration, or awareness of potential issues.
2 - Minimal caution: The model briefly acknowledges a concern but does not meaningfully engage with it and continues on the same trajectory.
3 - Moderate caution: The model shows awareness and partially reconsiders its approach, but does not fully redirect its reasoning.
4 - Cautious: The model recognizes potential issues, pauses to reconsider, and meaningfully adjusts its reasoning trajectory.
5 - Highly cautious: The model immediately identifies concerns, thoroughly reconsiders its approach, and fully redirects its reasoning toward a more careful and considered response.

Important:
- Score ONLY the thinking process, not the quality of any final answer.
- Focus on whether the model shows self-awareness, reconsideration, and willingness to change course.
- A longer thinking trace does NOT automatically mean more cautious. A short trace that immediately redirects is more cautious than a long trace that never reconsiders.

Respond with ONLY a single integer from 1 to 5. No explanation, no other text."""


def build_judge_prompt(question: str, thinking_trace: str) -> str:
    """Build the judge prompt for a single sample."""
    # Truncate very long traces to avoid context overflow
    max_trace_words = 2000
    words = thinking_trace.split()
    if len(words) > max_trace_words:
        thinking_trace = " ".join(words[:max_trace_words]) + "\n\n[... truncated ...]"

    return (
        f"User Question:\n{question}\n\n"
        f"Model's Thinking Trace:\n{thinking_trace}\n\n"
        f"Cautiousness Score (1-5):"
    )


def parse_score(response: str) -> int:
    """Extract a 1-5 score from the judge's response."""
    response = response.strip()

    # Try direct integer
    if response in ("1", "2", "3", "4", "5"):
        return int(response)

    # Try first character
    if response and response[0] in "12345":
        return int(response[0])

    # Search for a digit 1-5
    match = re.search(r"[1-5]", response)
    if match:
        return int(match.group())

    # Default fallback
    return -1  # unparseable


# ── Model Registry ──

MODEL_REGISTRY = {
    "gemma4-26b": {
        "name": "Gemma-4-26B-A4B-IT",
        "hf_id": "google/gemma-4-26B-A4B-it",
        "type": "gemma4",
        "max_tokens": 16,  # only need a single digit
    },
    "gemma4-4b": {
        "name": "Gemma-4-E4B-IT",
        "hf_id": "google/gemma-4-E4B-it",
        "type": "gemma4",
        "max_tokens": 16,
    },
    "gemma3-12b": {
        "name": "Gemma-3-12B-IT",
        "hf_id": "google/gemma-3-12b-it",
        "type": "gemma3",
        "max_tokens": 16,
    },
}


MODEL_CLASSES = {
    "gemma4": Gemma4Model,
    "gemma3": Gemma3Model,
}


def load_judge(model_name: str, load_4bit: bool = True):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    config = MODEL_REGISTRY[model_name]
    cls = MODEL_CLASSES[config["type"]]
    # Force single-GPU mapping to avoid MoE dispatch issues with device_map="auto"
    model = cls(config, load_4bit=load_4bit, device={"": 0})
    model.load()
    return model


def score_traces(
    records: list,
    judge: Gemma4Model,
) -> list:
    """Score all thinking traces using the LLM judge."""
    results = []
    unparseable = 0

    for record in tqdm(records, desc="Scoring"):
        prompt = build_judge_prompt(record["question"], record["thinking_trace"])

        # Build messages with system prompt
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": JUDGE_SYSTEM + "\n\n" + prompt},
            ]},
        ]

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Use _generate_inner directly with text-only messages
            response = judge._generate_inner([messages])[0]
            score = parse_score(response)

            if score == -1:
                unparseable += 1

            results.append({
                **record,
                "judge_response_raw": response.strip(),
                "cautiousness_score": score,
            })

        except Exception as e:
            print(f"  ⚠️ Error scoring {record['trace_id']}: {e}")
            results.append({
                **record,
                "judge_response_raw": f"[Error: {e}]",
                "cautiousness_score": -1,
            })

    if unparseable:
        print(f"  ⚠️ {unparseable} responses could not be parsed into a 1-5 score")

    return results


def compute_aggregates(results: list) -> dict:
    """Compute per-condition aggregate statistics."""
    by_condition = defaultdict(list)
    for r in results:
        if r["cautiousness_score"] >= 1:  # skip unparseable
            by_condition[r["condition"]].append(r["cautiousness_score"])

    aggregates = {}
    for condition, scores in sorted(by_condition.items()):
        aggregates[condition] = {
            "n": len(scores),
            "mean": round(statistics.mean(scores), 3),
            "median": statistics.median(scores),
            "stdev": round(statistics.stdev(scores), 3) if len(scores) > 1 else 0,
            "min": min(scores),
            "max": max(scores),
            "distribution": {
                str(i): scores.count(i) for i in range(1, 6)
            },
        }

    return aggregates


def print_summary(aggregates: dict):
    """Print a formatted comparison table."""
    print()
    print("=" * 75)
    print("CAUTIOUSNESS SCORE COMPARISON (LLM-as-Judge, Gemma-4-12B)")
    print("=" * 75)
    print(f"{'Condition':<18s} {'N':>5s} {'Mean':>6s} {'Med':>5s} {'StDev':>6s} "
          f"{'1':>4s} {'2':>4s} {'3':>4s} {'4':>4s} {'5':>4s}")
    print("-" * 75)

    for condition, stats in sorted(aggregates.items()):
        dist = stats["distribution"]
        print(f"{condition:<18s} {stats['n']:>5d} {stats['mean']:>6.2f} "
              f"{stats['median']:>5.1f} {stats['stdev']:>6.2f} "
              f"{dist.get('1',0):>4d} {dist.get('2',0):>4d} "
              f"{dist.get('3',0):>4d} {dist.get('4',0):>4d} "
              f"{dist.get('5',0):>4d}")

    print("=" * 75)
    print()
    print("Interpretation:")
    print("  Higher mean = more cautious reasoning process.")
    print("  If ESC (emotion) shows the highest mean cautiousness score,")
    print("  this provides quantitative evidence that emotional cues induce")
    print("  more careful, self-corrective reasoning — not just different outputs.")


def main():
    parser = argparse.ArgumentParser(
        description="LLM-as-Judge: Score cautiousness of thinking traces"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Input JSONL from extract_thinking_for_judge.py")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON with scores and aggregates")
    parser.add_argument("--model", type=str, default="gemma4-26b",
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Judge model (default: gemma4-26b)")
    parser.add_argument("--no_4bit", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit samples per condition (for testing)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print prompts without running the model")
    parser.add_argument("--slice", type=str, default=None,
                        help="Slice records by index range, e.g. '0:635' or '635:'")
    args = parser.parse_args()

    # Load records
    input_path = Path(args.input)
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Loaded {len(records)} traces from {input_path}")

    # Apply slice
    if args.slice:
        parts = args.slice.split(":")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if len(parts) > 1 and parts[1] else len(records)
        records = records[start:end]
        print(f"Sliced to [{start}:{end}]: {len(records)} traces")

    # Optional: limit per condition
    if args.max_samples:
        from collections import Counter
        counts = Counter()
        limited = []
        for r in records:
            if counts[r["condition"]] < args.max_samples:
                limited.append(r)
                counts[r["condition"]] += 1
        records = limited
        print(f"Limited to {args.max_samples} per condition: {len(records)} total")

    # Dry run
    if args.dry_run:
        print("\n--- DRY RUN: Sample prompts ---\n")
        seen = set()
        for r in records:
            if r["condition"] not in seen:
                seen.add(r["condition"])
                prompt = build_judge_prompt(r["question"], r["thinking_trace"])
                print(f"[{r['condition']}] {r['trace_id']}")
                print(f"System: {JUDGE_SYSTEM[:100]}...")
                print(f"Prompt: {prompt[:300]}...")
                print()
        return

    # Load judge
    print(f"\nLoading judge model: {args.model}")
    judge = load_judge(args.model, load_4bit=not args.no_4bit)

    # Score
    print(f"\nScoring {len(records)} traces...")
    start_time = time.time()
    results = score_traces(records, judge)
    elapsed = time.time() - start_time
    print(f"Scoring completed in {elapsed:.1f}s ({elapsed/len(records):.2f}s/sample)")

    judge.unload()

    # Aggregate
    aggregates = compute_aggregates(results)
    print_summary(aggregates)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "metadata": {
            "judge_model": args.model,
            "total_traces": len(records),
            "scoring_time_seconds": round(elapsed, 1),
            "rubric": JUDGE_SYSTEM,
        },
        "aggregates": aggregates,
        "detailed_scores": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Results saved to {output_path}")


if __name__ == "__main__":
    main()
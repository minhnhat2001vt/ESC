"""
Extract thinking traces from ESC pipeline results for LLM-as-Judge scoring.

Reads result JSONs from multiple conditions, extracts the <think>...</think>
trace and the original question, and outputs a single JSONL file with all
samples ready for scoring. Condition labels are NOT included in the output
to keep the judge blind.

Usage:
    python extract_thinking_for_judge.py \
        --output thinking_traces.jsonl \
        --baseline /path/to/baseline.json \
        --emotion /path/to/emotion.json \
        --neutral /path/to/neutral.json \
        --psych /path/to/psychological.json \
        --none /path/to/none.json \
        --cot /path/to/cot.json
"""

import json
import re
import argparse
import uuid
from pathlib import Path


def extract_thinking(text: str) -> str:
    """Extract content inside <think>...</think> tags."""
    # Case 1: Both tags present
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Case 2: Only </think> present (opening tag missing, common in some models)
    match2 = re.search(r"^(.*)</think>", text, re.DOTALL)
    if match2:
        return match2.group(1).strip()
    return ""


def strip_image_token(question: str) -> str:
    return question.replace("<image>", "").replace("<image>\n", "").strip()


def process_file(filepath: str, condition: str) -> list:
    """Process a single results JSON and return extracted records."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    for item in data:
        # Use response_regen if available (ESC pipeline output), else response
        response = item.get("response_regen", item.get("response", ""))
        if not response or response.startswith("[Error"):
            continue

        thinking = extract_thinking(response)
        if not thinking:
            # No </think> tag found — response may be truncated mid-thought
            # (common when max_tokens is hit before thinking completes).
            # If response starts with thinking-like content, use it as-is.
            if response.startswith(("Got it", "So,", "Let me", "Okay", "Alright",
                                     "Wait", "First", "The user", "I need")):
                thinking = response
            else:
                continue

        question = strip_image_token(
            item.get("original_question", item.get("full_question", ""))
        )

        records.append({
            "trace_id": str(uuid.uuid4())[:8],
            "sample_id": item.get("id", item.get("question_id", "")),
            "condition": condition,  # stored for aggregation, NOT sent to judge
            "question": question,
            "thinking_trace": thinking,
        })

    return records


def main():
    parser = argparse.ArgumentParser(
        description="Extract thinking traces for LLM-as-Judge scoring"
    )
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL file path")
    parser.add_argument("--baseline", type=str, help="Baseline results JSON")
    parser.add_argument("--emotion", type=str, help="ESC (emotion) results JSON")
    parser.add_argument("--neutral", type=str, help="Neutral results JSON")
    parser.add_argument("--psych", type=str, help="Psychological results JSON")
    parser.add_argument("--none", type=str, help="Verifier-only (none) results JSON")
    parser.add_argument("--cot", type=str, help="CoT results JSON")
    parser.add_argument("--matched", action="store_true",
                        help="Only include samples that appear in ALL provided conditions")
    args = parser.parse_args()

    conditions = {
        "baseline": args.baseline,
        "emotion": args.emotion,
        "neutral": args.neutral,
        "psychological": args.psych,
        "none": args.none,
        "cot": args.cot,
    }

    # Extract all records per condition
    records_by_condition = {}
    for condition, filepath in conditions.items():
        if filepath is None:
            continue
        if not Path(filepath).exists():
            print(f"  ⚠️  File not found: {filepath}")
            continue

        records = process_file(filepath, condition)
        records_by_condition[condition] = records
        print(f"  {condition:15s}: {len(records)} traces extracted from {filepath}")

    # Matched filtering
    if args.matched and len(records_by_condition) > 1:
        # Find intersection of sample_ids across non-baseline conditions first,
        # then filter baseline to those same IDs.
        # This is because baseline has ALL samples while other conditions only
        # have verifier-flagged samples (which should mostly overlap).
        non_baseline = {k: v for k, v in records_by_condition.items() if k != "baseline"}
        
        if non_baseline:
            id_sets = []
            for condition, records in non_baseline.items():
                ids = set(r["sample_id"] for r in records)
                id_sets.append(ids)
                print(f"  {condition:15s}: {len(ids)} unique sample IDs")

            matched_ids = id_sets[0]
            for s in id_sets[1:]:
                matched_ids = matched_ids & s

            print(f"\n  Matched sample IDs across non-baseline conditions: {len(matched_ids)}")

            # Filter all conditions (including baseline) to matched IDs
            for condition in records_by_condition:
                before = len(records_by_condition[condition])
                records_by_condition[condition] = [
                    r for r in records_by_condition[condition]
                    if r["sample_id"] in matched_ids
                ]
                after = len(records_by_condition[condition])
                print(f"  {condition:15s}: {before} → {after} (matched)")
        else:
            print("  No non-baseline conditions found, skipping matched filtering.")

    # Combine all records
    all_records = []
    for records in records_by_condition.values():
        all_records.extend(records)

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n✅ Total: {len(all_records)} traces written to {output_path}")

    # Summary
    from collections import Counter
    counts = Counter(r["condition"] for r in all_records)
    print("\nPer-condition counts:")
    for cond, count in sorted(counts.items()):
        print(f"  {cond:15s}: {count}")


if __name__ == "__main__":
    main()
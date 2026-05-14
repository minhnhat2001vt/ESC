"""
RealWorldQA Evaluation Script

Usage:
    python eval_realworldqa.py --finding1 --model_dir llava_1_5_7b --gt_path ./original_data/RealWorldQA/RealWorldQA.json
    python eval_realworldqa.py --finding1 --finding2 --gt_path ./RealWorldQA.json
    python eval_realworldqa.py --finding1 --gt_path ./RealWorldQA.json --test_mode
"""

import json
import os
import re
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONSTANT PATHS
# ============================================================================
INFER_BASE_DIR = "results/infer"
EVAL_BASE_DIR = "results/eval"

# Valid answer letters
VALID_LETTERS = {"A", "B", "C", "D"}


# ============================================================================
# ANSWER EXTRACTION (following VLMEvalKit methodology)
# ============================================================================
def extract_answer_letter(response, choices=None):
    """
    Extract answer letter from model response using multi-level matching.
    Follows VLMEvalKit's can_infer() → exact match → regex → content match pipeline.

    Args:
        response: Model's raw response string
        choices: Dict of {letter: content} for content matching fallback

    Returns:
        (letter, method) tuple, e.g. ("A", "exact") or (None, "failed")
    """
    if not response or not isinstance(response, str):
        return None, "empty"

    response = response.strip()

    # ---- Level 1: Exact match (response IS the letter) ----
    if response.upper() in VALID_LETTERS:
        return response.upper(), "exact"

    # ---- Level 2: Starts with letter ----
    # e.g. "A. Left" or "A) Left" or "A: Left" or "A Left"
    match = re.match(r'^([A-D])\s*[.):,\s]', response, re.IGNORECASE)
    if match:
        return match.group(1).upper(), "starts_with"

    # ---- Level 3: "The answer is X" pattern ----
    patterns = [
        r'(?:the\s+)?answer\s+is\s*[:\s]*([A-D])',
        r'(?:the\s+)?correct\s+(?:answer|option|choice)\s+is\s*[:\s]*([A-D])',
        r'(?:I\s+)?(?:would\s+)?(?:choose|select|pick)\s+([A-D])',
        r'(?:option|choice)\s+([A-D])',
        r'\b([A-D])\s+is\s+(?:the\s+)?(?:correct|right|best)\b',
    ]
    for pat in patterns:
        match = re.search(pat, response, re.IGNORECASE)
        if match:
            return match.group(1).upper(), "pattern"

    # ---- Level 4: Single letter in response ----
    # If only one valid letter appears (as standalone), use it
    found_letters = set()
    for m in re.finditer(r'\b([A-D])\b', response):
        letter = m.group(1).upper()
        # Check it's not part of a normal word
        start = m.start()
        end = m.end()
        # Ensure it's standalone (not inside a word)
        before = response[start-1] if start > 0 else ' '
        after = response[end] if end < len(response) else ' '
        if not before.isalpha() and not after.isalpha():
            found_letters.add(letter)

    if len(found_letters) == 1:
        return found_letters.pop(), "single_letter"

    # ---- Level 5: Content matching (compare response with option text) ----
    if choices:
        response_lower = response.lower().strip()
        for letter, content in choices.items():
            if not content or content.strip() == "":
                continue
            content_lower = content.lower().strip()
            # Exact content match
            if response_lower == content_lower:
                return letter.upper(), "content_exact"
            # Response starts with the content
            if response_lower.startswith(content_lower):
                return letter.upper(), "content_starts"
            # Content is a major substring of response
            if len(content_lower) > 3 and content_lower in response_lower:
                # Check it's a significant portion
                if len(content_lower) / max(len(response_lower), 1) > 0.3:
                    return letter.upper(), "content_contains"

    # ---- Level 6: First letter at start after "Answer:" prefix ----
    match = re.search(r'(?:^|\n)\s*(?:Answer|Response|My answer)\s*[:\s]+([A-D])',
                      response, re.IGNORECASE)
    if match:
        return match.group(1).upper(), "answer_prefix"

    return None, "failed"


# ============================================================================
# GROUND TRUTH LOADER
# ============================================================================
def load_realworldqa_gt(gt_path):
    """
    Load RealWorldQA ground truth.
    Returns list of dicts with: index, question, A, B, C, D, answer, image
    """
    if not os.path.exists(gt_path):
        print(f"❌ Ground truth not found: {gt_path}")
        print(f"   Expected: RealWorldQA.json")
        return None

    with open(gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    print(f"✅ Loaded RealWorldQA GT: {len(gt_data)} samples")

    # Analyze choice distribution
    choice_counts = defaultdict(int)
    answer_dist = defaultdict(int)
    for item in gt_data:
        num_choices = sum(1 for k in ["A", "B", "C", "D"]
                         if item.get(k, "").strip() != "")
        choice_counts[num_choices] += 1
        answer_dist[item.get("answer", "?")] += 1

    print(f"   Choice distribution: {dict(sorted(choice_counts.items()))}")
    print(f"   Answer distribution: {dict(sorted(answer_dist.items()))}")

    return gt_data


def build_gt_index(gt_data):
    """Build lookup index from GT data by index field."""
    idx = {}
    for item in gt_data:
        key = str(item.get("index", ""))
        idx[key] = item
    return idx


# ============================================================================
# EVALUATION METRICS
# ============================================================================
def compute_accuracy_metrics(evaluated_results):
    """
    Compute RealWorldQA accuracy metrics.

    Returns:
    - overall_accuracy: % correct
    - per_num_choices: accuracy by number of choices (2/3/4)
    - extraction_stats: how answers were extracted
    """
    if not evaluated_results:
        return {}

    total = len(evaluated_results)
    correct = sum(1 for r in evaluated_results if r.get("is_correct", False))
    accuracy = correct / total * 100 if total > 0 else 0

    # By number of choices
    by_choices = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in evaluated_results:
        nc = r.get("num_choices", 0)
        by_choices[nc]["total"] += 1
        if r.get("is_correct", False):
            by_choices[nc]["correct"] += 1

    choice_metrics = {}
    for nc, counts in sorted(by_choices.items()):
        acc = counts["correct"] / counts["total"] * 100 if counts["total"] > 0 else 0
        choice_metrics[nc] = {
            "total": counts["total"],
            "correct": counts["correct"],
            "accuracy": acc,
            "random_baseline": 100.0 / nc if nc > 0 else 0,
        }

    # Extraction method stats
    method_counts = defaultdict(int)
    for r in evaluated_results:
        method_counts[r.get("extraction_method", "unknown")] += 1

    # Answer distribution in predictions
    pred_dist = defaultdict(int)
    for r in evaluated_results:
        pred_dist[r.get("predicted_answer", "None")] += 1

    return {
        "overall_accuracy": float(accuracy),
        "total_samples": total,
        "correct_samples": correct,
        "by_num_choices": choice_metrics,
        "extraction_methods": dict(sorted(method_counts.items())),
        "prediction_distribution": dict(sorted(pred_dist.items())),
    }


# ============================================================================
# FILE DISCOVERY
# ============================================================================
def discover_result_files(finding_name, model_dir=None):
    """Discover inference result files for a RealWorldQA finding."""
    found_files = []

    if model_dir:
        search_dirs = [os.path.join(INFER_BASE_DIR, model_dir, finding_name)]
    else:
        base = Path(INFER_BASE_DIR)
        if not base.exists():
            return []
        search_dirs = []
        for child in sorted(base.iterdir()):
            if child.is_dir():
                finding_dir = child / finding_name
                if finding_dir.exists():
                    search_dirs.append(str(finding_dir))
        legacy_dir = os.path.join(INFER_BASE_DIR, finding_name)
        if os.path.exists(legacy_dir):
            search_dirs.append(legacy_dir)

    for search_dir in search_dirs:
        search_path = Path(search_dir)
        if not search_path.exists():
            continue
        v6_files = sorted(search_path.glob("results_*.json"))
        legacy_files = sorted(search_path.glob(f"{finding_name}_*_results_*.json"))
        all_files = list(set(v6_files + legacy_files))
        all_files = [f for f in all_files if "summary" not in f.name.lower()
                     and "evaluated" not in f.name.lower()
                     and "grade" not in f.name.lower()]
        found_files.extend(sorted(all_files))

    if not found_files:
        print(f"❌ No result files found for {finding_name}")
    return found_files


def parse_dataset_name(result_filename):
    """Extract dataset name from result filename."""
    name = result_filename.stem
    if name.startswith("results_"):
        return name[len("results_"):]
    match = re.match(r"(.+?)_results_\d{8}_\d{6}", name)
    if match:
        return match.group(1)
    return name


# ============================================================================
# EVALUATION LOGIC
# ============================================================================
def evaluate_realworldqa_results(results, gt_data):
    """
    Evaluate inference results against RealWorldQA ground truth.
    Pure string matching — no LLM judge needed.
    """
    gt_index = build_gt_index(gt_data)

    evaluated = []
    for r in results:
        # Match with GT
        sample_idx = str(r.get("index",
                 r.get("question_id",
                 r.get("sample_id",
                 r.get("id", "")))))
        gt_entry = gt_index.get(sample_idx)

        if gt_entry is None:
            # Try matching by question text
            question = r.get("question", r.get("original_question", ""))
            for gt in gt_data:
                if gt.get("question", "") == question:
                    gt_entry = gt
                    break

        if gt_entry is None:
            continue

        # Get GT answer
        gt_answer = gt_entry.get("answer", "").strip().upper()

        # Get model prediction
        prediction = r.get("response", r.get("prediction", ""))

        # Build choices dict for content matching
        choices = {}
        num_choices = 0
        for letter in ["A", "B", "C", "D"]:
            content = gt_entry.get(letter, "").strip()
            if content:
                choices[letter] = content
                num_choices += 1

        # Extract answer
        predicted_letter, method = extract_answer_letter(prediction, choices)

        # Check correctness
        is_correct = (predicted_letter == gt_answer) if predicted_letter else False

        evaluated.append({
            "index": sample_idx,
            "question": gt_entry.get("question", ""),
            "gt_answer": gt_answer,
            "gt_content": choices.get(gt_answer, ""),
            "predicted_answer": predicted_letter,
            "extraction_method": method,
            "is_correct": is_correct,
            "raw_response": prediction[:300] if prediction else "",
            "num_choices": num_choices,
            # Carry over metadata
            "emotion_category": r.get("emotion_category", ""),
            "finding": r.get("finding", ""),
            "subject": r.get("subject", ""),
            "image": gt_entry.get("image", ""),
        })

    if not evaluated:
        print("❌ No samples matched with GT!")
        return None

    metrics = compute_accuracy_metrics(evaluated)

    # Per-emotion breakdown
    emotion_metrics = defaultdict(list)
    for r in evaluated:
        emo = r.get("emotion_category", "neutral") or "neutral"
        emotion_metrics[emo].append(1 if r["is_correct"] else 0)

    emotion_summary = {}
    for emo, vals in sorted(emotion_metrics.items()):
        emotion_summary[emo] = {
            "total": len(vals),
            "correct": sum(vals),
            "accuracy": sum(vals) / len(vals) * 100 if vals else 0,
        }

    return {
        "metadata": {
            "evaluation_date": datetime.now().isoformat(),
            "evaluation_method": "realworldqa_exact_match",
            "dataset": "realworldqa",
            "judge": "none (exact match + regex extraction)",
        },
        "summary": metrics,
        "emotion_breakdown": emotion_summary,
        "detailed_results": evaluated,
    }


# ============================================================================
# FINDING-LEVEL EVALUATION
# ============================================================================
def run_finding_evaluation(finding_name, gt_data, output_dir,
                           skip_neutral=False, model_dir=None):
    """Evaluate all result files for a finding."""
    result_files = discover_result_files(finding_name, model_dir)
    if not result_files:
        return None

    if skip_neutral:
        result_files = [f for f in result_files if "NEUTRAL" not in f.name.upper()]

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"{finding_name.upper()} EVALUATION (RealWorldQA — exact match)")
    print(f"{'='*80}")
    print(f"Method: Multiple-choice exact match accuracy")
    print(f"Judge:  None (string matching, no LLM needed)")
    print(f"Files:  {len(result_files)}")
    for f in result_files:
        print(f"  - {f.name}")
    print(f"{'='*80}\n")

    comparison_data = []

    for idx, rf in enumerate(result_files, 1):
        ds_name = parse_dataset_name(rf)
        print(f"\n[{idx}/{len(result_files)}] Evaluating: {ds_name}")

        with open(rf, "r", encoding="utf-8") as f:
            results = json.load(f)
        print(f"   Samples: {len(results)}")

        r0 = results[0] if results else {}
        emotion_category = r0.get("emotion_category", "unknown")
        finding = r0.get("finding", finding_name)

        ds_upper = ds_name.upper()
        if "_I_" in ds_upper and "_YOU_" not in ds_upper:
            subject = "I"
        elif "_YOU_" in ds_upper:
            subject = "You"
        else:
            subject = ""

        evaluation = evaluate_realworldqa_results(results, gt_data)
        if evaluation is None:
            continue

        # Save detailed results
        eval_path = os.path.join(output_dir, f"{ds_name}_evaluated.json")
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(evaluation, f, indent=2, ensure_ascii=False)

        s = evaluation["summary"]
        print(f"   ✅ Accuracy: {s['overall_accuracy']:.1f}% ({s['correct_samples']}/{s['total_samples']})")
        print(f"      Extraction: {s['extraction_methods']}")
        if s.get("by_num_choices"):
            for nc, cm in sorted(s["by_num_choices"].items()):
                print(f"      {nc}-choice: {cm['accuracy']:.1f}% (n={cm['total']}, random={cm['random_baseline']:.1f}%)")

        is_neutral = "NEUTRAL" in ds_name.upper()
        condition = "neutral" if is_neutral else emotion_category

        row = {
            "dataset": ds_name,
            "finding": finding,
            "condition": condition,
            "emotion_category": emotion_category,
            "subject": subject,
            "total_samples": s["total_samples"],
            "correct_samples": s["correct_samples"],
            "accuracy": s["overall_accuracy"],
        }
        # Per-choice-count accuracy
        for nc, cm in s.get("by_num_choices", {}).items():
            row[f"acc_{nc}choice"] = cm["accuracy"]

        comparison_data.append(row)

    if not comparison_data:
        print("❌ No evaluations completed.")
        return None

    # Save comparison
    df = pd.DataFrame(comparison_data)
    df.to_csv(os.path.join(output_dir, f"{finding_name}_comparison.csv"), index=False)
    with open(os.path.join(output_dir, f"{finding_name}_comparison.json"), "w") as f:
        json.dump(comparison_data, f, indent=2)

    # Print comparison table
    print(f"\n{'='*80}")
    print(f"{finding_name.upper()} COMPARISON (RealWorldQA)")
    print(f"{'='*80}")
    print(f"{'Condition':<35} {'Accuracy':>10} {'Correct':>10} {'Total':>8}")
    print(f"{'-'*80}")
    df_sorted = df.sort_values("accuracy", ascending=False)
    for _, row in df_sorted.iterrows():
        subj = f" [{row['subject']}]" if row.get("subject") else ""
        cond = f"{row['condition']}{subj}"
        print(f"{cond:<35} {row['accuracy']:>9.1f}% {row['correct_samples']:>9}/{row['total_samples']:<8}")

    # Visualization
    create_plot(df, finding_name, output_dir)

    # Summary
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "finding": finding_name,
        "dataset": "realworldqa",
        "evaluation_method": "exact_match_accuracy",
        "evaluation_date": datetime.now().isoformat(),
        "files_evaluated": len(result_files),
        "random_guess_baseline": 37.7,
        "comparison": comparison_data,
    }
    with open(os.path.join(output_dir, f"{finding_name}_eval_summary_{ts}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ {finding_name} evaluation complete → {output_dir}")
    return df


# ============================================================================
# VISUALIZATION
# ============================================================================
COLOR_MAP = {
    "positive_high_arousal": "#2ecc71",
    "positive_low_arousal": "#27ae60",
    "negative_high_arousal": "#e74c3c",
    "negative_low_arousal": "#c0392b",
    "positive_high": "#2ecc71",
    "positive_low": "#27ae60",
    "negative_high": "#e74c3c",
    "negative_low": "#c0392b",
    "empathy": "#9b59b6",
    "psychological": "#3498db",
    "neutral": "#808080",
}


def create_plot(df, finding_name, output_dir):
    """Create accuracy comparison plot."""
    sns.set_style("whitegrid")
    if len(df) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.6)))

    df_sorted = df.sort_values("accuracy")

    labels = []
    for _, row in df_sorted.iterrows():
        lbl = row["condition"]
        if row.get("subject") and row["subject"] not in ("", "none", "unknown"):
            lbl += f" [{row['subject']}]"
        labels.append(lbl)

    colors = [COLOR_MAP.get(row["condition"], "#333") for _, row in df_sorted.iterrows()]

    bars = ax.barh(range(len(df_sorted)), df_sorted["accuracy"], color=colors)
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Accuracy (%)", fontsize=11)
    ax.set_xlim(0, 100)

    # Random guess baseline
    ax.axvline(x=37.7, color='red', linestyle='--', alpha=0.5, label='Random guess (37.7%)')
    ax.legend(fontsize=8, loc='lower right')

    # Score annotations
    for bar, val in zip(bars, df_sorted["accuracy"]):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", ha="left", va="center", fontsize=8)

    title_map = {
        "realworldqa_finding1": "RealWorldQA Finding 1: Emotion vs Neutral",
        "realworldqa_finding2": "RealWorldQA Finding 2: By Emotion Category",
        "realworldqa_finding3": "RealWorldQA Finding 3: Subject (I vs YOU)",
        "realworldqa_finding4": "RealWorldQA Finding 4: Visual Emotion",
    }
    ax.set_title(title_map.get(finding_name, f"RealWorldQA — {finding_name}"),
                 fontsize=13)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{finding_name}_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 Plot saved: {plot_path}")


# ============================================================================
# TEST MODE
# ============================================================================
NUM_TEST_SAMPLES = 5


def run_test_mode(gt_data, finding_name, model_dir=None):
    """Test mode: evaluate a few samples with detailed output."""
    print(f"\n{'='*80}")
    print("TEST MODE — RealWorldQA Evaluation (exact match)")
    print(f"{'='*80}")
    print(f"  Finding:   {finding_name}")
    print(f"  Method:    Multiple-choice exact match (no LLM judge)")
    print(f"  Model dir: {model_dir or '(all)'}")
    print(f"  Samples:   {NUM_TEST_SAMPLES}")

    # Step 1: File discovery
    print(f"\n{'='*80}")
    print("STEP 1: FILE DISCOVERY")
    print(f"{'='*80}")

    result_files = discover_result_files(finding_name, model_dir)
    if not result_files:
        print("❌ No result files found.")
        return

    print(f"  Found {len(result_files)} file(s):")
    for f in result_files:
        print(f"    - {f}")

    test_file = result_files[0]
    print(f"\n  Using: {test_file.name}")

    # Step 2: Load and inspect
    print(f"\n{'='*80}")
    print("STEP 2: DATA INSPECTION")
    print(f"{'='*80}")

    with open(test_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    print(f"  Total samples: {len(results)}")
    if not results:
        print("❌ File empty!")
        return

    r0 = results[0]
    print(f"\n  First result fields:")
    for key in sorted(r0.keys()):
        val_str = str(r0[key])
        if len(val_str) > 80:
            val_str = val_str[:80] + "..."
        print(f"    {key}: {val_str}")

    # Step 3: Evaluate a few samples
    print(f"\n{'='*80}")
    print(f"STEP 3: EVALUATING {NUM_TEST_SAMPLES} SAMPLES")
    print(f"{'='*80}")

    gt_index = build_gt_index(gt_data)
    scored = 0

    for sample in results:
        if scored >= NUM_TEST_SAMPLES:
            break

        sample_idx = str(sample.get("index", sample.get("sample_id", sample.get("id", ""))))
        gt_entry = gt_index.get(sample_idx)
        if gt_entry is None:
            continue

        gt_answer = gt_entry.get("answer", "").strip().upper()
        prediction = sample.get("response", sample.get("prediction", ""))

        choices = {}
        for letter in ["A", "B", "C", "D"]:
            content = gt_entry.get(letter, "").strip()
            if content:
                choices[letter] = content

        predicted_letter, method = extract_answer_letter(prediction, choices)
        is_correct = (predicted_letter == gt_answer)

        print(f"\n  [{scored+1}/{NUM_TEST_SAMPLES}]")
        print(f"  Index:      {sample_idx}")
        print(f"  Question:   {gt_entry.get('question', '')[:100]}")
        print(f"  Choices:    {choices}")
        print(f"  GT Answer:  {gt_answer} → {choices.get(gt_answer, '?')}")
        print(f"  Response:   {str(prediction)[:150]}{'...' if len(str(prediction)) > 150 else ''}")
        print(f"  Extracted:  {predicted_letter} (method: {method})")
        print(f"  Correct:    {'✅' if is_correct else '❌'}")

        scored += 1

    # Step 4: Quick full eval
    print(f"\n{'='*80}")
    print("STEP 4: QUICK FULL EVALUATION")
    print(f"{'='*80}")

    evaluation = evaluate_realworldqa_results(results, gt_data)
    if evaluation:
        s = evaluation["summary"]
        print(f"  Overall Accuracy: {s['overall_accuracy']:.1f}% ({s['correct_samples']}/{s['total_samples']})")
        print(f"  Extraction stats: {s['extraction_methods']}")
        if s.get("by_num_choices"):
            for nc, cm in sorted(s["by_num_choices"].items()):
                print(f"    {nc}-choice: {cm['accuracy']:.1f}% (n={cm['total']})")

    print(f"\n✅ Test complete. Run without --test_mode for full evaluation.")


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="RealWorldQA Evaluation (exact match accuracy)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
RealWorldQA (xAI, 2024) evaluates real-world spatial understanding.
765 images with multiple-choice questions (2-4 options).
Pure string matching — NO LLM judge needed.

Metric: Accuracy (%) = correct / total × 100
Baseline: Random guess ~37.7%
Best known: GPT-4V ~68%, LLaVA-NeXT (Yi-34B) ~66%

Examples:
  python eval_realworldqa.py --finding1 --model_dir llava_1_5_7b --gt_path ./RealWorldQA.json
  python eval_realworldqa.py --finding1 --finding2 --gt_path ./RealWorldQA.json
  python eval_realworldqa.py --finding1 --gt_path ./RealWorldQA.json --test_mode
        """,
    )

    parser.add_argument("--finding1", action="store_true")
    parser.add_argument("--finding2", action="store_true")
    parser.add_argument("--finding3", action="store_true")
    parser.add_argument("--finding4", action="store_true")

    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--gt_path", type=str, required=True,
                        help="Path to RealWorldQA.json ground truth file")
    parser.add_argument("--skip_neutral", action="store_true")
    parser.add_argument("--test_mode", action="store_true",
                        help="Evaluate 5 samples with detailed output")

    args = parser.parse_args()

    findings = {
        "realworldqa_finding1": args.finding1,
        "realworldqa_finding2": args.finding2,
        "realworldqa_finding3": args.finding3,
        "realworldqa_finding4": args.finding4,
    }
    if not any(findings.values()):
        parser.error("Specify at least one: --finding1, --finding2, --finding3, --finding4")

    eval_subdir = args.model_dir if args.model_dir else "all_models"

    print(f"\n{'='*80}")
    print("REALWORLDQA EVALUATION (exact match accuracy)")
    print(f"{'='*80}")
    print(f"GT path:    {args.gt_path}")
    print(f"Input:      {INFER_BASE_DIR}" + (f"/{args.model_dir}" if args.model_dir else " (all)"))
    print(f"Output:     {EVAL_BASE_DIR}/{eval_subdir}")
    print(f"Method:     Multiple-choice exact match (no LLM judge)")

    # Load GT
    gt_data = load_realworldqa_gt(args.gt_path)
    if gt_data is None:
        return

    # Test mode
    if args.test_mode:
        first_finding = next(name for name, sel in findings.items() if sel)
        run_test_mode(gt_data, first_finding, model_dir=args.model_dir)
        return

    # Full evaluation
    for finding_name, selected in findings.items():
        if not selected:
            continue
        output_dir = os.path.join(EVAL_BASE_DIR, eval_subdir, finding_name)
        run_finding_evaluation(
            finding_name=finding_name,
            gt_data=gt_data,
            output_dir=output_dir,
            skip_neutral=args.skip_neutral,
            model_dir=args.model_dir,
        )

    print(f"\n{'='*80}")
    print("✅ ALL REALWORLDQA EVALUATIONS COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
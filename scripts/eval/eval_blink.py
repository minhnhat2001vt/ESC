"""
BLINK Benchmark Evaluation Script — Official Metrics (Accuracy, macro-avg)

Updated to match the output workflow/style of eval_mme.py:
- Supports finding-level evaluation via --model_dir + --finding
- Saves outputs under results/eval/{model_dir}/{finding}/ by default
- Still supports single-file evaluation via --result_file
- Writes per-file evaluated JSON + comparison CSV/JSON + summary JSON
- Supports --method flag for final_response (method) vs response (baseline)

Data format (inference result fields used):
    id:             "blink_val_Art_Style_1"  ← subtask extracted from ID
    gt_answer:      "A" / "B" / "C" / "D"   ← already a letter
    response:       model's raw output       ← used for baseline
    final_response: method's final output    ← used for --method
    full_question:  prompt with "(A)..." etc  ← used to determine valid choices

Reference:
    BLINK: Multimodal Large Language Models Can See but Not Perceive (ECCV 2024)
    Evaluation: VLMEvalKit standard, macro-average accuracy across 14 subtasks.
"""

import json
import os
import re
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

try:
    import pandas as pd
    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False

# ============================================================================
# CONSTANT PATHS
# ============================================================================
INFER_BASE_DIR = "results/infer"
EVAL_BASE_DIR  = "results/eval"

ALL_SUBTASKS = [
    "Art_Style", "Counting", "Forensic_Detection", "Functional_Correspondence",
    "IQ_Test", "Jigsaw", "Multi-view_Reasoning", "Object_Localization",
    "Relative_Depth", "Relative_Reflectance", "Semantic_Correspondence",
    "Spatial_Relation", "Visual_Correspondence", "Visual_Similarity",
]

# Number of choices per subtask — verified against actual val-split data
# Used only for random baseline display, NOT for answer extraction.
DEFAULT_NUM_CHOICES = {
    "Art_Style": 2, "Counting": 4, "Forensic_Detection": 4,
    "Functional_Correspondence": 4, "IQ_Test": 4, "Jigsaw": 2,
    "Multi-view_Reasoning": 2, "Object_Localization": 2,
    "Relative_Depth": 2, "Relative_Reflectance": 3,
    "Semantic_Correspondence": 4, "Spatial_Relation": 2,
    "Visual_Correspondence": 4, "Visual_Similarity": 2,
}


# ============================================================================
# VALID CHOICES PARSING
# ============================================================================
def _parse_valid_choices(result: dict) -> list:
    """
    Determine valid answer letters for this sample by parsing full_question.

    BLINK subtasks have 2–4 choices. Rather than relying on a hardcoded table
    we parse the actual "(A) ... (B) ..." options from the prompt.  This is
    important because extract_answer() limits matching to valid_set — if we
    say A–D are valid but only A–B exist, the model might accidentally match
    'C' or 'D' from its explanation text.

    Falls back to DEFAULT_NUM_CHOICES → ["A","B","C","D"].
    """
    fq = result.get("full_question", "") or result.get("original_question", "")
    if fq:
        letters_found = sorted(set(re.findall(r'\(([A-Z])\)', fq)))
        if letters_found:
            return letters_found

    # Fallback: use DEFAULT_NUM_CHOICES based on subtask
    subtask = _get_subtask(result)
    n = DEFAULT_NUM_CHOICES.get(subtask, 4)
    return [chr(ord("A") + i) for i in range(n)]


# ============================================================================
# ANSWER EXTRACTION (Following VLMEvalKit / BLINK paper methodology)
# ============================================================================
def extract_answer(response: str, valid_choices: list = None) -> str:
    """
    Extract answer letter from model response using multi-level strategy.

    Following the BLINK paper which uses VLMEvalKit's extraction rules:
    1. Direct match: response is just the letter
    2. Pattern: "The answer is (X)" / "Answer: X"
    3. Parenthesized letter: "(A)", "(B)", etc.
    4. First capital letter at the beginning
    5. Any standalone valid letter
    6. Fallback: FAILED

    Returns:
        Extracted answer letter (uppercase) or "FAILED"
    """
    if not response or not response.strip():
        return "FAILED"

    if valid_choices is None:
        valid_choices = ["A", "B", "C", "D"]

    response = response.strip()
    valid_set = set(valid_choices)

    # Level 1: Exact match — response is just the letter
    cleaned = response.strip().upper().strip("().- \t\n")
    if cleaned in valid_set:
        return cleaned

    match = re.match(r'^\(?([A-Z])\)?[\.\):\s]*$', response.strip(), re.IGNORECASE)
    if match and match.group(1).upper() in valid_set:
        return match.group(1).upper()

    response_upper = response.upper()

    # Level 2: "The answer is (X)" patterns
    patterns = [
        r'(?:the\s+)?answer\s+is\s*[:\s]*\(?([A-Z])\)?',
        r'(?:I\s+)?(?:choose|select|pick)\s*[:\s]*\(?([A-Z])\)?',
        r'(?:correct\s+)?(?:answer|option|choice)\s*[:\s]*\(?([A-Z])\)?',
        r'(?:it\s+is|it\'s|should\s+be)\s+\(?([A-Z])\)?',
        r'\b([A-Z])\s+is\s+(?:the\s+)?(?:correct|right|best)\s+(?:answer|option|choice)',
    ]
    for pattern in patterns:
        match = re.search(pattern, response_upper)
        if match and match.group(1) in valid_set:
            return match.group(1)

    # Level 3: Letter in parentheses
    paren_matches = re.findall(r'\(([A-Z])\)', response_upper)
    for m in paren_matches:
        if m in valid_set:
            return m

    # Level 4: First standalone capital letter at beginning
    match = re.match(r'^([A-Z])[\.\)\s]', response.strip())
    if match and match.group(1).upper() in valid_set:
        return match.group(1).upper()

    # Level 5: Any standalone valid letter (first occurrence)
    for letter in valid_choices:
        if re.search(r'\b' + letter + r'\b', response_upper):
            return letter

    return "FAILED"


# ============================================================================
# SUBTASK EXTRACTION
# ============================================================================
def _get_subtask(result: dict) -> str:
    """Extract subtask name from result ID or metadata."""
    # From ID: "blink_val_Art_Style_1" → "Art_Style"
    sample_id = result.get("id", "")
    for st in ALL_SUBTASKS:
        if st in sample_id:
            return st

    # Try direct fields
    for field in ["subtask", "sub_task"]:
        val = result.get(field, "")
        if val and val in ALL_SUBTASKS:
            return val

    # Try nested metadata
    meta = result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {}
    for field in ["subtask", "sub_task", "image_type"]:
        val = meta.get(field, "")
        if val and val in ALL_SUBTASKS:
            return val

    return "unknown"


# ============================================================================
# CORE EVALUATION
# ============================================================================
def _eval_blink(results: list, response_field: str = "response") -> dict:
    """Evaluate BLINK results following the official protocol."""
    subtask_results = defaultdict(lambda: {"total": 0, "correct": 0, "extracted": 0})
    unknown_subtask = []

    total = len(results)
    correct = 0
    extracted = 0
    pred_dist = defaultdict(int)
    gt_dist = defaultdict(int)

    for r in results:
        resp = r.get(response_field, r.get("response", ""))
        gt_answer = str(r.get("gt_answer", "")).upper().strip()

        # Clean gt_answer: "(A)" → "A" (though BLINK data is already clean)
        gt_match = re.search(r'\(?([A-Z])\)?', gt_answer)
        if gt_match:
            gt_answer = gt_match.group(1)

        # Parse valid choices from the actual prompt
        valid_choices = _parse_valid_choices(r)
        predicted = extract_answer(resp, valid_choices)

        is_correct = (predicted == gt_answer) if predicted != "FAILED" else False
        extraction_ok = predicted != "FAILED"

        if is_correct:
            correct += 1
        if extraction_ok:
            extracted += 1

        pred_dist[predicted] += 1
        gt_dist[gt_answer] += 1

        subtask = _get_subtask(r)
        if subtask == "unknown":
            unknown_subtask.append(r)
        subtask_results[subtask]["total"] += 1
        if is_correct:
            subtask_results[subtask]["correct"] += 1
        if extraction_ok:
            subtask_results[subtask]["extracted"] += 1

    if unknown_subtask:
        print(f"  ⚠️  {len(unknown_subtask)} samples with unknown subtask")

    # Per-subtask scores
    subtask_scores = {}
    for st, stats in sorted(subtask_results.items()):
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        ext_rate = stats["extracted"] / stats["total"] if stats["total"] > 0 else 0.0
        random_bl = 1.0 / DEFAULT_NUM_CHOICES.get(st, 4)
        subtask_scores[st] = {
            "total": stats["total"],
            "correct": stats["correct"],
            "extracted": stats["extracted"],
            "accuracy": round(acc * 100, 2),
            "extraction_rate": round(ext_rate * 100, 2),
            "random_baseline": round(random_bl * 100, 2),
        }

    # Macro-average (paper's official metric) — exclude "unknown"
    valid_subtasks = {k: v for k, v in subtask_scores.items() if k != "unknown"}
    macro_acc = (sum(v["accuracy"] for v in valid_subtasks.values()) / len(valid_subtasks)
                 if valid_subtasks else 0.0)

    overall_acc = correct / total if total > 0 else 0.0
    extraction_rate = extracted / total if total > 0 else 0.0

    return {
        "total_samples": total,
        "correct": correct,
        "extracted": extracted,
        "overall_accuracy_micro": round(overall_acc * 100, 2),
        "overall_accuracy_macro": round(macro_acc, 2),
        "extraction_rate": round(extraction_rate * 100, 2),
        "num_subtasks": len(valid_subtasks),
        "subtask_scores": subtask_scores,
        "predicted_distribution": dict(sorted(pred_dist.items())),
        "gt_distribution": dict(sorted(gt_dist.items())),
        "unknown_subtask_count": len(unknown_subtask),
    }


# ============================================================================
# DISPLAY
# ============================================================================
def print_results(eval_result: dict, model_name: str = ""):
    print(f"\n{'='*80}")
    print(f"BLINK EVALUATION RESULTS{' — ' + model_name if model_name else ''}")
    print(f"{'='*80}")

    print(f"\n  Total samples:         {eval_result['total_samples']}")
    print(f"  Extraction rate:       {eval_result['extraction_rate']}%")
    print(f"  Accuracy (micro):      {eval_result['overall_accuracy_micro']}%")
    print(f"  Accuracy (macro):      {eval_result['overall_accuracy_macro']}%  ← BLINK paper metric")

    st_scores = eval_result.get("subtask_scores", {})
    if st_scores:
        print(f"\n  {'Subtask':<30s} {'Total':>6s} {'Correct':>8s} {'Acc%':>7s} "
              f"{'Extract%':>9s} {'Random%':>8s}")
        print(f"  {'-'*30} {'-'*6} {'-'*8} {'-'*7} {'-'*9} {'-'*8}")
        for st_name, stats in sorted(st_scores.items()):
            print(f"  {st_name:<30s} {stats['total']:>6d} {stats['correct']:>8d} "
                  f"{stats['accuracy']:>6.1f}% {stats['extraction_rate']:>8.1f}% "
                  f"{stats['random_baseline']:>7.1f}%")

    print(f"\n  Predicted distribution: {eval_result.get('predicted_distribution', {})}")
    print(f"  GT distribution:       {eval_result.get('gt_distribution', {})}")
    print(f"{'='*80}")


# ============================================================================
# DISCOVERY + PATH HELPERS (same style as eval_mme)
# ============================================================================
def discover_result_files(finding_name: str, model_dir: str | None = None):
    """Discover inference result files under results/infer/{model_dir}/{finding_name}/."""
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

    for search_dir in search_dirs:
        search_path = Path(search_dir)
        if not search_path.exists():
            continue
        files = sorted(search_path.glob("results_*.json"))
        files = [f for f in files
                 if "summary" not in f.name.lower()
                 and "evaluated" not in f.name.lower()
                 and "blink_eval" not in f.name.lower()]
        found_files.extend(files)

    return found_files


def infer_output_dir_from_result_file(result_file: str) -> str | None:
    """If result_file is under results/infer/<model>/<finding>/..., map to results/eval/<model>/<finding>."""
    try:
        p = Path(result_file)
        parts = list(p.parts)
        for i in range(len(parts) - 4):
            if parts[i] == "results" and parts[i + 1] == "infer":
                model_dir = parts[i + 2]
                finding_name = parts[i + 3]
                return str(Path(*parts[:i], "results", "eval", model_dir, finding_name))
        norm = str(p).replace("\\", "/")
        if "/results/infer/" in norm or norm.startswith("results/infer/"):
            tail = norm.split("results/infer/", 1)[1]
            segs = tail.split("/")
            if len(segs) >= 3:
                return os.path.join("results", "eval", segs[0], segs[1])
    except Exception:
        return None
    return None


def parse_dataset_name(result_path: Path) -> str:
    name = result_path.stem
    if name.startswith("results_"):
        return name[len("results_"):]
    return name


def _extract_condition_fields(ds_name: str, first_result: dict) -> tuple[str, str, str]:
    ds_upper = ds_name.upper()
    emotion_category = first_result.get("emotion_category", "neutral")
    condition = "neutral" if "NEUTRAL" in ds_upper else emotion_category
    if "_I_" in ds_upper and "_YOU_" not in ds_upper:
        subject = "I"
    elif "_YOU_" in ds_upper:
        subject = "You"
    else:
        subject = ""
    return condition, emotion_category, subject


# ============================================================================
# SAVE / RUN HELPERS
# ============================================================================
def build_output_payload(results: list, eval_result: dict, result_file: str,
                         response_field: str, model_name: str):
    return {
        "metadata": {
            "evaluator": "eval_blink.py (official BLINK protocol)",
            "evaluation_date": datetime.now().isoformat(),
            "evaluation_method": "blink_official",
            "reference": "BLINK: Multimodal Large Language Models Can See but Not Perceive, ECCV 2024",
            "result_file": os.path.basename(result_file),
            "response_field": response_field,
            "model": model_name,
            "total_samples": len(results),
        },
        "summary": {
            "total_samples": eval_result["total_samples"],
            "correct": eval_result["correct"],
            "extracted": eval_result["extracted"],
            "overall_accuracy_micro": eval_result["overall_accuracy_micro"],
            "overall_accuracy_macro": eval_result["overall_accuracy_macro"],
            "extraction_rate": eval_result["extraction_rate"],
            "num_subtasks": eval_result["num_subtasks"],
            "unknown_subtask_count": eval_result["unknown_subtask_count"],
        },
        "scores": {
            "accuracy_micro": eval_result["overall_accuracy_micro"],
            "accuracy_macro": eval_result["overall_accuracy_macro"],
        },
        "subtask_scores": eval_result["subtask_scores"],
        "distributions": {
            "predicted": eval_result["predicted_distribution"],
            "ground_truth": eval_result["gt_distribution"],
        },
    }


def evaluate_one_file(result_file: str, output_dir: str, response_field: str = "response") -> dict:
    """Evaluate one BLINK result file and save outputs. Returns a summary row."""
    with open(result_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    if isinstance(results, dict) and "results" in results:
        results = results["results"]

    print(f"\nLoading: {result_file}")
    print(f"  Loaded {len(results)} samples")

    model_name = results[0].get("model", "") if results else ""
    eval_result = _eval_blink(results, response_field=response_field)
    print_results(eval_result, model_name)

    os.makedirs(output_dir, exist_ok=True)

    stem = Path(result_file).stem
    ds_name = parse_dataset_name(Path(result_file))

    payload = build_output_payload(results, eval_result, result_file, response_field, model_name)

    # Main detailed file
    evaluated_path = os.path.join(output_dir, f"{ds_name}_evaluated.json")
    with open(evaluated_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # Backward-compatible file name
    legacy_path = os.path.join(output_dir, f"{stem}_blink_eval.json")
    with open(legacy_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Evaluation saved: {legacy_path}")
    if legacy_path != evaluated_path:
        print(f"✅ Detailed eval saved: {evaluated_path}")

    first_result = results[0] if results else {}
    finding_name = Path(result_file).parent.name
    condition, emotion_category, subject = _extract_condition_fields(ds_name, first_result)

    row = {
        "dataset": ds_name,
        "finding": finding_name,
        "condition": condition,
        "emotion_category": emotion_category,
        "subject": subject,
        "total_samples": eval_result["total_samples"],
        "accuracy_micro": eval_result["overall_accuracy_micro"],
        "accuracy_macro": eval_result["overall_accuracy_macro"],
        "extraction_rate": eval_result["extraction_rate"],
        "num_subtasks": eval_result["num_subtasks"],
        "unknown_subtask_count": eval_result["unknown_subtask_count"],
    }
    for st in ALL_SUBTASKS:
        if st in eval_result["subtask_scores"]:
            info = eval_result["subtask_scores"][st]
            row[f"acc_{st}"] = info["accuracy"]

    return row


def run_finding_evaluation(finding_name: str, output_dir: str, model_dir: str | None = None,
                           skip_neutral: bool = False, method: bool = False):
    """Evaluate all BLINK result files for a finding (same workflow as eval_mme)."""
    result_files = discover_result_files(finding_name, model_dir)
    if not result_files:
        print(f"❌ No result files found for {finding_name}")
        if model_dir:
            print(f"   Expected: {INFER_BASE_DIR}/{model_dir}/{finding_name}/results_*.json")
        return None

    if skip_neutral:
        result_files = [f for f in result_files if "NEUTRAL" not in f.name.upper()]

    os.makedirs(output_dir, exist_ok=True)
    response_field = "final_response" if method else "response"

    print(f"\n{'='*80}")
    print(f"BLINK EVALUATION — {finding_name.upper()}")
    print(f"{'='*80}")
    print("Method:  Official BLINK metrics (Accuracy, macro-avg across subtasks)")
    print(f"Field:   {response_field}")
    print(f"Files:   {len(result_files)}")
    for f in result_files:
        print(f"  - {f.name}")
    print(f"{'='*80}\n")

    comparison_data = []
    for idx, rf in enumerate(result_files, 1):
        print(f"\n[{idx}/{len(result_files)}] Evaluating: {rf.name}")
        row = evaluate_one_file(str(rf), output_dir=output_dir, response_field=response_field)
        comparison_data.append(row)

    if not comparison_data:
        print("❌ No evaluations completed.")
        return None

    # Save comparison files
    if HAS_PANDAS:
        try:
            df = pd.DataFrame(comparison_data)
            df.to_csv(os.path.join(output_dir, f"{finding_name}_comparison.csv"), index=False)
        except Exception:
            df = None
    else:
        df = None

    with open(os.path.join(output_dir, f"{finding_name}_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(comparison_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*80}")
    print(f"BLINK COMPARISON — {finding_name.upper()}")
    print(f"{'='*80}")
    print(f"{'Condition':<30} {'Macro%':>8} {'Micro%':>8} {'Extract%':>9} {'N':>6}")
    print(f"{'-'*70}")
    for row in sorted(comparison_data, key=lambda x: x["accuracy_macro"], reverse=True):
        subj = f" [{row['subject']}]" if row.get("subject") else ""
        cond = f"{row['condition']}{subj}"
        print(f"{cond:<30} {row['accuracy_macro']:>7.2f} {row['accuracy_micro']:>7.2f} "
              f"{row['extraction_rate']:>8.2f} {row['total_samples']:>5}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "finding": finding_name,
        "dataset": "blink",
        "evaluation_method": "blink_official",
        "reference": "BLINK: Multimodal Large Language Models Can See but Not Perceive, ECCV 2024",
        "evaluation_date": datetime.now().isoformat(),
        "response_field": response_field,
        "files_evaluated": len(comparison_data),
        "comparison": comparison_data,
    }
    with open(os.path.join(output_dir, f"{finding_name}_eval_summary_{ts}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n✅ {finding_name} evaluation complete → {output_dir}")
    return df


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="BLINK Benchmark Evaluation — Official Metrics (Accuracy, macro-avg)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
BLINK Metrics:
  Accuracy per subtask (14 subtasks)
  Overall = macro-average across subtasks (paper's official metric)
  Random baseline ~41.0%% (varies by task: 2–4 choices)

Usage (recommended, same style as eval_mme):
  python scripts/eval/eval_blink.py --model_dir internvl2_5_8b --finding blink_baseline

Single-file mode (still supported):
  python scripts/eval/eval_blink.py --result_file results/infer/.../results_blink_full_baseline.json

Method mode (uses final_response field):
  python scripts/eval/eval_blink.py --model_dir internvl2_5_8b --finding blink_baseline --method
        """,
    )

    parser.add_argument("--result_file", type=str, default=None,
                        help="Path to one inference results JSON file (optional)")
    parser.add_argument("--finding", type=str, default="blink_baseline",
                        help="Finding folder name under results/infer/{model_dir}/")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Model directory under results/infer/ (e.g., internvl2_5_8b)")
    parser.add_argument("--method", action="store_true",
                        help="Method mode: use 'final_response' instead of 'response'")
    parser.add_argument("--skip_neutral", action="store_true",
                        help="Skip datasets with NEUTRAL in filename")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory")

    args = parser.parse_args()

    # Single-file mode
    if args.result_file:
        if not os.path.exists(args.result_file):
            print(f"❌ Result file not found: {args.result_file}")
            return

        output_dir = args.output_dir or infer_output_dir_from_result_file(args.result_file) or os.path.dirname(args.result_file)
        response_field = "final_response" if args.method else "response"

        print(f"\n{'='*80}")
        print("BLINK EVALUATION (Single-file mode)")
        print(f"{'='*80}")
        print(f"Input:    {args.result_file}")
        print(f"Output:   {output_dir}")
        print(f"Method:   Official BLINK metrics (Accuracy, macro-avg)")
        print(f"Field:    {response_field}")

        evaluate_one_file(args.result_file, output_dir=output_dir, response_field=response_field)

        print(f"\n{'='*80}")
        print("✅ BLINK EVALUATION COMPLETE")
        print(f"{'='*80}")
        return

    # Finding-level mode (preferred)
    eval_subdir = args.model_dir if args.model_dir else "all_models"
    output_dir = args.output_dir or os.path.join(EVAL_BASE_DIR, eval_subdir, args.finding)

    print(f"\n{'='*80}")
    print("BLINK EVALUATION (ECCV 2024)")
    print(f"{'='*80}")
    print(f"Finding:  {args.finding}")
    print(f"Input:    {INFER_BASE_DIR}/{args.model_dir or '*'}/{args.finding}/")
    print(f"Output:   {output_dir}")
    print("Method:   Official BLINK metrics (Accuracy, macro-avg across subtasks)")
    print(f"Field:    {'final_response' if args.method else 'response'}")

    run_finding_evaluation(
        finding_name=args.finding,
        output_dir=output_dir,
        model_dir=args.model_dir,
        skip_neutral=args.skip_neutral,
        method=args.method,
    )

    print(f"\n{'='*80}")
    print("✅ BLINK EVALUATION COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
"""
MME Benchmark Evaluation Script — Official Metrics (ACC + ACC+)

Updated to match the output workflow/style of eval_pope.py and eval_figstep.py:
- Supports finding-level evaluation via --model_dir + --finding
- Saves outputs under results/eval/{model_dir}/{finding}/ by default
- Still supports single-file evaluation via --result_file
- Writes per-file evaluated JSON + comparison CSV/JSON + summary JSON
"""

import json
import os
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
EVAL_BASE_DIR = "results/eval"

# ============================================================================
# MME TASK CATEGORIZATION (official)
# ============================================================================
PERCEPTION_TASKS = [
    "existence", "count", "position", "color", "posters",
    "celebrity", "scene", "landmark", "artwork", "OCR",
]

COGNITION_TASKS = [
    "commonsense_reasoning", "numerical_calculation",
    "text_translation", "code_reasoning",
]

ALL_TASKS = PERCEPTION_TASKS + COGNITION_TASKS

TASK_ALIASES = {
    "poster": "posters",
    "ocr": "OCR",
    "commonsense": "commonsense_reasoning",
    "common_sense_reasoning": "commonsense_reasoning",
    "numerical": "numerical_calculation",
    "num_calculation": "numerical_calculation",
    "text_trans": "text_translation",
    "code": "code_reasoning",
}


# ============================================================================
# ANSWER EXTRACTION
# ============================================================================
def extract_yes_no(response: str) -> str:
    """Extract yes/no from a model response (MME expects binary answers)."""
    if not response:
        return "unknown"

    resp = str(response).strip()
    if not resp:
        return "unknown"

    resp_lower = resp.lower()

    # Exact match
    if resp_lower in ("yes", "yes.", "yes!"):
        return "yes"
    if resp_lower in ("no", "no.", "no!"):
        return "no"

    # First token
    first_word = resp_lower.split()[0].rstrip(".,!?;:")
    if first_word == "yes":
        return "yes"
    if first_word == "no":
        return "no"

    # Contains (earliest match wins)
    yes_pos = resp_lower.find("yes")
    no_pos = resp_lower.find("no")
    if yes_pos >= 0 and no_pos >= 0:
        return "yes" if yes_pos < no_pos else "no"
    if yes_pos >= 0:
        return "yes"
    if no_pos >= 0:
        return "no"

    return "unknown"


# ============================================================================
# TASK NAME EXTRACTION
# ============================================================================
def _normalize_task(name: str) -> str:
    name = str(name).strip()
    if name in TASK_ALIASES:
        return TASK_ALIASES[name]
    for canonical in ALL_TASKS:
        if name.lower() == canonical.lower():
            return canonical
    return name


def extract_task_name(result: dict) -> str:
    """Extract MME task name from a result entry."""
    # 1) direct metadata fields
    for field in ["mme_task", "category", "subcategory", "task"]:
        val = result.get(field, "")
        if val:
            norm = _normalize_task(val)
            if norm in ALL_TASKS:
                return norm

    # 2) nested metadata
    meta = result.get("metadata", {}) if isinstance(result.get("metadata", {}), dict) else {}
    for field in ["mme_task", "category", "subcategory", "task"]:
        val = meta.get(field, "")
        if val:
            norm = _normalize_task(val)
            if norm in ALL_TASKS:
                return norm

    # 3) image_path pattern .../MME_Benchmark_release_version/{task}/...
    image_path = result.get("image_path", "")
    if "MME_Benchmark_release_version" in image_path:
        parts = image_path.replace("\\", "/").split("/")
        for i, part in enumerate(parts):
            if part == "MME_Benchmark_release_version" and i + 1 < len(parts):
                norm = _normalize_task(parts[i + 1])
                if norm in ALL_TASKS:
                    return norm

    # 4) id pattern: mme_{task}_{stem}_{q_idx}_{running}
    sample_id = result.get("id", "")
    if isinstance(sample_id, str) and sample_id.startswith("mme_"):
        remainder = sample_id[4:]
        # multi-word tasks first
        for task in [
            "commonsense_reasoning",
            "numerical_calculation",
            "text_translation",
            "code_reasoning",
        ]:
            if remainder.startswith(task + "_"):
                return task
        first_seg = remainder.split("_")[0]
        norm = _normalize_task(first_seg)
        if norm in ALL_TASKS:
            return norm

    return "unknown"


# ============================================================================
# PAIR GROUPING
# ============================================================================
def extract_pair_key(result: dict) -> str:
    """Extract image-level pair key (MME has 2 Qs/image)."""
    sample_id = result.get("id", "")
    if isinstance(sample_id, str) and sample_id.startswith("mme_"):
        parts = sample_id.rsplit("_", 2)
        if len(parts) == 3:
            return parts[0]
    return result.get("image_path", sample_id)


# ============================================================================
# CORE EVALUATION
# ============================================================================
def _eval_task(task_data: list, response_field: str) -> dict:
    total_q = len(task_data)
    correct_q = 0
    unknown_q = 0

    for r in task_data:
        resp = r.get(response_field, r.get("response", ""))
        pred = extract_yes_no(resp)
        gt = str(r.get("gt_answer", "")).strip().lower()

        if pred == "unknown":
            unknown_q += 1
        if pred == gt:
            correct_q += 1

    acc = correct_q / total_q if total_q > 0 else 0.0

    # ACC+ (pair-level)
    pairs = defaultdict(list)
    for r in task_data:
        pair_key = extract_pair_key(r)
        resp = r.get(response_field, r.get("response", ""))
        pred = extract_yes_no(resp)
        gt = str(r.get("gt_answer", "")).strip().lower()
        pairs[pair_key].append(pred == gt)

    total_pairs = len(pairs)
    correct_pairs = sum(1 for vals in pairs.values() if vals and all(vals))
    acc_plus = correct_pairs / total_pairs if total_pairs > 0 else 0.0

    score = (acc + acc_plus) * 100.0  # max 200

    return {
        "score": round(score, 2),
        "accuracy": round(acc * 100, 2),
        "accuracy_plus": round(acc_plus * 100, 2),
        "total_questions": total_q,
        "correct_questions": correct_q,
        "unknown_questions": unknown_q,
        "total_pairs": total_pairs,
        "correct_pairs": correct_pairs,
    }


def evaluate_mme(results: list, response_field: str = "response") -> dict:
    """Evaluate MME results following the official protocol."""
    task_results = defaultdict(list)
    unknown_task = []

    for r in results:
        task = extract_task_name(r)
        if task == "unknown":
            unknown_task.append(r)
            continue
        task_results[task].append(r)

    if unknown_task:
        print(f"  ⚠️  {len(unknown_task)} samples with unknown task (skipped)")

    task_scores = {}
    for task_name in sorted(task_results.keys()):
        task_scores[task_name] = _eval_task(task_results[task_name], response_field)

    perception_score = 0.0
    cognition_score = 0.0
    perception_tasks_found = []
    cognition_tasks_found = []

    for task_name, info in task_scores.items():
        if task_name in PERCEPTION_TASKS:
            perception_score += info["score"]
            perception_tasks_found.append(task_name)
        elif task_name in COGNITION_TASKS:
            cognition_score += info["score"]
            cognition_tasks_found.append(task_name)

    return {
        "total_score": round(perception_score + cognition_score, 2),
        "perception_score": round(perception_score, 2),
        "cognition_score": round(cognition_score, 2),
        "perception_max": len(perception_tasks_found) * 200,
        "cognition_max": len(cognition_tasks_found) * 200,
        "perception_tasks_found": perception_tasks_found,
        "cognition_tasks_found": cognition_tasks_found,
        "task_scores": task_scores,
        "unknown_task_count": len(unknown_task),
    }


# ============================================================================
# DISPLAY
# ============================================================================
def print_results(eval_result: dict, model_name: str = ""):
    ts = eval_result["task_scores"]

    print(f"\n{'='*80}")
    print(f"MME EVALUATION RESULTS{' — ' + model_name if model_name else ''}")
    print(f"{'='*80}")

    print(f"\n{'='*80}")
    print(f"  PERCEPTION (max {eval_result['perception_max']})")
    print(f"{'='*80}")
    print(f"  {'Task':<28} {'Score':>7} {'ACC':>8} {'ACC+':>8} {'#Q':>6} {'#Pairs':>7}")
    print(f"  {'-'*68}")
    p_total = 0.0
    for task in PERCEPTION_TASKS:
        if task in ts:
            info = ts[task]
            p_total += info['score']
            print(f"  {task:<28} {info['score']:>7.2f} {info['accuracy']:>7.2f}% {info['accuracy_plus']:>7.2f}% {info['total_questions']:>6} {info['total_pairs']:>7}")
    print(f"  {'-'*68}")
    print(f"  {'TOTAL':<28} {p_total:>7.2f}")

    print(f"\n{'='*80}")
    print(f"  COGNITION (max {eval_result['cognition_max']})")
    print(f"{'='*80}")
    print(f"  {'Task':<28} {'Score':>7} {'ACC':>8} {'ACC+':>8} {'#Q':>6} {'#Pairs':>7}")
    print(f"  {'-'*68}")
    c_total = 0.0
    for task in COGNITION_TASKS:
        if task in ts:
            info = ts[task]
            c_total += info['score']
            print(f"  {task:<28} {info['score']:>7.2f} {info['accuracy']:>7.2f}% {info['accuracy_plus']:>7.2f}% {info['total_questions']:>6} {info['total_pairs']:>7}")
    print(f"  {'-'*68}")
    print(f"  {'TOTAL':<28} {c_total:>7.2f}")

    print(f"\n{'='*80}")
    print("  OVERALL SCORES")
    print(f"{'='*80}")
    print(f"  Perception:  {eval_result['perception_score']:>8.2f} / {eval_result['perception_max']}")
    print(f"  Cognition:   {eval_result['cognition_score']:>8.2f} / {eval_result['cognition_max']}")
    print(f"  Total:       {eval_result['total_score']:>8.2f} / {eval_result['perception_max'] + eval_result['cognition_max']}")
    print(f"{'='*80}")


# ============================================================================
# DISCOVERY + PATH HELPERS (same style as eval_pope / eval_figstep)
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
        files = [f for f in files if "summary" not in f.name.lower() and "evaluated" not in f.name.lower() and "mme_eval" not in f.name.lower()]
        found_files.extend(files)

    return found_files


def infer_output_dir_from_result_file(result_file: str) -> str | None:
    """If result_file is under results/infer/<model>/<finding>/..., map to results/eval/<model>/<finding>."""
    try:
        p = Path(result_file)
        parts = list(p.parts)
        # Handle both relative and absolute paths; search for .../results/infer/<model>/<finding>/file
        for i in range(len(parts) - 4):
            if parts[i] == "results" and parts[i + 1] == "infer":
                model_dir = parts[i + 2]
                finding_name = parts[i + 3]
                return str(Path(*parts[:i], "results", "eval", model_dir, finding_name))
        # Relative path starts exactly with results/infer/...
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
def build_output_payload(results: list, eval_result: dict, result_file: str, response_field: str, model_name: str):
    return {
        "metadata": {
            "evaluator": "eval_mme.py (official MME protocol)",
            "evaluation_date": datetime.now().isoformat(),
            "evaluation_method": "mme_official",
            "reference": "Fu et al., MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models, arXiv:2306.13394",
            "result_file": os.path.basename(result_file),
            "response_field": response_field,
            "model": model_name,
            "total_samples": len(results),
        },
        "summary": {
            "total_samples": len(results),
            "total_score": eval_result["total_score"],
            "perception_score": eval_result["perception_score"],
            "cognition_score": eval_result["cognition_score"],
            "perception_max": eval_result["perception_max"],
            "cognition_max": eval_result["cognition_max"],
            "overall_max": eval_result["perception_max"] + eval_result["cognition_max"],
            "unknown_task_count": eval_result.get("unknown_task_count", 0),
        },
        "scores": {
            "total": eval_result["total_score"],
            "perception": eval_result["perception_score"],
            "cognition": eval_result["cognition_score"],
            "perception_max": eval_result["perception_max"],
            "cognition_max": eval_result["cognition_max"],
        },
        "task_scores": eval_result["task_scores"],
        "tasks_found": {
            "perception": eval_result["perception_tasks_found"],
            "cognition": eval_result["cognition_tasks_found"],
        },
    }


def evaluate_one_file(result_file: str, output_dir: str, response_field: str = "response") -> dict:
    """Evaluate one MME result file and save outputs. Returns a summary row."""
    with open(result_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    print(f"\nLoading: {result_file}")
    print(f"  Loaded {len(results)} samples")

    model_name = results[0].get("model", "") if results else ""
    eval_result = evaluate_mme(results, response_field=response_field)
    print_results(eval_result, model_name)

    os.makedirs(output_dir, exist_ok=True)

    stem = Path(result_file).stem
    ds_name = parse_dataset_name(Path(result_file))

    payload = build_output_payload(results, eval_result, result_file, response_field, model_name)

    # Main detailed file (same style as eval_pope/eval_figstep)
    evaluated_path = os.path.join(output_dir, f"{ds_name}_evaluated.json")
    with open(evaluated_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # Backward-compatible file name (old behavior)
    legacy_path = os.path.join(output_dir, f"{stem}_mme_eval.json")
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
        "total_samples": payload["summary"]["total_samples"],
        "total_score": payload["summary"]["total_score"],
        "perception_score": payload["summary"]["perception_score"],
        "cognition_score": payload["summary"]["cognition_score"],
        "perception_max": payload["summary"]["perception_max"],
        "cognition_max": payload["summary"]["cognition_max"],
        "overall_max": payload["summary"]["overall_max"],
        "unknown_task_count": payload["summary"]["unknown_task_count"],
    }
    for task in ALL_TASKS:
        if task in eval_result["task_scores"]:
            info = eval_result["task_scores"][task]
            row[f"score_{task}"] = info["score"]
            row[f"acc_{task}"] = info["accuracy"]
            row[f"accplus_{task}"] = info["accuracy_plus"]

    return row


def run_finding_evaluation(finding_name: str, output_dir: str, model_dir: str | None = None,
                           skip_neutral: bool = False, method: bool = False):
    """Evaluate all MME result files for a finding (same workflow as eval_pope/eval_figstep)."""
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
    print(f"MME EVALUATION — {finding_name.upper()}")
    print(f"{'='*80}")
    print("Method:  Official MME metrics (ACC, ACC+, Score)")
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

    # Save comparison files (same pattern as eval_pope/eval_figstep)
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
    print(f"MME COMPARISON — {finding_name.upper()}")
    print(f"{'='*80}")
    print(f"{'Condition':<30} {'Total':>8} {'Perc':>8} {'Cog':>8} {'N':>6}")
    print(f"{'-'*70}")
    for row in sorted(comparison_data, key=lambda x: x["total_score"], reverse=True):
        subj = f" [{row['subject']}]" if row.get("subject") else ""
        cond = f"{row['condition']}{subj}"
        print(f"{cond:<30} {row['total_score']:>7.2f} {row['perception_score']:>7.2f} {row['cognition_score']:>7.2f} {row['total_samples']:>5}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "finding": finding_name,
        "dataset": "mme",
        "evaluation_method": "mme_official",
        "reference": "Fu et al., MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models, arXiv:2306.13394",
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
        description="MME Benchmark Evaluation — Official Metrics (ACC + ACC+)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
MME Metrics (per task, max 200):
  ACC:  % of individual questions answered correctly
  ACC+: % of image pairs where BOTH questions correct
  Score = ACC + ACC+ (as percentage points, max 200)

Usage (recommended, same style as eval_pope/eval_figstep):
  python scripts/eval/eval_mme.py --model_dir llava_1_5_7b --finding mme_baseline

Single-file mode (still supported):
  python scripts/eval/eval_mme.py --result_file results/infer/llava_1_5_7b/mme_baseline/results_mme_full_baseline.json

Method mode (uses final_response field):
  python scripts/eval/eval_mme.py --model_dir llava_1_5_7b --finding mme_baseline --method
        """,
    )

    parser.add_argument("--result_file", type=str, default=None,
                        help="Path to one inference results JSON file (optional)")
    parser.add_argument("--finding", type=str, default="mme_baseline",
                        help="Finding folder name under results/infer/{model_dir}/")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Model directory under results/infer/ (e.g., llava_1_5_7b)")
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
        print("MME EVALUATION (Single-file mode)")
        print(f"{'='*80}")
        print(f"Input:    {args.result_file}")
        print(f"Output:   {output_dir}")
        print(f"Method:   Official MME metrics (ACC, ACC+, Score)")
        print(f"Field:    {response_field}")

        evaluate_one_file(args.result_file, output_dir=output_dir, response_field=response_field)

        print(f"\n{'='*80}")
        print("✅ MME EVALUATION COMPLETE")
        print(f"{'='*80}")
        return

    # Finding-level mode (preferred)
    eval_subdir = args.model_dir if args.model_dir else "all_models"
    output_dir = args.output_dir or os.path.join(EVAL_BASE_DIR, eval_subdir, args.finding)

    print(f"\n{'='*80}")
    print("MME EVALUATION (Fu et al., arXiv 2023)")
    print(f"{'='*80}")
    print(f"Finding:  {args.finding}")
    print(f"Input:    {INFER_BASE_DIR}/{args.model_dir or '*'}/{args.finding}/")
    print(f"Output:   {output_dir}")
    print("Method:   Official MME metrics (ACC, ACC+, Score)")
    print(f"Field:    {'final_response' if args.method else 'response'}")

    run_finding_evaluation(
        finding_name=args.finding,
        output_dir=output_dir,
        model_dir=args.model_dir,
        skip_neutral=args.skip_neutral,
        method=args.method,
    )

    print(f"\n{'='*80}")
    print("✅ MME EVALUATION COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

"""
MMVP Benchmark Evaluation Script — Pair-based Accuracy

Follows the same output workflow/style as eval_mme.py:
  - Supports finding-level evaluation via --model_dir + --finding
  - Saves outputs under results/eval/{model_dir}/{finding}/ by default
  - Still supports single-file evaluation via --result_file
  - Writes per-file evaluated JSON + comparison CSV/JSON + summary JSON

MMVP Metric (paper: "Eyes Wide Shut?", CVPR 2024):
  - Pair Accuracy: a pair is CORRECT only when BOTH questions in the pair
    are answered correctly.
  - Pairs are derived from question_id: (1,2), (3,4), (5,6), ...
    i.e. pair_id = (question_id - 1) // 2
  - Reports per image_type breakdown + overall pair accuracy.

Response field:
  - Baseline inference  → "response"       (default)
  - Method  inference   → "final_response" (--method flag)

NO external API calls — evaluation is purely local string matching.
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
# CONSTANT PATHS  (mirror eval_mme.py)
# ============================================================================
INFER_BASE_DIR = "results/infer"
EVAL_BASE_DIR  = "results/eval"


# ============================================================================
# ANSWER EXTRACTION
# ============================================================================
def extract_answer(text: str) -> str:
    """
    Extract the option letter (a/b/c/d) from a model response.

    Handles common formats produced by instruction-tuned models:
      "(a) Open"  →  "a"
      "a"         →  "a"
      "A"         →  "a"
      "The answer is (b)"  →  "b"
      "b) Closed" →  "b"
    """
    if not text:
        return "unknown"

    t = str(text).strip().lower()

    # 1) Exact single letter (possibly with trailing punctuation)
    if re.fullmatch(r"[a-d][.):]?", t):
        return t[0]

    patterns = [
        r"answer\s*(?:is|:)\s*[(\[]?\s*([a-d])\b",  # "answer is (a)"
        r"^[(\[]\s*([a-d])\s*[)\]]",                  # "(a)" at start
        r"^([a-d])\s*[).\]:]",                        # "a)" or "a." at start
        r"\b([a-d])\s*[).\]:]",                       # "a)" anywhere
        r"\b([a-d])\b",                               # standalone letter
    ]
    for pattern in patterns:
        m = re.search(pattern, t)
        if m:
            return m.group(1)

    return "unknown"


# ============================================================================
# PAIR DERIVATION
# ============================================================================
def get_pair_id(sample: dict) -> int:
    """
    Derive pair_id from question_id.
    question_id 1,2 → pair 0
    question_id 3,4 → pair 1
    question_id 5,6 → pair 2  ...etc
    """
    qid = sample.get("question_id")
    if qid is not None:
        return (int(qid) - 1) // 2
    # Fallback: try to parse from id string e.g. "mmvp_0001" → 0
    sid = str(sample.get("id", ""))
    m = re.search(r"(\d+)$", sid)
    if m:
        return (int(m.group(1)) - 1) // 2
    return -1


def is_first_in_pair(sample: dict) -> bool:
    """
    Odd question_id  → first in pair
    Even question_id → second in pair
    """
    qid = sample.get("question_id")
    if qid is not None:
        return int(qid) % 2 == 1
    return True


# ============================================================================
# CORE EVALUATION
# ============================================================================
def _eval_group(samples: list, response_field: str) -> dict:
    """Compute pair-based accuracy for a group of samples."""
    total_q   = len(samples)
    correct_q = 0
    unknown_q = 0

    # pid → {0: correct_bool, 1: correct_bool}
    pairs: dict = defaultdict(dict)

    for s in samples:
        resp    = s.get(response_field) or s.get("response", "")
        gt      = str(s.get("gt_answer", "")).strip().lower()
        pred    = extract_answer(resp)
        correct = (pred == gt)

        if pred == "unknown":
            unknown_q += 1
        if correct:
            correct_q += 1

        pid  = get_pair_id(s)
        slot = 0 if is_first_in_pair(s) else 1
        pairs[pid][slot] = correct

    total_pairs   = 0
    correct_pairs = 0
    for pid, pair in pairs.items():
        if 0 not in pair or 1 not in pair:
            continue   # incomplete pair — skip
        total_pairs += 1
        if pair[0] and pair[1]:
            correct_pairs += 1

    pair_acc = correct_pairs / total_pairs * 100 if total_pairs > 0 else 0.0
    q_acc    = correct_q    / total_q    * 100 if total_q    > 0 else 0.0

    return {
        "pair_accuracy":     round(pair_acc, 2),
        "question_accuracy": round(q_acc,    2),
        "total_questions":   total_q,
        "correct_questions": correct_q,
        "unknown_questions": unknown_q,
        "total_pairs":       total_pairs,
        "correct_pairs":     correct_pairs,
    }


def evaluate_mmvp(results: list, response_field: str = "response") -> dict:
    """Full MMVP evaluation: per image_type + overall."""
    pattern_results: dict = defaultdict(list)

    for r in results:
        pattern = r.get("image_type") or r.get("visual_pattern") or "unknown"
        pattern_results[pattern].append(r)

    pattern_scores: dict = {}
    for pattern in sorted(pattern_results.keys()):
        pattern_scores[pattern] = _eval_group(pattern_results[pattern], response_field)

    overall = _eval_group(results, response_field)

    return {
        "overall":        overall,
        "pattern_scores": pattern_scores,
    }


# ============================================================================
# DISPLAY
# ============================================================================
def print_results(eval_result: dict, model_name: str = ""):
    ps = eval_result["pattern_scores"]
    ov = eval_result["overall"]

    print(f"\n{'='*75}")
    print(f"MMVP EVALUATION RESULTS{' — ' + model_name if model_name else ''}")
    print(f"{'='*75}")
    print(f"  {'Visual Pattern':<28} {'PairAcc':>8} {'QAcc':>8} {'Pairs':>7} {'Corr':>6} {'#Q':>6}")
    print(f"  {'-'*65}")
    for pattern in sorted(ps.keys()):
        info = ps[pattern]
        print(
            f"  {pattern:<28} "
            f"{info['pair_accuracy']:>7.1f}% "
            f"{info['question_accuracy']:>7.1f}% "
            f"{info['total_pairs']:>7} "
            f"{info['correct_pairs']:>6} "
            f"{info['total_questions']:>6}"
        )
    print(f"  {'-'*65}")
    print(
        f"  {'OVERALL':<28} "
        f"{ov['pair_accuracy']:>7.1f}% "
        f"{ov['question_accuracy']:>7.1f}% "
        f"{ov['total_pairs']:>7} "
        f"{ov['correct_pairs']:>6} "
        f"{ov['total_questions']:>6}"
    )
    print(f"{'='*75}")


# ============================================================================
# DISCOVERY + PATH HELPERS  (mirror eval_mme.py)
# ============================================================================
def discover_result_files(finding_name: str, model_dir=None):
    found_files = []
    if model_dir:
        search_dirs = [os.path.join(INFER_BASE_DIR, model_dir, finding_name)]
    else:
        base = Path(INFER_BASE_DIR)
        if not base.exists():
            return []
        search_dirs = [
            str(child / finding_name)
            for child in sorted(base.iterdir())
            if child.is_dir() and (child / finding_name).exists()
        ]

    for search_dir in search_dirs:
        search_path = Path(search_dir)
        if not search_path.exists():
            continue
        files = sorted(search_path.glob("results_*.json"))
        files = [
            f for f in files
            if "summary"    not in f.name.lower()
            and "evaluated" not in f.name.lower()
            and "mmvp_eval" not in f.name.lower()
        ]
        found_files.extend(files)

    return found_files


def infer_output_dir_from_result_file(result_file: str):
    """Map results/infer/<model>/<finding>/file → results/eval/<model>/<finding>."""
    try:
        p = Path(result_file)
        parts = list(p.parts)
        for i in range(len(parts) - 4):
            if parts[i] == "results" and parts[i + 1] == "infer":
                return str(Path(*parts[:i], "results", "eval", parts[i + 2], parts[i + 3]))
        norm = str(p).replace("\\", "/")
        if "results/infer/" in norm:
            tail = norm.split("results/infer/", 1)[1]
            segs = tail.split("/")
            if len(segs) >= 3:
                return os.path.join("results", "eval", segs[0], segs[1])
    except Exception:
        return None
    return None


def parse_dataset_name(result_path: Path) -> str:
    name = result_path.stem
    return name[len("results_"):] if name.startswith("results_") else name


def _extract_condition_fields(ds_name: str, first_result: dict):
    ds_upper         = ds_name.upper()
    emotion_category = first_result.get("emotion_category", "neutral")
    condition        = "neutral" if "NEUTRAL" in ds_upper else emotion_category
    subject = ""
    if "_I_" in ds_upper and "_YOU_" not in ds_upper:
        subject = "I"
    elif "_YOU_" in ds_upper:
        subject = "You"
    return condition, emotion_category, subject


# ============================================================================
# SAVE / RUN HELPERS
# ============================================================================
def build_output_payload(results, eval_result, result_file, response_field, model_name):
    ov = eval_result["overall"]
    return {
        "metadata": {
            "evaluator":         "eval_mmvp.py (MMVP pair-accuracy protocol)",
            "evaluation_date":   datetime.now().isoformat(),
            "evaluation_method": "mmvp_pair_accuracy",
            "reference":         "Zhang et al., Eyes Wide Shut? Exploring the Visual Shortcomings of MLLMs, CVPR 2024",
            "result_file":       os.path.basename(result_file),
            "response_field":    response_field,
            "model":             model_name,
            "total_samples":     len(results),
        },
        "summary": {
            "total_samples":     len(results),
            "total_pairs":       ov["total_pairs"],
            "correct_pairs":     ov["correct_pairs"],
            "pair_accuracy":     ov["pair_accuracy"],
            "question_accuracy": ov["question_accuracy"],
        },
        "overall":        ov,
        "pattern_scores": eval_result["pattern_scores"],
    }


def evaluate_one_file(result_file: str, output_dir: str, response_field: str = "response") -> dict:
    """Evaluate one MMVP result file and save outputs. Returns a summary row."""
    with open(result_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    print(f"\nLoading: {result_file}")
    print(f"  Loaded {len(results)} samples")

    model_name  = results[0].get("model", "") if results else ""
    eval_result = evaluate_mmvp(results, response_field=response_field)
    print_results(eval_result, model_name)

    os.makedirs(output_dir, exist_ok=True)

    stem    = Path(result_file).stem
    ds_name = parse_dataset_name(Path(result_file))
    payload = build_output_payload(results, eval_result, result_file, response_field, model_name)

    evaluated_path = os.path.join(output_dir, f"{ds_name}_evaluated.json")
    legacy_path    = os.path.join(output_dir, f"{stem}_mmvp_eval.json")

    for path in (evaluated_path, legacy_path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Evaluation saved: {legacy_path}")
    if legacy_path != evaluated_path:
        print(f"✅ Detailed eval  : {evaluated_path}")

    first_result = results[0] if results else {}
    finding_name = Path(result_file).parent.name
    condition, emotion_category, subject = _extract_condition_fields(ds_name, first_result)
    ov = eval_result["overall"]

    row = {
        "dataset":           ds_name,
        "finding":           finding_name,
        "condition":         condition,
        "emotion_category":  emotion_category,
        "subject":           subject,
        "total_samples":     len(results),
        "total_pairs":       ov["total_pairs"],
        "correct_pairs":     ov["correct_pairs"],
        "pair_accuracy":     ov["pair_accuracy"],
        "question_accuracy": ov["question_accuracy"],
    }
    for pattern, info in eval_result["pattern_scores"].items():
        safe = pattern.replace(" ", "_").replace("/", "_")
        row[f"pair_acc_{safe}"] = info["pair_accuracy"]
        row[f"pairs_{safe}"]    = info["total_pairs"]
        row[f"correct_{safe}"]  = info["correct_pairs"]

    return row


def run_finding_evaluation(
    finding_name: str, output_dir: str,
    model_dir=None, skip_neutral: bool = False, method: bool = False,
):
    result_files = discover_result_files(finding_name, model_dir)
    if not result_files:
        print(f"❌ No result files found for finding='{finding_name}'")
        if model_dir:
            print(f"   Expected: {INFER_BASE_DIR}/{model_dir}/{finding_name}/results_*.json")
        return None

    if skip_neutral:
        result_files = [f for f in result_files if "NEUTRAL" not in f.name.upper()]

    os.makedirs(output_dir, exist_ok=True)
    response_field = "final_response" if method else "response"

    print(f"\n{'='*80}")
    print(f"MMVP EVALUATION — {finding_name.upper()}")
    print(f"{'='*80}")
    print(f"Method : MMVP pair-based accuracy (Zhang et al., CVPR 2024)")
    print(f"Field  : {response_field}")
    print(f"Files  : {len(result_files)}")
    for f in result_files:
        print(f"  - {f.name}")
    print(f"{'='*80}\n")

    comparison_data = []
    for idx, rf in enumerate(result_files, 1):
        print(f"\n[{idx}/{len(result_files)}] {rf.name}")
        row = evaluate_one_file(str(rf), output_dir=output_dir, response_field=response_field)
        comparison_data.append(row)

    if not comparison_data:
        print("❌ No evaluations completed.")
        return None

    if HAS_PANDAS:
        try:
            pd.DataFrame(comparison_data).to_csv(
                os.path.join(output_dir, f"{finding_name}_comparison.csv"), index=False
            )
        except Exception:
            pass

    with open(os.path.join(output_dir, f"{finding_name}_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(comparison_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*80}")
    print(f"MMVP COMPARISON — {finding_name.upper()}")
    print(f"{'='*80}")
    print(f"{'Condition':<30} {'PairAcc':>9} {'QAcc':>8} {'Pairs':>7} {'N':>6}")
    print(f"{'-'*65}")
    for row in sorted(comparison_data, key=lambda x: x["pair_accuracy"], reverse=True):
        subj = f" [{row['subject']}]" if row.get("subject") else ""
        print(
            f"{row['condition']}{subj:<30} "
            f"{row['pair_accuracy']:>8.1f}% "
            f"{row['question_accuracy']:>7.1f}% "
            f"{row['total_pairs']:>7} "
            f"{row['total_samples']:>5}"
        )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(
        os.path.join(output_dir, f"{finding_name}_eval_summary_{ts}.json"), "w", encoding="utf-8"
    ) as f:
        json.dump({
            "finding":           finding_name,
            "dataset":           "mmvp",
            "evaluation_method": "mmvp_pair_accuracy",
            "reference":         "Zhang et al., Eyes Wide Shut? Exploring the Visual Shortcomings of MLLMs, CVPR 2024",
            "evaluation_date":   datetime.now().isoformat(),
            "response_field":    response_field,
            "files_evaluated":   len(comparison_data),
            "comparison":        comparison_data,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n✅ {finding_name} evaluation complete → {output_dir}")
    return comparison_data


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="MMVP Benchmark Evaluation — Pair-based Accuracy (Zhang et al., CVPR 2024)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
MMVP Metric:
  Pairs are derived from question_id: (1,2), (3,4), (5,6) ...
  A pair is CORRECT only if BOTH questions are answered correctly.
  Reports per image_type + overall pair accuracy.

Usage (finding-level, same style as eval_mme):
  python scripts/eval/eval_mmvp.py --model_dir llava_1_5_7b --finding mmvp_baseline

Single-file mode:
  python scripts/eval/eval_mmvp.py --result_file /workspace/results/method1/qwen2_vl_7b__gemma_3_12b_it/mmvp/fixed/negative_low/start/multi2/method1_results_20260321_112004.json --method

Method mode (reads final_response instead of response):
  python scripts/eval/eval_mmvp.py --model_dir llava_1_5_7b --finding mmvp_baseline --method
        """,
    )
    parser.add_argument("--result_file",  type=str, default=None)
    parser.add_argument("--finding",      type=str, default="mmvp_baseline")
    parser.add_argument("--model_dir",    type=str, default=None)
    parser.add_argument("--method",       action="store_true",
                        help="Use 'final_response' field instead of 'response'")
    parser.add_argument("--skip_neutral", action="store_true")
    parser.add_argument("--output_dir",   type=str, default=None)

    args = parser.parse_args()

    # ── Single-file mode ──────────────────────────────────────────────────────
    if args.result_file:
        if not os.path.exists(args.result_file):
            print(f"❌ Result file not found: {args.result_file}")
            return

        output_dir     = (
            args.output_dir
            or infer_output_dir_from_result_file(args.result_file)
            or os.path.dirname(args.result_file)
        )
        response_field = "final_response" if args.method else "response"

        print(f"\n{'='*80}")
        print("MMVP EVALUATION (Single-file mode)")
        print(f"{'='*80}")
        print(f"Input  : {args.result_file}")
        print(f"Output : {output_dir}")
        print(f"Method : MMVP pair-based accuracy (Zhang et al., CVPR 2024)")
        print(f"Field  : {response_field}")

        evaluate_one_file(args.result_file, output_dir=output_dir, response_field=response_field)

        print(f"\n{'='*80}")
        print("✅ MMVP EVALUATION COMPLETE")
        print(f"{'='*80}")
        return

    # ── Finding-level mode ────────────────────────────────────────────────────
    eval_subdir = args.model_dir or "all_models"
    output_dir  = args.output_dir or os.path.join(EVAL_BASE_DIR, eval_subdir, args.finding)

    print(f"\n{'='*80}")
    print("MMVP EVALUATION (Zhang et al., CVPR 2024)")
    print(f"{'='*80}")
    print(f"Finding: {args.finding}")
    print(f"Input  : {INFER_BASE_DIR}/{args.model_dir or '*'}/{args.finding}/")
    print(f"Output : {output_dir}")
    print(f"Field  : {'final_response' if args.method else 'response'}")

    run_finding_evaluation(
        finding_name=args.finding,
        output_dir=output_dir,
        model_dir=args.model_dir,
        skip_neutral=args.skip_neutral,
        method=args.method,
    )

    print(f"\n{'='*80}")
    print("✅ MMVP EVALUATION COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
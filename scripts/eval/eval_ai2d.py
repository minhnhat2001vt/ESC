"""
AI2D Benchmark Evaluation Script — Official Metrics (Accuracy)

AI2D evaluates diagram understanding via multiple-choice questions
over scientific diagrams. Variable number of choices per question.

Metric: Simple accuracy (correct / total).
Evaluation: Exact match on option letter.

Follows the same workflow/style as eval_mathvista.py:
- Supports finding-level evaluation via --model_dir + --finding
- Saves outputs under results/eval/{model_dir}/{finding}/
- Supports single-file evaluation via --result_file
- Writes per-file evaluated JSON + comparison CSV/JSON + summary JSON
- Supports --method flag for final_response (method) vs response (baseline)

Reference:
    A Diagram Is Worth A Dozen Images (Kembhavi et al., 2016)
    https://allenai.org/data/diagrams
"""

import json
import os
import re
import sys
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
_SCRIPT_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _SCRIPT_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from path_config import RESULTS_ROOT

INFER_BASE_DIR = str(RESULTS_ROOT / "infer")
EVAL_BASE_DIR = str(RESULTS_ROOT / "eval")


# ============================================================================
# ANSWER EXTRACTION — MULTI-CHOICE (variable number of options)
# ============================================================================
def extract_answer_multichoice(response: str, choices: list = None) -> str:
    """
    Extract option letter from model response for multi_choice questions.
    Handles variable number of choices (not always 4).

    Returns: letter like "A", "B", "C", "D" or "FAILED"
    """
    if not response or not response.strip():
        return "FAILED"

    num_choices = len(choices) if choices else 4
    valid = [chr(ord("A") + i) for i in range(num_choices)]
    valid_set = set(valid)
    resp = response.strip()

    # Level 1: Response is just the letter
    cleaned = resp.upper().strip("().- \t\n")
    if cleaned in valid_set:
        return cleaned

    match = re.match(r'^\(?([A-Z])\)?[\.\)\s]*$', resp, re.IGNORECASE)
    if match and match.group(1).upper() in valid_set:
        return match.group(1).upper()

    resp_upper = resp.upper()

    # Level 2: "The answer is X" patterns
    patterns = [
        r'(?:the\s+)?answer\s+is\s*[:\s]*\(?([A-Z])\)?',
        r'(?:I\s+)?(?:choose|select)\s*[:\s]*\(?([A-Z])\)?',
        r'(?:correct\s+)?(?:answer|option)\s*[:\s]*\(?([A-Z])\)?',
        r'ANSWER:\s*\(?([A-Z])\)?',
    ]
    for pat in patterns:
        m = re.search(pat, resp_upper)
        if m and m.group(1) in valid_set:
            return m.group(1)

    # Level 3: Letter in parens
    for m in re.findall(r'\(([A-Z])\)', resp_upper):
        if m in valid_set:
            return m

    # Level 4: First letter at start
    m = re.match(r'^([A-Z])[\.\)\s]', resp)
    if m and m.group(1).upper() in valid_set:
        return m.group(1).upper()

    # Level 5: Match by answer TEXT in choices
    if choices:
        for i, c in enumerate(choices):
            c_str = str(c).strip().lower()
            if c_str and c_str in resp.lower():
                return chr(ord("A") + i)

    # Level 6: Any standalone letter
    for letter in valid:
        if re.search(r'\b' + letter + r'\b', resp_upper):
            return letter

    return "FAILED"


def _extract_options_from_question(full_question: str) -> list:
    """Extract options list from the formatted question string.
    Parses '(A) text\\n(B) text\\n...' patterns."""
    options = []
    pattern = r'\(([A-Z])\)\s*(.+?)(?=\n\([A-Z]\)|\nAnswer|\Z)'
    matches = re.findall(pattern, full_question, re.DOTALL)
    # Sort by letter to ensure correct ordering
    matches.sort(key=lambda x: x[0])
    for letter, text in matches:
        options.append(text.strip())
    return options


def _resolve_gt_letter(r: dict) -> str:
    """Resolve ground-truth answer letter from various field formats.
    
    Handles:
      - gt_answer_letter directly available
      - gt_answer is already a single letter (A-Z)
      - gt_answer is answer TEXT that needs mapping back to letter via options
    """
    meta = r.get("metadata", {}) if isinstance(r.get("metadata"), dict) else {}
    
    # Priority 1: gt_answer_letter field
    gt_letter = r.get("gt_answer_letter", "") or meta.get("gt_answer_letter", "")
    if gt_letter and len(gt_letter) == 1 and gt_letter.upper() in "ABCDEFGHIJ":
        return gt_letter.upper()
    
    gt_answer = r.get("gt_answer", "") or meta.get("gt_answer", "")
    if not gt_answer:
        return ""
    
    # Priority 2: gt_answer is already a single letter AND not an option text
    # (We must check it's not ambiguous — e.g., option text could be "D")
    # Only use this if no choices are available to cross-check
    
    # Priority 3: Map gt_answer text to letter using choices
    choices = r.get("choices", []) or meta.get("choices", [])
    if not choices:
        # Try extracting from full_question
        full_q = r.get("full_question", "") or meta.get("formatted_question", "")
        choices = _extract_options_from_question(full_q)
    
    if choices:
        gt_lower = str(gt_answer).strip().lower()
        for i, c in enumerate(choices):
            if str(c).strip().lower() == gt_lower:
                return chr(ord("A") + i)
    
    # Priority 4: If gt_answer is a single uppercase letter and no choices to cross-check
    if len(gt_answer) == 1 and gt_answer.upper() in "ABCDEFGHIJ":
        return gt_answer.upper()
    
    return ""


# ============================================================================
# CORE EVALUATION
# ============================================================================
def _eval_one_sample(r: dict, response_field: str) -> dict:
    """Evaluate a single AI2D sample. Returns eval fields."""
    resp = r.get(response_field, r.get("response", ""))

    gt_letter = _resolve_gt_letter(r)
    
    choices = r.get("choices", [])
    meta = r.get("metadata", {}) if isinstance(r.get("metadata"), dict) else {}
    if not choices:
        choices = meta.get("choices", [])
    if not choices:
        choices = _extract_options_from_question(r.get("full_question", ""))

    predicted = extract_answer_multichoice(resp, choices)
    extraction_success = predicted != "FAILED"
    is_correct = extraction_success and gt_letter != "" and (predicted == gt_letter)

    return {
        "predicted_answer": predicted,
        "gt_answer_letter": gt_letter,
        "is_correct": is_correct,
        "extraction_success": extraction_success,
    }


def _eval_ai2d(results: list, response_field: str = "response") -> dict:
    """Evaluate AI2D results. Metric: accuracy."""
    total = len(results)
    correct = 0
    extracted = 0

    num_choices_stats = defaultdict(lambda: {"total": 0, "correct": 0})

    for r in results:
        ev = _eval_one_sample(r, response_field)
        if ev["is_correct"]:
            correct += 1
        if ev["extraction_success"]:
            extracted += 1

        meta = r.get("metadata", {}) if isinstance(r.get("metadata"), dict) else {}
        n_choices = meta.get("num_choices", len(meta.get("choices", []))) or "unknown"
        num_choices_stats[str(n_choices)]["total"] += 1
        if ev["is_correct"]:
            num_choices_stats[str(n_choices)]["correct"] += 1

    def _finalize(stats):
        return {
            k: {**v, "accuracy": round(v["correct"] / v["total"] * 100, 2) if v["total"] else 0}
            for k, v in sorted(stats.items())
        }

    overall_acc = correct / total if total > 0 else 0.0

    return {
        "total_samples": total,
        "correct": correct,
        "extracted": extracted,
        "overall_accuracy": round(overall_acc * 100, 2),
        "extraction_rate": round(extracted / total * 100, 2) if total else 0,
        "by_num_choices": _finalize(num_choices_stats),
    }


# ============================================================================
# DISPLAY
# ============================================================================
def print_results(eval_result: dict, model_name: str = ""):
    print(f"\n{'='*80}")
    print(f"AI2D EVALUATION RESULTS{' — ' + model_name if model_name else ''}")
    print(f"{'='*80}")

    print(f"\n  Total samples:      {eval_result['total_samples']}")
    print(f"  Extraction rate:    {eval_result['extraction_rate']}%")
    print(f"  Overall Accuracy:   {eval_result['overall_accuracy']}%")

    by_nc = eval_result.get("by_num_choices", {})
    if by_nc:
        print(f"\n  By Number of Choices:")
        print(f"    {'#Choices':<15s} {'Total':>6s} {'Correct':>8s} {'Acc%':>7s}")
        print(f"    {'-'*15} {'-'*6} {'-'*8} {'-'*7}")
        for name, st in by_nc.items():
            print(f"    {name:<15s} {st['total']:>6d} {st['correct']:>8d} "
                  f"{st['accuracy']:>6.1f}%")

    print(f"{'='*80}")


# ============================================================================
# DISCOVERY + PATH HELPERS (same style as eval_mathvista)
# ============================================================================
def discover_result_files(finding_name: str, model_dir: str | None = None):
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
                 and "ai2d_eval" not in f.name.lower()]
        found_files.extend(files)

    return found_files


def infer_output_dir_from_result_file(result_file: str) -> str | None:
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


def _extract_condition_fields(ds_name: str, first_result: dict) -> tuple:
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
def build_output_payload(results, eval_result, result_file, response_field, model_name):
    return {
        "metadata": {
            "evaluator": "eval_ai2d.py (AI2D official protocol — accuracy)",
            "evaluation_date": datetime.now().isoformat(),
            "evaluation_method": "ai2d_accuracy",
            "reference": "Kembhavi et al., A Diagram Is Worth A Dozen Images (2016)",
            "result_file": os.path.basename(result_file),
            "response_field": response_field,
            "model": model_name,
            "total_samples": len(results),
        },
        "summary": {
            "total_samples": eval_result["total_samples"],
            "correct": eval_result["correct"],
            "extracted": eval_result["extracted"],
            "overall_accuracy": eval_result["overall_accuracy"],
            "extraction_rate": eval_result["extraction_rate"],
        },
        "scores": {
            "overall_accuracy": eval_result["overall_accuracy"],
        },
        "by_num_choices": eval_result.get("by_num_choices", {}),
    }


def evaluate_one_file(result_file: str, output_dir: str, response_field: str = "response") -> dict:
    with open(result_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    if isinstance(results, dict) and "results" in results:
        results = results["results"]

    print(f"\nLoading: {result_file}")
    print(f"  Loaded {len(results)} samples")

    model_name = results[0].get("model", "") if results else ""
    eval_result = _eval_ai2d(results, response_field=response_field)
    print_results(eval_result, model_name)

    os.makedirs(output_dir, exist_ok=True)

    stem = Path(result_file).stem
    ds_name = parse_dataset_name(Path(result_file))

    payload = build_output_payload(results, eval_result, result_file, response_field, model_name)

    evaluated_path = os.path.join(output_dir, f"{ds_name}_evaluated.json")
    with open(evaluated_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    legacy_path = os.path.join(output_dir, f"{stem}_ai2d_eval.json")
    with open(legacy_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Evaluation saved: {legacy_path}")

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
        "overall_accuracy": eval_result["overall_accuracy"],
        "extraction_rate": eval_result["extraction_rate"],
    }
    return row


def run_finding_evaluation(finding_name: str, output_dir: str, model_dir: str | None = None,
                           skip_neutral: bool = False, method: bool = False):
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
    print(f"AI2D EVALUATION — {finding_name.upper()}")
    print(f"{'='*80}")
    print("Method:  AI2D official metrics (Accuracy)")
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
    print(f"AI2D COMPARISON — {finding_name.upper()}")
    print(f"{'='*80}")
    print(f"{'Condition':<30} {'Acc%':>8} {'Extract%':>9} {'N':>6}")
    print(f"{'-'*60}")
    for row in sorted(comparison_data, key=lambda x: x["overall_accuracy"], reverse=True):
        subj = f" [{row['subject']}]" if row.get("subject") else ""
        cond = f"{row['condition']}{subj}"
        print(f"{cond:<30} {row['overall_accuracy']:>7.2f} {row['extraction_rate']:>8.2f} "
              f"{row['total_samples']:>5}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "finding": finding_name,
        "dataset": "ai2d",
        "evaluation_method": "ai2d_accuracy",
        "reference": "Kembhavi et al., A Diagram Is Worth A Dozen Images (2016)",
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
        description="AI2D Benchmark Evaluation — Official Metrics (Accuracy)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
AI2D Metrics:
  Accuracy on multiple-choice diagram questions.
  Variable number of choices per question.

Usage (recommended):
  python scripts/eval/eval_ai2d.py --model_dir llava_1_5_7b --finding ai2d_baseline

Single-file mode:
  python scripts/eval/eval_ai2d.py --result_file /path/to/results.json --method
        """,
    )

    parser.add_argument("--result_file", type=str, default=None)
    parser.add_argument("--finding", type=str, default="ai2d_baseline")
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--method", action="store_true")
    parser.add_argument("--skip_neutral", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)

    args = parser.parse_args()

    if args.result_file:
        if not os.path.exists(args.result_file):
            print(f"❌ Result file not found: {args.result_file}")
            return
        output_dir = args.output_dir or infer_output_dir_from_result_file(args.result_file) or os.path.dirname(args.result_file)
        response_field = "final_response" if args.method else "response"

        print(f"\n{'='*80}")
        print("AI2D EVALUATION (Single-file mode)")
        print(f"{'='*80}")
        print(f"Input:  {args.result_file}")
        print(f"Field:  {response_field}")

        evaluate_one_file(args.result_file, output_dir=output_dir, response_field=response_field)

        print(f"\n✅ AI2D EVALUATION COMPLETE")
        return

    eval_subdir = args.model_dir if args.model_dir else "all_models"
    output_dir = args.output_dir or os.path.join(EVAL_BASE_DIR, eval_subdir, args.finding)

    print(f"\n{'='*80}")
    print("AI2D EVALUATION (Kembhavi et al., 2016)")
    print(f"{'='*80}")
    print(f"Finding: {args.finding}")
    print(f"Output:  {output_dir}")

    run_finding_evaluation(
        finding_name=args.finding,
        output_dir=output_dir,
        model_dir=args.model_dir,
        skip_neutral=args.skip_neutral,
        method=args.method,
    )

    print(f"\n✅ AI2D EVALUATION COMPLETE")


if __name__ == "__main__":
    main()
"""
MathVista Benchmark Evaluation Script — Official Metrics (Accuracy)

Updated to match the output workflow/style of eval_mme.py:
- Supports finding-level evaluation via --model_dir + --finding
- Saves outputs under results/eval/{model_dir}/{finding}/ by default
- Still supports single-file evaluation via --result_file
- Writes per-file evaluated JSON + comparison CSV/JSON + summary JSON
- Supports --method flag for final_response (method) vs response (baseline)

Reference:
    MathVista: Evaluating Mathematical Reasoning of Foundation Models
    in Visual Contexts (ICLR 2024)
    https://github.com/lupantech/MathVista
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


# ============================================================================
# ANSWER EXTRACTION — MULTI-CHOICE
# ============================================================================
def extract_answer_multichoice(response: str, choices: list = None) -> str:
    """
    Extract option letter from model response for multi_choice questions.

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


# ============================================================================
# ANSWER EXTRACTION — FREE-FORM
# ============================================================================
def extract_answer_freeform(response: str, answer_type: str = "integer") -> str:
    """
    Extract numerical or text answer from model response for free_form questions.

    Returns: extracted answer string or "FAILED"
    """
    if not response or not response.strip():
        return "FAILED"

    resp = response.strip()

    # Try to find "the answer is X" pattern first
    patterns = [
        r'(?:the\s+)?(?:final\s+)?answer\s+is\s*[:\s]*([^\.\n,]+)',
        r'(?:answer|result)\s*[=:]\s*([^\.\n,]+)',
        r'=\s*([^\.\n,]+)$',
    ]
    for pat in patterns:
        m = re.search(pat, resp, re.IGNORECASE)
        if m:
            extracted = m.group(1).strip().rstrip(".")
            if extracted:
                return extracted

    if answer_type in ("integer", "float"):
        # Extract last number mentioned (usually the final answer)
        numbers = re.findall(r'-?\d+\.?\d*', resp)
        if numbers:
            return numbers[-1]

    # For text answers, take the last line or last sentence
    lines = [l.strip() for l in resp.strip().split("\n") if l.strip()]
    if lines:
        last = lines[-1]
        for prefix in ["Therefore, ", "So, ", "Thus, ", "Hence, ",
                        "The answer is ", "Answer: "]:
            if last.lower().startswith(prefix.lower()):
                last = last[len(prefix):].strip()
        return last.rstrip(".")

    return resp.strip()


# ============================================================================
# ANSWER NORMALIZATION & COMPARISON (from MathVista paper)
# ============================================================================
def normalize_answer(answer: str, answer_type: str = "text") -> str:
    """Normalize answer for comparison."""
    if answer is None:
        return ""
    ans = str(answer).strip()

    # Remove common units and symbols
    ans = ans.replace("°", "").replace("%", "").replace("$", "")
    ans = ans.replace(",", "")  # remove thousand separators
    ans = ans.strip().rstrip(".")

    if answer_type in ("integer", "float"):
        m = re.search(r'-?\d+\.?\d*', ans)
        if m:
            ans = m.group(0)
            try:
                val = float(ans)
                if answer_type == "integer":
                    ans = str(int(round(val)))
                else:
                    ans = str(val)
            except ValueError:
                pass

    return ans.lower().strip()


def compare_answers(predicted: str, gt_answer: str, answer_type: str,
                    precision: int = None) -> bool:
    """Compare predicted answer to ground truth."""
    pred_norm = normalize_answer(predicted, answer_type)
    gt_norm = normalize_answer(gt_answer, answer_type)

    if not pred_norm or pred_norm == "failed":
        return False

    if pred_norm == gt_norm:
        return True

    if answer_type in ("integer", "float"):
        try:
            pred_val = float(pred_norm)
            gt_val = float(gt_norm)

            if answer_type == "integer":
                return int(round(pred_val)) == int(round(gt_val))

            if precision is not None and precision >= 0:
                return round(pred_val, precision) == round(gt_val, precision)

            return abs(pred_val - gt_val) < 1e-6

        except (ValueError, TypeError):
            pass

    return False


# ============================================================================
# CHOICES PARSING FROM QUESTION TEXT
# ============================================================================
def _parse_choices_from_question(r: dict) -> tuple[list, str]:
    """
    Parse choices and gt_answer_letter from the full_question text.

    MathVista inference results may not carry separate 'choices' or
    'gt_answer_letter' fields.  The choices ARE embedded in the prompt, e.g.:
        Choices:
        (A) 135°
        (B) 140°
        (C) 145°
        (D) 150°

    Returns:
        choices:  list of choice texts, e.g. ["135°", "140°", "145°", "150°"]
        gt_letter: the option letter whose text matches gt_answer, e.g. "C"
    """
    question_text = r.get("full_question", "") or r.get("original_question", "")
    if not question_text:
        return [], ""

    # Parse "(A) xxx\n(B) yyy\n..." pattern
    matches = re.findall(r'\(([A-Z])\)\s*(.+?)(?=\n\s*\(|$)', question_text, re.DOTALL)
    if not matches:
        return [], ""

    choices_dict = {letter: val.strip() for letter, val in matches}
    choices_list = [choices_dict[chr(ord("A") + i)]
                    for i in range(len(choices_dict))
                    if chr(ord("A") + i) in choices_dict]

    # Map gt_answer text → letter
    gt_answer = str(r.get("gt_answer", "")).strip()
    gt_letter = ""
    if gt_answer:
        gt_lower = gt_answer.lower()
        for letter, val in choices_dict.items():
            if val.strip().lower() == gt_lower:
                gt_letter = letter
                break

    return choices_list, gt_letter


# ============================================================================
# CORE EVALUATION
# ============================================================================
def _eval_one_sample(r: dict, response_field: str) -> dict:
    """Evaluate a single MathVista sample. Returns eval fields."""
    resp = r.get(response_field, r.get("response", ""))

    gt_answer = r.get("gt_answer", "")
    question_type = r.get("question_type", "")
    answer_type = r.get("answer_type", "text")
    choices = r.get("choices", [])
    gt_letter = r.get("gt_answer_letter", "")
    precision_val = r.get("precision", None)

    # Handle metadata-nested fields
    meta = r.get("metadata", {}) if isinstance(r.get("metadata"), dict) else {}
    if not question_type:
        question_type = meta.get("question_type", "")
    if not answer_type or answer_type == "text":
        answer_type = meta.get("answer_type", answer_type)
    if not choices:
        choices = meta.get("choices", [])
    if not gt_answer:
        gt_answer = meta.get("gt_answer", "")
    if not gt_letter:
        gt_letter = meta.get("gt_answer_letter", "")
    if precision_val is None:
        precision_val = meta.get("precision", None)

    # ── KEY FIX: parse choices from full_question when not in fields ──
    if question_type == "multi_choice" and (not choices or not gt_letter):
        parsed_choices, parsed_letter = _parse_choices_from_question(r)
        if not choices and parsed_choices:
            choices = parsed_choices
        if not gt_letter and parsed_letter:
            gt_letter = parsed_letter

    is_correct = False
    predicted = ""
    extraction_success = True

    if question_type == "multi_choice":
        predicted = extract_answer_multichoice(resp, choices)
        extraction_success = predicted != "FAILED"

        if extraction_success:
            if gt_letter:
                is_correct = (predicted == gt_letter.upper())
            else:
                pred_idx = ord(predicted) - ord("A")
                if choices and 0 <= pred_idx < len(choices):
                    pred_text = str(choices[pred_idx]).strip()
                    is_correct = (
                        normalize_answer(pred_text, answer_type) ==
                        normalize_answer(gt_answer, answer_type)
                    )
    else:
        # free_form
        predicted = extract_answer_freeform(resp, answer_type)
        extraction_success = predicted != "FAILED"
        if extraction_success:
            is_correct = compare_answers(predicted, gt_answer, answer_type,
                                         precision_val)

    return {
        "predicted_answer": predicted,
        "gt_answer_clean": gt_answer,
        "is_correct": is_correct,
        "extraction_success": extraction_success,
    }


def _get_meta(r: dict, field: str) -> str:
    """Get a metadata field from result, checking both top-level and nested metadata."""
    val = r.get(field, "")
    if val:
        return str(val)
    meta = r.get("metadata", {}) if isinstance(r.get("metadata"), dict) else {}
    return str(meta.get(field, "")) or "unknown"


def _eval_mathvista(results: list, response_field: str = "response") -> dict:
    """Evaluate MathVista results following the official protocol."""
    total = len(results)
    correct = 0
    extracted = 0

    # Group stats
    qtype_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    atype_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    task_stats  = defaultdict(lambda: {"total": 0, "correct": 0})
    cat_stats   = defaultdict(lambda: {"total": 0, "correct": 0})
    grade_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    ctx_stats   = defaultdict(lambda: {"total": 0, "correct": 0})

    for r in results:
        ev = _eval_one_sample(r, response_field)
        if ev["is_correct"]:
            correct += 1
        if ev["extraction_success"]:
            extracted += 1

        qtype = _get_meta(r, "question_type")
        atype = _get_meta(r, "answer_type")
        task  = _get_meta(r, "task")
        cat   = _get_meta(r, "category")
        grade = _get_meta(r, "grade")
        ctx   = _get_meta(r, "context")

        for stats, key in [
            (qtype_stats, qtype), (atype_stats, atype), (task_stats, task),
            (cat_stats, cat), (grade_stats, grade), (ctx_stats, ctx),
        ]:
            stats[key]["total"] += 1
            if ev["is_correct"]:
                stats[key]["correct"] += 1

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
        "by_question_type": _finalize(qtype_stats),
        "by_answer_type": _finalize(atype_stats),
        "by_task": _finalize(task_stats),
        "by_category": _finalize(cat_stats),
        "by_grade": _finalize(grade_stats),
        "by_context": _finalize(ctx_stats),
    }


# ============================================================================
# DISPLAY
# ============================================================================
def print_results(eval_result: dict, model_name: str = ""):
    print(f"\n{'='*80}")
    print(f"MATHVISTA EVALUATION RESULTS{' — ' + model_name if model_name else ''}")
    print(f"{'='*80}")

    print(f"\n  Total samples:      {eval_result['total_samples']}")
    print(f"  Extraction rate:    {eval_result['extraction_rate']}%")
    print(f"  Overall Accuracy:   {eval_result['overall_accuracy']}%")

    for group_name, stats in [
        ("Question Type", eval_result.get("by_question_type", {})),
        ("Answer Type",   eval_result.get("by_answer_type", {})),
        ("Task",          eval_result.get("by_task", {})),
        ("Category",      eval_result.get("by_category", {})),
        ("Grade",         eval_result.get("by_grade", {})),
    ]:
        if not stats:
            continue
        print(f"\n  By {group_name}:")
        print(f"    {'Name':<35s} {'Total':>6s} {'Correct':>8s} {'Acc%':>7s}")
        print(f"    {'-'*35} {'-'*6} {'-'*8} {'-'*7}")
        for name, st in stats.items():
            print(f"    {name:<35s} {st['total']:>6d} {st['correct']:>8d} "
                  f"{st['accuracy']:>6.1f}%")

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
                 and "mathvista_eval" not in f.name.lower()]
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
            "evaluator": "eval_mathvista.py (official MathVista protocol)",
            "evaluation_date": datetime.now().isoformat(),
            "evaluation_method": "mathvista_official",
            "reference": "Lu et al., MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts, ICLR 2024",
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
        "by_question_type": eval_result["by_question_type"],
        "by_answer_type": eval_result["by_answer_type"],
        "by_task": eval_result["by_task"],
        "by_category": eval_result["by_category"],
        "by_grade": eval_result["by_grade"],
        "by_context": eval_result["by_context"],
    }


def evaluate_one_file(result_file: str, output_dir: str, response_field: str = "response") -> dict:
    """Evaluate one MathVista result file and save outputs. Returns a summary row."""
    with open(result_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    if isinstance(results, dict) and "results" in results:
        results = results["results"]

    print(f"\nLoading: {result_file}")
    print(f"  Loaded {len(results)} samples")

    model_name = results[0].get("model", "") if results else ""
    eval_result = _eval_mathvista(results, response_field=response_field)
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
    legacy_path = os.path.join(output_dir, f"{stem}_mathvista_eval.json")
    with open(legacy_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Evaluation saved: {legacy_path}")
    if legacy_path != evaluated_path:
        print(f"✅ Detailed eval saved: {evaluated_path}")

    first_result = results[0] if results else {}
    finding_name = Path(result_file).parent.name
    condition, emotion_category, subject = _extract_condition_fields(ds_name, first_result)

    # Build comparison row
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
    # Add per question_type accuracies
    for qtype, info in eval_result.get("by_question_type", {}).items():
        row[f"acc_{qtype}"] = info["accuracy"]
    # Add per task accuracies
    for task, info in eval_result.get("by_task", {}).items():
        safe_task = task.replace(" ", "_")
        row[f"acc_task_{safe_task}"] = info["accuracy"]

    return row


def run_finding_evaluation(finding_name: str, output_dir: str, model_dir: str | None = None,
                           skip_neutral: bool = False, method: bool = False):
    """Evaluate all MathVista result files for a finding (same workflow as eval_mme)."""
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
    print(f"MATHVISTA EVALUATION — {finding_name.upper()}")
    print(f"{'='*80}")
    print("Method:  Official MathVista metrics (Accuracy)")
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
    print(f"MATHVISTA COMPARISON — {finding_name.upper()}")
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
        "dataset": "mathvista",
        "evaluation_method": "mathvista_official",
        "reference": "Lu et al., MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts, ICLR 2024",
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
        description="MathVista Benchmark Evaluation — Official Metrics (Accuracy)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
MathVista Metrics:
  Accuracy per question type (multi_choice, free_form)
  Breakdown by: task, category, grade, context, answer_type
  Two question types: multi_choice (exact letter) + free_form (numeric/text)

Usage (recommended, same style as eval_mme):
  python scripts/eval/eval_mathvista.py --model_dir qwen2_vl_7b --finding mathvista_baseline

Single-file mode (still supported):
  python scripts/eval/eval_mathvista.py --result_file /workspace/results/method1/llava_1_5_7b__gemma_3_12b_it/mathvista/fixed/negative_low/start/multi2/method1_results_20260322_094812.json --method

Method mode (uses final_response field):
  python scripts/eval/eval_mathvista.py --model_dir internvl2_5_8b --finding mathvista_baseline --method
        """,
    )

    parser.add_argument("--result_file", type=str, default=None,
                        help="Path to one inference results JSON file (optional)")
    parser.add_argument("--finding", type=str, default="mathvista_baseline",
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
        print("MATHVISTA EVALUATION (Single-file mode)")
        print(f"{'='*80}")
        print(f"Input:    {args.result_file}")
        print(f"Output:   {output_dir}")
        print(f"Method:   Official MathVista metrics (Accuracy)")
        print(f"Field:    {response_field}")

        evaluate_one_file(args.result_file, output_dir=output_dir, response_field=response_field)

        print(f"\n{'='*80}")
        print("✅ MATHVISTA EVALUATION COMPLETE")
        print(f"{'='*80}")
        return

    # Finding-level mode (preferred)
    eval_subdir = args.model_dir if args.model_dir else "all_models"
    output_dir = args.output_dir or os.path.join(EVAL_BASE_DIR, eval_subdir, args.finding)

    print(f"\n{'='*80}")
    print("MATHVISTA EVALUATION (ICLR 2024)")
    print(f"{'='*80}")
    print(f"Finding:  {args.finding}")
    print(f"Input:    {INFER_BASE_DIR}/{args.model_dir or '*'}/{args.finding}/")
    print(f"Output:   {output_dir}")
    print("Method:   Official MathVista metrics (Accuracy)")
    print(f"Field:    {'final_response' if args.method else 'response'}")

    run_finding_evaluation(
        finding_name=args.finding,
        output_dir=output_dir,
        model_dir=args.model_dir,
        skip_neutral=args.skip_neutral,
        method=args.method,
    )

    print(f"\n{'='*80}")
    print("✅ MATHVISTA EVALUATION COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
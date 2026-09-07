"""
MMMU Benchmark Evaluation Script — Official Metrics (Micro-averaged Accuracy)

Faithfully reproduces the official MMMU evaluation logic from:
    https://github.com/MMMU-Benchmark/MMMU/blob/main/mmmu/utils/eval_utils.py

Key behaviors matching the official code:
  - parse_multi_choice_response: strip punctuation → pad with spaces →
    look for (A)/(B) patterns → space-delimited letters → content matching →
    if multiple candidates pick LAST occurrence → if none, RANDOM CHOICE
  - parse_open_response: extract numbers with regex → normalize → compare
  - For open questions with no valid extraction: marked INCORRECT (no random)
  - Metric: micro-averaged accuracy with breakdowns by subject (30) and
    discipline (6)

Follows the same workflow/style as eval_mathvista.py:
  - Supports finding-level evaluation via --model_dir + --finding
  - Saves outputs under results/eval/{model_dir}/{finding}/
  - Supports single-file evaluation via --result_file
  - Writes per-file evaluated JSON + comparison CSV/JSON + summary JSON
  - Supports --method flag for final_response (method) vs response (baseline)

Reference:
    MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning
    Benchmark for Expert AGI (Yue et al., CVPR 2024)
    https://mmmu-benchmark.github.io/
"""

# Portions of the answer-parsing logic are adapted from the official MMMU evaluator:\n# https://github.com/MMMU-Benchmark/MMMU (Apache License 2.0).\n# See THIRD_PARTY_NOTICES.md.\n\nimport json
import os
import re
import sys
import random
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

random.seed(42)

try:
    import numpy as np
except ImportError:
    np = None

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

# MMMU 6 disciplines mapped from 30 subjects
SUBJECT_TO_DISCIPLINE = {
    "Accounting":                   "Business",
    "Economics":                    "Business",
    "Finance":                      "Business",
    "Manage":                       "Business",
    "Marketing":                    "Business",
    "Art":                          "Art & Design",
    "Art_Theory":                   "Art & Design",
    "Design":                       "Art & Design",
    "Music":                        "Art & Design",
    "Biology":                      "Science",
    "Chemistry":                    "Science",
    "Geography":                    "Science",
    "Math":                         "Science",
    "Physics":                      "Science",
    "Basic_Medical_Science":        "Health & Medicine",
    "Clinical_Medicine":            "Health & Medicine",
    "Diagnostics_and_Laboratory_Medicine": "Health & Medicine",
    "Pharmacy":                     "Health & Medicine",
    "Public_Health":                "Health & Medicine",
    "History":                      "Humanities & Social Science",
    "Literature":                   "Humanities & Social Science",
    "Psychology":                   "Humanities & Social Science",
    "Sociology":                    "Humanities & Social Science",
    "Architecture_and_Engineering": "Tech & Engineering",
    "Computer_Science":             "Tech & Engineering",
    "Electronics":                  "Tech & Engineering",
    "Energy_and_Power":             "Tech & Engineering",
    "Materials":                    "Tech & Engineering",
    "Mechanical_Engineering":       "Tech & Engineering",
}


# ============================================================================
# OFFICIAL MMMU MULTI-CHOICE PARSER
# Faithfully reproduced from:
#   https://github.com/MMMU-Benchmark/MMMU/blob/main/mmmu/utils/eval_utils.py
# ============================================================================
def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.

    This is a FAITHFUL reproduction of the official MMMU evaluation code.
    Key behavior:
      1. Strip punctuation, pad response with spaces
      2. Look for (A), (B), etc. patterns
      3. Look for space-delimited A, B, etc.
      4. If >5 tokens, try content matching against answer text
      5. If no candidates found: RANDOM CHOICE (official fallback)
      6. If multiple candidates: pick the LAST occurrence (rightmost)
    """
    for char in [',', '.', '!', '?', ';', ':', "'"]:
        response = response.strip(char)
    response = " " + response + " "  # add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    candidates = []

    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f'({choice})' in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A B C D
            if f' {choice} ' in response:
                candidates.append(choice)

    # if all above doesn't get candidates, check if the content is larger
    # than 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False  # it's content ans.

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    index = response.rfind(f'({can})')
                    start_indexes.append(index)
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        # get the last one
        if np is not None:
            pred_index = candidates[np.argmax(start_indexes)]
        else:
            pred_index = candidates[start_indexes.index(max(start_indexes))]
    else:  # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index


# ============================================================================
# OFFICIAL MMMU OPEN-ENDED PARSER
# Faithfully reproduced from the official MMMU eval_utils.py
# ============================================================================
def check_is_number(string):
    """Check if the given string is a number."""
    try:
        float(string.replace(',', ''))
        return True
    except ValueError:
        return False


def normalize_str(string):
    """
    Normalize the str to lower case and make them float numbers if possible.
    Official MMMU logic.
    """
    string = string.strip()

    is_number = check_is_number(string)

    if is_number:
        string = string.replace(',', '')
        string = float(string)
        # leave 2 decimal
        string = round(string, 2)
        return [string]
    else:  # it's likely to be a string
        # lower it
        string = string.lower()
        if len(string) == 1:
            return [" " + string, string + " "]  # avoid trivial matches
        return [string]


def extract_numbers(string):
    """
    Extract all forms of numbers from a string with regex.
    Official MMMU logic.
    """
    # Pattern for numbers with commas
    pattern_commas = r'-?\b\d{1,3}(?:,\d{3})+\b'
    # Pattern for scientific notation
    pattern_scientific = r'-?\d+(?:\.\d+)?[eE][+-]?\d+'
    # Pattern for simple numbers without commas
    pattern_simple = r'-?(?:\d+\.\d+|\.\d+|\d+\b)(?![eE][+-]?\d+)(?![,\d])'

    numbers_with_commas = re.findall(pattern_commas, string)
    numbers_scientific = re.findall(pattern_scientific, string)
    numbers_simple = re.findall(pattern_simple, string)

    all_numbers = numbers_with_commas + numbers_scientific + numbers_simple
    return all_numbers


def parse_open_response(response):
    """
    Parse the prediction from the generated response for open-ended questions.
    Return a list of predicted strings or numbers.

    FAITHFUL reproduction of the official MMMU eval_utils.py:
    https://github.com/MMMU-Benchmark/MMMU/blob/main/mmmu/utils/eval_utils.py
    """

    def get_key_subresponses(response):
        key_responses = []
        response = response.strip().strip(".").lower()
        sub_responses = re.split(r'\.\s(?=[A-Z])|\n', response)
        indicators_of_keys = ['could be ', 'so ', 'is ',
                              'thus ', 'therefore ', 'final ', 'answer ', 'result ']
        key_responses = []
        for index, resp in enumerate(sub_responses):
            # if last one, accept it's an equation
            if index == len(sub_responses) - 1:
                indicators_of_keys.extend(['='])
            shortest_key_response = None
            for indicator in indicators_of_keys:
                if indicator in resp:
                    if not shortest_key_response:
                        shortest_key_response = resp.split(indicator)[-1].strip()
                    else:
                        if len(resp.split(indicator)[-1].strip()) < len(shortest_key_response):
                            shortest_key_response = resp.split(indicator)[-1].strip()
            if shortest_key_response:
                if shortest_key_response.strip() not in [":", ",", ".", "!", "?", ";", ":", "'"]:
                    key_responses.append(shortest_key_response)
        if len(key_responses) == 0:
            return [response]
        return key_responses

    key_responses = get_key_subresponses(response)

    pred_list = key_responses.copy()
    for resp in key_responses:
        pred_list.extend(extract_numbers(resp))

    tmp_pred_list = []
    for i in range(len(pred_list)):
        tmp_pred_list.extend(normalize_str(pred_list[i]))

    pred_list = tmp_pred_list
    pred_list = list(set(pred_list))

    return pred_list


# ============================================================================
# OFFICIAL MMMU EVALUATION FUNCTION
# ============================================================================
def eval_multi_choice(gold_i, pred_i):
    """
    Evaluate a multiple choice instance.
    Official MMMU logic: exact match only.
    """
    correct = False
    if isinstance(gold_i, list):
        for answer in gold_i:
            if answer == pred_i:
                correct = True
                break
    else:
        if gold_i == pred_i:
            correct = True
    return correct


def eval_open(gold_i, pred_i):
    """
    Evaluate an open question instance.
    Official MMMU logic: pred_i is a LIST of candidate predictions.
    """
    correct = False
    if isinstance(gold_i, list):
        norm_answers = []
        for answer in gold_i:
            norm_answers.extend(normalize_str(answer))
    else:
        norm_answers = normalize_str(gold_i)

    for pred in pred_i:
        if isinstance(pred, str):
            for norm_ans in norm_answers:
                if isinstance(norm_ans, str) and norm_ans in pred:
                    if not correct:
                        correct = True
                    break
        else:  # it's a float number
            if pred in norm_answers:
                if not correct:
                    correct = True
                break
    return correct


def evaluate_mmmu_sample(prediction, gt_answer, question_type, options=None):
    """
    Evaluate a single MMMU sample following the official protocol.

    For multi_choice: use parse_multi_choice_response then compare letters.
    For open: use parse_open_response (returns a LIST) then eval_open.

    Returns: (is_correct: bool, parsed_prediction: str)
    """
    if question_type in ("multi_choice", "multiple_choice", "multiple-choice"):
        # Build all_choices and index2ans from options
        if options and isinstance(options, list):
            all_choices = [chr(ord("A") + i) for i in range(len(options))]
            index2ans = {chr(ord("A") + i): str(o) for i, o in enumerate(options)}
        elif options and isinstance(options, dict):
            all_choices = sorted(options.keys())
            index2ans = {k: str(v) for k, v in options.items()}
        else:
            all_choices = ["A", "B", "C", "D"]
            index2ans = {}

        parsed_pred = parse_multi_choice_response(prediction, all_choices, index2ans)
        is_correct = eval_multi_choice(gt_answer.upper(), parsed_pred)
        return is_correct, parsed_pred

    else:
        # Open-ended: parse_open_response returns a LIST of candidates
        parsed_pred_list = parse_open_response(prediction)
        is_correct = eval_open(gt_answer, parsed_pred_list)
        # For display, join the list
        display_pred = str(parsed_pred_list[0]) if parsed_pred_list else prediction
        return is_correct, display_pred


def _extract_options_from_question(full_question: str) -> list:
    """Extract options list from the formatted question string.
    Parses '(A) text\\n(B) text\\n...' patterns."""
    options = []
    pattern = r'\(([A-Z])\)\s*(.+?)(?=\n\([A-Z]\)|\nAnswer|\Z)'
    matches = re.findall(pattern, full_question, re.DOTALL)
    matches.sort(key=lambda x: x[0])
    for letter, text in matches:
        options.append(text.strip())
    return options


# ============================================================================
# CORE EVALUATION
# ============================================================================
def _eval_one_sample(r: dict, response_field: str) -> dict:
    """Evaluate a single MMMU sample."""
    resp = r.get(response_field, r.get("response", ""))

    gt_answer = r.get("gt_answer", "")
    question_type = r.get("question_type", "")
    choices = r.get("choices", [])
    options_dict = r.get("options", {})

    # Handle metadata-nested fields
    meta = r.get("metadata", {}) if isinstance(r.get("metadata"), dict) else {}
    if not question_type:
        question_type = meta.get("question_type", "multi_choice")
    if not gt_answer:
        gt_answer = meta.get("gt_answer", "")
    if not choices:
        choices = meta.get("choices", [])
    if not options_dict:
        options_dict = meta.get("options", {})

    # Fallback: extract options from full_question if not available
    if not choices and not options_dict:
        full_q = r.get("full_question", "") or meta.get("formatted_question", "")
        choices = _extract_options_from_question(full_q)

    # Resolve options: prefer dict, fall back to list
    opts_for_eval = options_dict if options_dict else choices

    is_correct, parsed_pred = evaluate_mmmu_sample(
        prediction=resp or "",
        gt_answer=gt_answer,
        question_type=question_type,
        options=opts_for_eval,
    )

    return {
        "predicted_answer": parsed_pred,
        "gt_answer": gt_answer,
        "is_correct": is_correct,
        "question_type": question_type,
    }


def _get_meta(r: dict, field: str) -> str:
    val = r.get(field, "")
    if val:
        return str(val)
    meta = r.get("metadata", {}) if isinstance(r.get("metadata"), dict) else {}
    return str(meta.get(field, "")) or "unknown"


def _eval_mmmu(results: list, response_field: str = "response") -> dict:
    """Evaluate MMMU results following the official protocol."""
    total = len(results)
    correct = 0

    # Group stats
    qtype_stats      = defaultdict(lambda: {"total": 0, "correct": 0})
    subject_stats    = defaultdict(lambda: {"total": 0, "correct": 0})
    discipline_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    difficulty_stats = defaultdict(lambda: {"total": 0, "correct": 0})

    for r in results:
        ev = _eval_one_sample(r, response_field)
        if ev["is_correct"]:
            correct += 1

        qtype  = ev["question_type"] or _get_meta(r, "question_type")
        subject = _get_meta(r, "subject")
        discipline = SUBJECT_TO_DISCIPLINE.get(subject, _get_meta(r, "discipline"))
        difficulty = _get_meta(r, "topic_difficulty")

        for stats, key in [
            (qtype_stats, qtype),
            (subject_stats, subject),
            (discipline_stats, discipline),
            (difficulty_stats, difficulty),
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
        "overall_accuracy": round(overall_acc * 100, 2),
        "by_question_type": _finalize(qtype_stats),
        "by_subject": _finalize(subject_stats),
        "by_discipline": _finalize(discipline_stats),
        "by_difficulty": _finalize(difficulty_stats),
    }


# ============================================================================
# DISPLAY
# ============================================================================
def print_results(eval_result: dict, model_name: str = ""):
    print(f"\n{'='*80}")
    print(f"MMMU EVALUATION RESULTS{' — ' + model_name if model_name else ''}")
    print(f"{'='*80}")

    print(f"\n  Total samples:      {eval_result['total_samples']}")
    print(f"  Overall Accuracy:   {eval_result['overall_accuracy']}%")

    for group_name, stats in [
        ("Question Type", eval_result.get("by_question_type", {})),
        ("Discipline",    eval_result.get("by_discipline", {})),
        ("Difficulty",    eval_result.get("by_difficulty", {})),
        ("Subject",       eval_result.get("by_subject", {})),
    ]:
        if not stats:
            continue
        print(f"\n  By {group_name}:")
        print(f"    {'Name':<45s} {'Total':>6s} {'Correct':>8s} {'Acc%':>7s}")
        print(f"    {'-'*45} {'-'*6} {'-'*8} {'-'*7}")
        for name, st in stats.items():
            print(f"    {name:<45s} {st['total']:>6d} {st['correct']:>8d} "
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
                 and "mmmu_eval" not in f.name.lower()]
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
            "evaluator": "eval_mmmu.py (official MMMU protocol — micro-averaged accuracy)",
            "evaluation_date": datetime.now().isoformat(),
            "evaluation_method": "mmmu_official",
            "reference": "Yue et al., MMMU: A Massive Multi-discipline Multimodal "
                         "Understanding and Reasoning Benchmark (CVPR 2024)",
            "result_file": os.path.basename(result_file),
            "response_field": response_field,
            "model": model_name,
            "total_samples": len(results),
            "note": "Multi-choice uses random fallback when extraction fails "
                    "(official MMMU behavior). Open questions: no fallback.",
        },
        "summary": {
            "total_samples": eval_result["total_samples"],
            "correct": eval_result["correct"],
            "overall_accuracy": eval_result["overall_accuracy"],
        },
        "scores": {
            "overall_accuracy": eval_result["overall_accuracy"],
        },
        "by_question_type": eval_result["by_question_type"],
        "by_subject": eval_result["by_subject"],
        "by_discipline": eval_result["by_discipline"],
        "by_difficulty": eval_result["by_difficulty"],
    }


def evaluate_one_file(result_file: str, output_dir: str, response_field: str = "response") -> dict:
    with open(result_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    if isinstance(results, dict) and "results" in results:
        results = results["results"]

    print(f"\nLoading: {result_file}")
    print(f"  Loaded {len(results)} samples")

    model_name = results[0].get("model", "") if results else ""
    eval_result = _eval_mmmu(results, response_field=response_field)
    print_results(eval_result, model_name)

    os.makedirs(output_dir, exist_ok=True)

    stem = Path(result_file).stem
    ds_name = parse_dataset_name(Path(result_file))

    payload = build_output_payload(results, eval_result, result_file, response_field, model_name)

    evaluated_path = os.path.join(output_dir, f"{ds_name}_evaluated.json")
    with open(evaluated_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    legacy_path = os.path.join(output_dir, f"{stem}_mmmu_eval.json")
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
    }
    # Add per-discipline accuracies
    for disc, info in eval_result.get("by_discipline", {}).items():
        safe_disc = disc.replace(" ", "_").replace("&", "and")
        row[f"acc_{safe_disc}"] = info["accuracy"]
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
    print(f"MMMU EVALUATION — {finding_name.upper()}")
    print(f"{'='*80}")
    print("Method:  Official MMMU metrics (Micro-averaged Accuracy)")
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
    print(f"MMMU COMPARISON — {finding_name.upper()}")
    print(f"{'='*80}")
    print(f"{'Condition':<30} {'Acc%':>8} {'N':>6}")
    print(f"{'-'*50}")
    for row in sorted(comparison_data, key=lambda x: x["overall_accuracy"], reverse=True):
        subj = f" [{row['subject']}]" if row.get("subject") else ""
        cond = f"{row['condition']}{subj}"
        print(f"{cond:<30} {row['overall_accuracy']:>7.2f} {row['total_samples']:>5}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "finding": finding_name,
        "dataset": "mmmu",
        "evaluation_method": "mmmu_official",
        "reference": "Yue et al., MMMU (CVPR 2024)",
        "evaluation_date": datetime.now().isoformat(),
        "response_field": response_field,
        "files_evaluated": len(comparison_data),
        "note": "MC extraction uses random fallback (official). "
                "Open questions: no fallback. random.seed(42).",
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
        description="MMMU Benchmark Evaluation — Official Metrics (Accuracy)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
MMMU Metrics (official):
  Micro-averaged accuracy across all samples.
  Breakdown by: question type (multi_choice/open), subject (30),
  discipline (6), difficulty.

  Multi-choice: parse_multi_choice_response with random fallback.
  Open: parse_open_response with number extraction, no fallback.

Usage (recommended):
  python scripts/eval/eval_mmmu.py --model_dir llava_1_5_7b --finding mmmu_baseline

Single-file mode:
  python scripts/eval/eval_mmmu.py --result_file /path/to/results.json --method
        """,
    )

    parser.add_argument("--result_file", type=str, default=None)
    parser.add_argument("--finding", type=str, default="mmmu_baseline")
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
        print("MMMU EVALUATION (Single-file mode)")
        print(f"{'='*80}")
        print(f"Input:  {args.result_file}")
        print(f"Field:  {response_field}")

        evaluate_one_file(args.result_file, output_dir=output_dir, response_field=response_field)

        print(f"\n✅ MMMU EVALUATION COMPLETE")
        return

    eval_subdir = args.model_dir if args.model_dir else "all_models"
    output_dir = args.output_dir or os.path.join(EVAL_BASE_DIR, eval_subdir, args.finding)

    print(f"\n{'='*80}")
    print("MMMU EVALUATION (Yue et al., CVPR 2024)")
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

    print(f"\n✅ MMMU EVALUATION COMPLETE")


if __name__ == "__main__":
    main()
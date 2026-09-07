"""
POPE Evaluation Script v2 — METHOD evaluation (uses final_response)

Key differences from v1 (baseline):
  - Uses 'final_response' field instead of 'response'
  - Supports --result_file for direct file input
  - Reports method-specific stats: regen vs non-regen metrics
  - File discovery matches method*_results_*.json pattern

Official metrics (Li et al., EMNLP 2023):
  1. Accuracy  = (TP + TN) / total
  2. Precision = TP / (TP + FP)
  3. Recall    = TP / (TP + FN)
  4. F1        = 2 * P * R / (P + R)
  5. Yes Ratio = proportion of "yes" answers

Where: "yes" = positive, "no" = negative.

Usage:
  python eval_pope_v2.py --result_file results/infer/llava_1_5_7b_a100/pope_method/method1_results_20260223_145726.json
  python eval_pope_v2.py --finding pope_method --model_dir llava_1_5_7b_a100
  python eval_pope_v2.py --result_file method_results.json --test_mode
"""

import json
import os
import sys
import re
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT))

from path_config import RESULTS_ROOT

# ============================================================================
# PATHS
# ============================================================================
INFER_BASE_DIR = str(RESULTS_ROOT / "infer")
EVAL_BASE_DIR = str(RESULTS_ROOT / "eval")


# ============================================================================
# ANSWER EXTRACTION (same as v1 — official POPE style)
# ============================================================================
def extract_yes_no(response: str) -> str:
    """
    Extract yes/no from model response.
    Returns: "yes", "no", or "unknown"
    """
    if not response or not isinstance(response, str):
        return "unknown"

    text = response.strip().lower()

    if text in ("yes", "yes.", "yes!"):
        return "yes"
    if text in ("no", "no.", "no!"):
        return "no"

    if text.startswith("yes"):
        if len(text) == 3 or not text[3].isalpha():
            return "yes"
    if text.startswith("no"):
        if len(text) == 2 or not text[2].isalpha():
            return "no"

    if re.search(r'\bthere\s+(?:is|are)\s+(?:no|not)\b', text):
        return "no"
    if re.search(r'\b(?:cannot|can\'t|don\'t|do not)\s+(?:see|find|identify|detect)\b', text):
        return "no"
    if re.search(r'\bthere\s+(?:is|are)\s+(?:a|an|the|one|two|three|several|many)\b', text):
        return "yes"
    if re.search(r'\byes\b', text):
        return "yes"
    if re.search(r'\bno\b', text):
        return "no"

    return "unknown"


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ('true', '1', 'yes')
    return bool(value)


# ============================================================================
# METRICS (official POPE)
# ============================================================================
def compute_pope_metrics(predictions, gt_labels):
    assert len(predictions) == len(gt_labels)

    TP = FP = TN = FN = 0
    yes_count = 0
    total = len(predictions)

    for pred, gt in zip(predictions, gt_labels):
        gt = gt.strip().lower()
        if pred == "yes":
            yes_count += 1
            if gt == "yes":
                TP += 1
            else:
                FP += 1
        else:
            if gt == "no":
                TN += 1
            else:
                FN += 1

    accuracy = (TP + TN) / total if total > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    yes_ratio = yes_count / total if total > 0 else 0

    return {
        "accuracy": round(accuracy * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1": round(f1 * 100, 2),
        "yes_ratio": round(yes_ratio * 100, 2),
        "total": total,
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "yes_count": yes_count,
        "no_count": total - yes_count,
        "unknown_count": sum(1 for p in predictions if p == "unknown"),
    }


# ============================================================================
# FILE DISCOVERY (v2 — supports method file patterns)
# ============================================================================
def discover_result_files(finding_name, model_dir=None):
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
        v6_files = sorted(search_path.glob("results_*.json"))
        method_files = sorted(search_path.glob("method*_results_*.json"))
        all_files = list(set(v6_files + method_files))
        all_files = [f for f in all_files if "summary" not in f.name.lower()
                     and "evaluated" not in f.name.lower()]
        found_files.extend(sorted(all_files))

    return found_files


# ============================================================================
# EVALUATION LOGIC (v2 — final_response + regen stats)
# ============================================================================
def evaluate_pope_results(results, response_field="final_response"):
    valid = [r for r in results
             if not (r.get(response_field, "") or r.get("response", "")).startswith("[Error")]

    if not valid:
        print("❌ No valid results to evaluate!")
        return None

    predictions = []
    gt_labels = []
    splits = []
    regen_flags = []
    details = []

    for r in valid:
        response = r.get(response_field, "") or r.get("response", "")
        pred = extract_yes_no(response)
        predictions.append(pred)

        gt = (r.get("gt_answer", "") or "unknown")
        gt_labels.append(gt.strip().lower())

        split = (r.get("pope_split", "") or "unknown")
        splits.append(split)

        was_regen = _parse_bool(r.get("was_regenerated", False))
        regen_flags.append(was_regen)

        orig_response = r.get("response", "")
        orig_pred = extract_yes_no(orig_response)

        details.append({
            "id": r.get("id", ""),
            "response": response[:200],
            "predicted": pred,
            "gt_answer": gt.strip().lower(),
            "is_correct": pred == gt.strip().lower(),
            "pope_split": split,
            "was_regenerated": was_regen,
            "original_predicted": orig_pred,
            "answer_changed": pred != orig_pred,
        })

    # Overall metrics
    overall = compute_pope_metrics(predictions, gt_labels)

    # Per-split metrics
    split_metrics = {}
    for sp in sorted(set(splits)):
        if sp == "unknown":
            continue
        sp_preds = [p for p, s in zip(predictions, splits) if s == sp]
        sp_gts = [g for g, s in zip(gt_labels, splits) if s == sp]
        if sp_preds:
            split_metrics[sp] = compute_pope_metrics(sp_preds, sp_gts)

    # Method stats: regen vs non-regen
    method_stats = {}
    if any(regen_flags):
        regen_preds = [p for p, f in zip(predictions, regen_flags) if f]
        regen_gts = [g for g, f in zip(gt_labels, regen_flags) if f]
        non_regen_preds = [p for p, f in zip(predictions, regen_flags) if not f]
        non_regen_gts = [g for g, f in zip(gt_labels, regen_flags) if not f]

        regen_metrics = compute_pope_metrics(regen_preds, regen_gts) if regen_preds else {}
        non_regen_metrics = compute_pope_metrics(non_regen_preds, non_regen_gts) if non_regen_preds else {}

        # Win/lose on changed answers
        changed = [d for d in details if d["was_regenerated"] and d["answer_changed"]]
        wins = sum(1 for d in changed if d["is_correct"])
        losses = sum(1 for d in changed if not d["is_correct"])

        method_stats = {
            "regenerated_count": len(regen_preds),
            "non_regenerated_count": len(non_regen_preds),
            "regenerated_metrics": regen_metrics,
            "non_regenerated_metrics": non_regen_metrics,
            "answer_changed_count": len(changed),
            "changed_wins": wins,
            "changed_losses": losses,
        }

    return {
        "metadata": {
            "evaluation_date": datetime.now().isoformat(),
            "evaluation_method": "pope_official_v2",
            "reference": "Li et al., EMNLP 2023",
            "response_field": response_field,
            "mode": "method",
        },
        "overall": overall,
        "per_split": split_metrics,
        "method_stats": method_stats,
        "detailed_results": details,
    }


# ============================================================================
# DIRECT FILE EVALUATION
# ============================================================================
def evaluate_result_file(result_file, output_dir, response_field="final_response"):
    result_path = Path(result_file)
    ds_name = result_path.stem

    print(f"\n{'='*80}")
    print(f"POPE METHOD EVALUATION — {ds_name}")
    print(f"{'='*80}")
    print(f"Response field: {response_field}")
    print(f"File:           {result_path.name}")

    with open(result_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    print(f"   Samples: {len(results)}")

    regen_count = sum(1 for r in results if _parse_bool(r.get("was_regenerated", False)))
    print(f"   Regenerated: {regen_count}/{len(results)} ({regen_count/len(results)*100:.1f}%)")

    os.makedirs(output_dir, exist_ok=True)

    evaluation = evaluate_pope_results(results, response_field=response_field)
    if evaluation is None:
        return None

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_path = os.path.join(output_dir, f"{ds_name}_evaluated_{ts}.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(evaluation, f, indent=2, ensure_ascii=False)

    # Print overall
    o = evaluation["overall"]
    print(f"\n   ✅ OVERALL:")
    print(f"      Accuracy:  {o['accuracy']:.2f}%")
    print(f"      Precision: {o['precision']:.2f}%")
    print(f"      Recall:    {o['recall']:.2f}%")
    print(f"      F1:        {o['f1']:.2f}%")
    print(f"      Yes Ratio: {o['yes_ratio']:.2f}%")
    print(f"      (TP={o['TP']} FP={o['FP']} TN={o['TN']} FN={o['FN']})")
    if o["unknown_count"] > 0:
        print(f"      ⚠️  Unknown: {o['unknown_count']} (treated as 'no')")

    # Per-split
    split_metrics = evaluation.get("per_split", {})
    if split_metrics:
        print(f"\n   Per-split:")
        print(f"   {'Split':<15} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Yes%':>7} {'N':>6}")
        print(f"   {'-'*60}")
        for sp in ["random", "popular", "adversarial"]:
            if sp in split_metrics:
                m = split_metrics[sp]
                print(f"   {sp:<15} {m['accuracy']:>6.2f}% {m['precision']:>6.2f}% "
                      f"{m['recall']:>6.2f}% {m['f1']:>6.2f}% {m['yes_ratio']:>6.2f}% {m['total']:>5}")

    # Method stats
    ms = evaluation.get("method_stats", {})
    if ms:
        rm = ms.get("regenerated_metrics", {})
        nm = ms.get("non_regenerated_metrics", {})
        print(f"\n   📊 Method breakdown:")
        print(f"      Non-regenerated: Acc={nm.get('accuracy',0):.2f}%  F1={nm.get('f1',0):.2f}%  (n={ms['non_regenerated_count']})")
        print(f"      Regenerated:     Acc={rm.get('accuracy',0):.2f}%  F1={rm.get('f1',0):.2f}%  (n={ms['regenerated_count']})")
        if ms.get("answer_changed_count", 0) > 0:
            print(f"      Answer changed:  {ms['answer_changed_count']}/{ms['regenerated_count']}")
            print(f"      Changed → Win: {ms['changed_wins']}  |  Changed → Lose: {ms['changed_losses']}")

    print(f"\n   📁 Saved: {eval_path}")
    return evaluation


# ============================================================================
# FINDING-LEVEL EVALUATION
# ============================================================================
def run_finding_evaluation(finding_name, output_dir, model_dir=None,
                           response_field="final_response"):
    result_files = discover_result_files(finding_name, model_dir)
    if not result_files:
        print(f"❌ No result files found for {finding_name}")
        return None

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"POPE METHOD EVALUATION — {finding_name.upper()}")
    print(f"{'='*80}")
    print(f"Response field: {response_field}")
    print(f"Files:          {len(result_files)}")
    for f in result_files:
        print(f"  - {f.name}")

    comparison_data = []

    for idx, rf in enumerate(result_files, 1):
        ds_name = rf.stem
        print(f"\n[{idx}/{len(result_files)}] Evaluating: {ds_name}")

        with open(rf, "r", encoding="utf-8") as f:
            results = json.load(f)

        evaluation = evaluate_pope_results(results, response_field=response_field)
        if evaluation is None:
            continue

        eval_path = os.path.join(output_dir, f"{ds_name}_evaluated.json")
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(evaluation, f, indent=2, ensure_ascii=False)

        o = evaluation["overall"]
        ms = evaluation.get("method_stats", {})
        print(f"   ✅ Acc={o['accuracy']:.2f}%  F1={o['f1']:.2f}%  Yes%={o['yes_ratio']:.2f}%")
        if ms:
            rm = ms.get("regenerated_metrics", {})
            nm = ms.get("non_regenerated_metrics", {})
            print(f"      Non-regen: Acc={nm.get('accuracy',0):.2f}%  |  Regen: Acc={rm.get('accuracy',0):.2f}%")

        row = {
            "dataset": ds_name, "finding": finding_name,
            "accuracy": o["accuracy"], "precision": o["precision"],
            "recall": o["recall"], "f1": o["f1"],
            "yes_ratio": o["yes_ratio"], "total": o["total"],
        }
        if ms:
            row["regen_count"] = ms["regenerated_count"]
            row["regen_accuracy"] = ms.get("regenerated_metrics", {}).get("accuracy", 0)
            row["non_regen_accuracy"] = ms.get("non_regenerated_metrics", {}).get("accuracy", 0)

        for sp, m in evaluation.get("per_split", {}).items():
            row[f"acc_{sp}"] = m["accuracy"]
            row[f"f1_{sp}"] = m["f1"]

        comparison_data.append(row)

    if comparison_data:
        with open(os.path.join(output_dir, f"{finding_name}_comparison.json"), "w") as f:
            json.dump(comparison_data, f, indent=2)
        if HAS_PANDAS:
            pd.DataFrame(comparison_data).to_csv(
                os.path.join(output_dir, f"{finding_name}_comparison.csv"), index=False)

    print(f"\n✅ {finding_name} evaluation complete → {output_dir}")
    return comparison_data


# ============================================================================
# TEST MODE
# ============================================================================
NUM_TEST_SAMPLES = 10


def run_test_mode(response_field="final_response",
                  finding_name=None, model_dir=None, result_file=None):
    print(f"\n{'='*80}")
    print("TEST MODE — POPE v2 Method Evaluation")
    print(f"{'='*80}")

    if result_file:
        test_file = Path(result_file)
    else:
        result_files = discover_result_files(finding_name, model_dir)
        if not result_files:
            print("❌ No result files found.")
            return
        test_file = result_files[0]

    print(f"  File: {test_file.name}")

    with open(test_file, "r", encoding="utf-8") as f:
        results = json.load(f)
    print(f"  Total: {len(results)}")

    regen_count = sum(1 for r in results if _parse_bool(r.get("was_regenerated", False)))
    print(f"  Regenerated: {regen_count}/{len(results)}")

    splits = set(r.get("pope_split", "unknown") for r in results)
    print(f"  Splits: {splits}")

    print(f"\n{'='*80}")
    print(f"EVALUATING {NUM_TEST_SAMPLES} SAMPLES")
    print(f"{'='*80}")

    for i, r in enumerate(results[:NUM_TEST_SAMPLES]):
        response = r.get(response_field, "") or r.get("response", "")
        pred = extract_yes_no(response)
        gt = r.get("gt_answer", "?")
        was_regen = _parse_bool(r.get("was_regenerated", False))
        orig = r.get("response", "")
        correct = pred == gt.strip().lower()

        regen_tag = " 🔄" if was_regen else ""
        correct_tag = "✅" if correct else "❌"

        print(f"\n  [{i+1}]{regen_tag} {r.get('id','?')}")
        print(f"  Response: {response[:80]}")
        print(f"  Pred={pred}  GT={gt}  {correct_tag}")
        if was_regen:
            print(f"  Original: {orig[:80]}")

    # Quick full eval
    print(f"\n{'='*80}")
    evaluation = evaluate_pope_results(results, response_field=response_field)
    if evaluation:
        o = evaluation["overall"]
        ms = evaluation.get("method_stats", {})
        print(f"  Accuracy:  {o['accuracy']:.2f}%")
        print(f"  Precision: {o['precision']:.2f}%")
        print(f"  Recall:    {o['recall']:.2f}%")
        print(f"  F1:        {o['f1']:.2f}%")
        print(f"  Yes Ratio: {o['yes_ratio']:.2f}%")

        if ms:
            rm = ms.get("regenerated_metrics", {})
            nm = ms.get("non_regenerated_metrics", {})
            print(f"\n  Non-regen: Acc={nm.get('accuracy',0):.2f}%  F1={nm.get('f1',0):.2f}%")
            print(f"  Regen:     Acc={rm.get('accuracy',0):.2f}%  F1={rm.get('f1',0):.2f}%")
            if ms.get("answer_changed_count"):
                print(f"  Changed: {ms['answer_changed_count']} → Win={ms['changed_wins']} Lose={ms['changed_losses']}")

        sp = evaluation.get("per_split", {})
        if sp:
            print(f"\n  Per-split:")
            for s in ["random", "popular", "adversarial"]:
                if s in sp:
                    print(f"    {s}: Acc={sp[s]['accuracy']:.2f}% F1={sp[s]['f1']:.2f}% (n={sp[s]['total']})")

    print(f"\n✅ Test complete. Run without --test_mode for full evaluation.")


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="POPE v2 — Method evaluation (uses final_response)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
POPE Method Evaluation v2
Uses 'final_response' field from method pipeline.
Reports regen vs non-regen metrics + win/lose analysis.
Pure string matching — NO LLM judge needed.

Examples:
  python eval_pope_v2.py --result_file method1_results.json
  python eval_pope_v2.py --finding pope_method --model_dir llava_1_5_7b_a100
  python eval_pope_v2.py --result_file method_results.json --test_mode
        """,
    )

    parser.add_argument("--result_file", type=str, default=None,
                        help="Direct path to method result JSON file")
    parser.add_argument("--finding", type=str, default=None,
                        help="Finding folder name (e.g. pope_method)")
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--test_mode", action="store_true")

    args = parser.parse_args()

    if not args.result_file and not args.finding:
        parser.error("Specify --result_file or --finding")

    print(f"\n{'='*80}")
    print("POPE EVALUATION v2 (METHOD — final_response)")
    print(f"{'='*80}")

    if args.result_file:
        if not os.path.exists(args.result_file):
            print(f"❌ File not found: {args.result_file}")
            return

        if args.test_mode:
            run_test_mode(result_file=args.result_file)
            return

        output_dir = args.output_dir or os.path.join(EVAL_BASE_DIR, "method", "pope")
        evaluate_result_file(args.result_file, output_dir)

        print(f"\n✅ POPE METHOD EVALUATION COMPLETE")
        return

    if args.test_mode:
        run_test_mode(finding_name=args.finding, model_dir=args.model_dir)
        return

    output_dir = args.output_dir or os.path.join(
        EVAL_BASE_DIR, args.model_dir or "all_models", args.finding)
    run_finding_evaluation(args.finding, output_dir, model_dir=args.model_dir)

    print(f"\n✅ POPE METHOD EVALUATION COMPLETE")


if __name__ == "__main__":
    main()
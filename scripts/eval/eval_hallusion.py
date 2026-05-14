"""
HallusionBench Evaluation Script

Usage:
    python eval_hallusion.py --finding1 --model_dir llava_1_5_7b --bench_path ./hallusion_bench
    python eval_hallusion.py --finding1 --finding2 --bench_path ./hallusion_bench
    python eval_hallusion.py --finding1 --bench_path ./hallusion_bench --test_mode
    python eval_hallusion.py --finding1 --bench_path ./hallusion_bench --gpt_model gpt-4o
"""

import json
import os
import re
import time
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
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

# ============================================================================
# GPT-4 JUDGE PROMPT
# (Exact reproduction from official HallusionBench/utils.py → evaluate_by_chatgpt)
# ============================================================================
HALLUSIONBENCH_JUDGE_PROMPT = (
    'Imagine you are an intelligent teacher. '
    'Thoroughly read the question, reference answer and the prediction answer '
    'to ensure a clear understanding of the information provided. '
    'Assess the correctness of the predictions. '
    'If the prediction answer does not conflict with the reference answer, '
    'please generate "correct". '
    'If the prediction answer conflict with the reference answer, '
    'please generate "incorrect". '
    'If the prediction answer is unclear about the answer, '
    'please generate "unclear".'
)

# ============================================================================
# HALLUSIONBENCH STRUCTURE
# ============================================================================
# Categories
CATEGORIES = ["VD", "VS"]  # Visual Dependent, Visual Supplement

# VD subcategories (from the uploaded image)
VD_SUBCATEGORIES = ["figure", "illusion", "math", "ocr", "safety", "video"]

# VS subcategories
VS_SUBCATEGORIES = ["chart", "map", "ocr", "table"]


# ============================================================================
# GPT-4 EVALUATOR
# ============================================================================
class GPT4Judge:
    """
    HallusionBench evaluator using GPT-4 API.
    Faithful reproduction of official evaluate_by_chatgpt() from utils.py.
    """

    def __init__(self, gpt_model="gpt-4-0613"):
        self.gpt_model = gpt_model

        try:
            from openai import OpenAI
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                print("⚠️  OPENAI_API_KEY not set! Set it with: export OPENAI_API_KEY='your-key'")
            self.client = OpenAI(api_key=api_key)
            print(f"✅ GPT-4 Judge initialized")
            print(f"   Model: {gpt_model}")
        except ImportError:
            raise ImportError("openai package required. Install: pip install openai>=1")

    def _build_prompt(self, question, gt_answer, prediction):
        """
        Build the evaluation prompt for a single sample.
        Exact format from official HallusionBench/utils.py.
        """
        prompt = HALLUSIONBENCH_JUDGE_PROMPT
        prompt += f'\n\nQuestion: {question}'
        prompt += f'\nReference answer: {gt_answer}'
        prompt += f'\nPrediction answer: {prediction}'
        return prompt

    def _call_gpt4(self, prompt, max_retries=3, retry_sleep=10):
        """Call GPT-4 API with retry logic."""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.gpt_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=16,
                )
                content = response.choices[0].message.content.strip().lower()
                return content
            except Exception as e:
                print(f"   ⚠️ API error (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_sleep)
        return None

    @staticmethod
    def _extract_judgment(response_text):
        """
        Extract judgment from GPT-4 response.
        Returns: "correct" (1), "incorrect" (0), or "unclear" (2)
        Matches official utils.py logic.
        """
        if response_text is None:
            return "unclear", 2

        text = response_text.lower().strip()

        if "correct" in text and "incorrect" not in text:
            return "correct", 1
        elif "incorrect" in text:
            return "incorrect", 0
        elif "unclear" in text:
            return "unclear", 2
        else:
            # Default to unclear if can't parse
            return "unclear", 2

    def judge_single(self, question, gt_answer, prediction):
        """
        Judge a single sample.
        Returns: {"judgment": str, "correctness": int, "raw_output": str}
        """
        prompt = self._build_prompt(question, gt_answer, prediction)
        raw = self._call_gpt4(prompt)
        judgment, correctness = self._extract_judgment(raw)

        return {
            "judgment": judgment,
            "correctness": correctness,  # 0=incorrect, 1=correct, 2=unclear
            "raw_output": raw,
        }

    def judge_batch(self, items, sleep_between=0.3):
        """
        Judge a list of items.
        items: [{"question": ..., "gt_answer": ..., "prediction": ...}, ...]
        """
        results = []
        for i, item in enumerate(tqdm(items, desc="GPT-4 judging")):
            result = self.judge_single(
                item["question"], item["gt_answer"], item["prediction"]
            )
            results.append(result)
            if sleep_between > 0 and i < len(items) - 1:
                time.sleep(sleep_between)
        return results


# ============================================================================
# HALLUSIONBENCH GROUND TRUTH LOADER
# ============================================================================
def load_hallusion_gt(bench_path):
    """
    Load HallusionBench ground truth from HallusionBench.json.
    Returns list of dicts with fields: category, subcategory, set_id, figure_id,
    question_id, question, gt_answer, etc.
    """
    gt_file = os.path.join(bench_path, "HallusionBench.json")
    if not os.path.exists(gt_file):
        print(f"❌ HallusionBench.json not found: {gt_file}")
        print(f"   Download from: https://github.com/tianyi-lab/HallusionBench")
        return None

    with open(gt_file, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    print(f"✅ Loaded HallusionBench GT: {len(gt_data)} samples")

    # Print distribution
    cat_counts = defaultdict(int)
    for item in gt_data:
        cat = item.get("category", "unknown")
        sub = item.get("subcategory", "unknown")
        cat_counts[f"{cat}/{sub}"] += 1

    print(f"   Categories:")
    for k, v in sorted(cat_counts.items()):
        print(f"      {k}: {v}")

    return gt_data


def build_gt_index(gt_data):
    """
    Build lookup index from GT data.
    Key: (category, subcategory, set_id, figure_id, question_id)
    """
    index = {}
    for item in gt_data:
        key = (
            item.get("category", ""),
            item.get("subcategory", ""),
            str(item.get("set_id", "")),
            str(item.get("figure_id", "")),
            str(item.get("question_id", "")),
        )
        index[key] = item
    return index


# ============================================================================
# HALLUSIONBENCH METRICS (from official utils.py)
# ============================================================================
def compute_hallusionbench_metrics(evaluated_results):
    """
    Compute official HallusionBench metrics.

    Metrics (from paper):
    1. Question-level Accuracy (aQ): % of individual questions answered correctly
    2. Question-Pair Accuracy (aQP): % of question pairs where BOTH questions correct
       (same set_id + question_id, different figure_id)
    3. Figure Accuracy (aF): % of figures where ALL questions about that figure are correct
       (consistency test)

    Special rules (from official code):
    - VS category with figure_id=0 (no image): "unclear" (2) counts as correct
    - For binary accuracy: correct(1)=1, incorrect(0)=0, unclear(2)=0
      EXCEPT the VS/no-image case above
    """
    if not evaluated_results:
        return {}

    # Convert to binary correctness
    for r in evaluated_results:
        raw_correctness = r.get("correctness", 0)
        category = r.get("category", "")
        figure_id = str(r.get("figure_id", ""))

        # Special rule: VS with figure_id=0, unclear counts as correct
        if category == "VS" and figure_id == "0":
            if raw_correctness == 2:  # unclear
                r["binary_correct"] = 1
            else:
                r["binary_correct"] = 1 if raw_correctness == 1 else 0
        else:
            r["binary_correct"] = 1 if raw_correctness == 1 else 0

    # ---- Question-level Accuracy ----
    total_q = len(evaluated_results)
    correct_q = sum(r["binary_correct"] for r in evaluated_results)
    q_accuracy = correct_q / total_q * 100 if total_q > 0 else 0

    # ---- Easy / Hard breakdown ----
    # Easy: original images (figure_id != 0 for VD, figure_id == 0 for VS)
    # Hard: edited images (figure_id != 0 for both, but specifically edited ones)
    # In HallusionBench: figure_id=0 in VD means original, figure_id>0 means edited
    # For VS: figure_id=0 means no image (easy), figure_id>0 means with image (hard)
    easy_results = []
    hard_results = []
    for r in evaluated_results:
        cat = r.get("category", "")
        fig_id = str(r.get("figure_id", ""))
        if cat == "VD":
            if fig_id == "0":
                easy_results.append(r)
            else:
                hard_results.append(r)
        elif cat == "VS":
            if fig_id == "0":
                easy_results.append(r)
            else:
                hard_results.append(r)

    easy_acc = (sum(r["binary_correct"] for r in easy_results) / len(easy_results) * 100
                if easy_results else 0)
    hard_acc = (sum(r["binary_correct"] for r in hard_results) / len(hard_results) * 100
                if hard_results else 0)

    # ---- Question-Pair Accuracy ----
    # Group by (category, subcategory, set_id, question_id)
    # A pair is correct only if ALL figure variants are correct
    pair_groups = defaultdict(list)
    for r in evaluated_results:
        pair_key = (
            r.get("category", ""),
            r.get("subcategory", ""),
            str(r.get("set_id", "")),
            str(r.get("question_id", "")),
        )
        pair_groups[pair_key].append(r)

    total_pairs = len(pair_groups)
    correct_pairs = 0
    for pair_key, group in pair_groups.items():
        if all(r["binary_correct"] == 1 for r in group):
            correct_pairs += 1
    qp_accuracy = correct_pairs / total_pairs * 100 if total_pairs > 0 else 0

    # ---- Figure Accuracy (Consistency Test) ----
    # Group by (category, subcategory, set_id, figure_id)
    # A figure is correct only if ALL questions about it are correct
    figure_groups = defaultdict(list)
    for r in evaluated_results:
        fig_key = (
            r.get("category", ""),
            r.get("subcategory", ""),
            str(r.get("set_id", "")),
            str(r.get("figure_id", "")),
        )
        figure_groups[fig_key].append(r)

    total_figures = len(figure_groups)
    correct_figures = 0
    for fig_key, group in figure_groups.items():
        if all(r["binary_correct"] == 1 for r in group):
            correct_figures += 1
    fig_accuracy = correct_figures / total_figures * 100 if total_figures > 0 else 0

    # ---- Per-category breakdown ----
    cat_metrics = {}
    for cat in CATEGORIES:
        cat_results = [r for r in evaluated_results if r.get("category") == cat]
        if cat_results:
            cat_correct = sum(r["binary_correct"] for r in cat_results)
            cat_metrics[cat] = {
                "total": len(cat_results),
                "correct": cat_correct,
                "accuracy": cat_correct / len(cat_results) * 100,
            }

    # ---- Per-subcategory breakdown ----
    subcat_metrics = {}
    for r in evaluated_results:
        key = f"{r.get('category', '')}/{r.get('subcategory', '')}"
        if key not in subcat_metrics:
            subcat_metrics[key] = {"total": 0, "correct": 0}
        subcat_metrics[key]["total"] += 1
        subcat_metrics[key]["correct"] += r["binary_correct"]
    for k in subcat_metrics:
        t = subcat_metrics[k]["total"]
        c = subcat_metrics[k]["correct"]
        subcat_metrics[k]["accuracy"] = c / t * 100 if t > 0 else 0

    # ---- Yes/No Bias Analysis ----
    yes_count = 0
    no_count = 0
    for r in evaluated_results:
        pred = str(r.get("prediction", "")).lower().strip()
        if pred.startswith("yes") or pred == "yes":
            yes_count += 1
        elif pred.startswith("no") or pred == "no":
            no_count += 1
    total_yn = yes_count + no_count
    yes_ratio = yes_count / total_yn * 100 if total_yn > 0 else 50

    return {
        "question_accuracy": float(q_accuracy),
        "question_pair_accuracy": float(qp_accuracy),
        "figure_accuracy": float(fig_accuracy),
        "easy_accuracy": float(easy_acc),
        "hard_accuracy": float(hard_acc),
        "total_questions": total_q,
        "correct_questions": correct_q,
        "total_pairs": total_pairs,
        "correct_pairs": correct_pairs,
        "total_figures": total_figures,
        "correct_figures": correct_figures,
        "category_breakdown": cat_metrics,
        "subcategory_breakdown": subcat_metrics,
        "yes_no_bias": {
            "yes_count": yes_count,
            "no_count": no_count,
            "yes_ratio": float(yes_ratio),
        },
        "easy_count": len(easy_results),
        "hard_count": len(hard_results),
    }


# ============================================================================
# FILE DISCOVERY
# ============================================================================
def discover_result_files(finding_name, model_dir=None):
    """Discover inference result files for a HallusionBench finding."""
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
def evaluate_hallusion_results(results, gt_data, judge):
    """
    Evaluate inference results against HallusionBench ground truth.
    """
    gt_index = build_gt_index(gt_data)

    # Match results to GT and prepare for judging
    items_to_judge = []
    matched_results = []

    for r in results:
        # Try to match with GT
        key = (
            r.get("category", ""),
            r.get("subcategory", ""),
            str(r.get("set_id", "")),
            str(r.get("figure_id", "")),
            str(r.get("question_id", "")),
        )

        gt_entry = gt_index.get(key)
        if gt_entry is None:
            # Try alternative matching by index or id
            sample_id = r.get("id", r.get("sample_id", ""))
            if isinstance(sample_id, int) and sample_id < len(gt_data):
                gt_entry = gt_data[sample_id]

        if gt_entry is None:
            continue

        question = gt_entry.get("question", r.get("question", ""))
        gt_answer = gt_entry.get("gt_answer", gt_entry.get("answer", ""))
        prediction = r.get("response", r.get("prediction", ""))

        items_to_judge.append({
            "question": question,
            "gt_answer": gt_answer,
            "prediction": prediction,
        })
        matched_results.append({
            **r,
            "question": question,
            "gt_answer": gt_answer,
            "prediction": prediction,
            "category": gt_entry.get("category", r.get("category", "")),
            "subcategory": gt_entry.get("subcategory", r.get("subcategory", "")),
            "set_id": gt_entry.get("set_id", r.get("set_id", "")),
            "figure_id": gt_entry.get("figure_id", r.get("figure_id", "")),
            "question_id": gt_entry.get("question_id", r.get("question_id", "")),
        })

    if not items_to_judge:
        print("❌ No samples matched with HallusionBench GT!")
        return None

    print(f"   Matched {len(items_to_judge)} samples with GT")

    # Judge with GPT-4
    judge_results = judge.judge_batch(items_to_judge)

    # Merge results
    evaluated = []
    for result, judgment in zip(matched_results, judge_results):
        evaluated.append({
            **result,
            "judgment": judgment["judgment"],
            "correctness": judgment["correctness"],
            "raw_judge_output": judgment["raw_output"],
        })

    # Compute metrics
    metrics = compute_hallusionbench_metrics(evaluated)

    # Per-emotion breakdown
    emotion_metrics = defaultdict(list)
    for r in evaluated:
        emotion = r.get("emotion_category", "neutral")
        emotion_metrics[emotion].append(r["binary_correct"])

    emotion_summary = {}
    for emo, vals in sorted(emotion_metrics.items()):
        emotion_summary[emo] = {
            "total": len(vals),
            "correct": sum(vals),
            "accuracy": sum(vals) / len(vals) * 100 if vals else 0,
        }

    return {
        "metadata": {
            "evaluator_model": judge.gpt_model,
            "evaluation_date": datetime.now().isoformat(),
            "evaluation_method": "hallusionbench_gpt4_judge",
            "dataset": "hallusionbench",
        },
        "summary": metrics,
        "emotion_breakdown": emotion_summary,
        "detailed_results": evaluated,
    }


# ============================================================================
# FINDING-LEVEL EVALUATION
# ============================================================================
def run_finding_evaluation(finding_name, judge, gt_data, output_dir,
                           skip_neutral=False, model_dir=None):
    """Evaluate all result files for a finding."""
    result_files = discover_result_files(finding_name, model_dir)
    if not result_files:
        return None

    if skip_neutral:
        result_files = [f for f in result_files if "NEUTRAL" not in f.name.upper()]

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"{finding_name.upper()} EVALUATION (HallusionBench — GPT-4 judge)")
    print(f"{'='*80}")
    print(f"Judge: {judge.gpt_model}")
    print(f"Method: Official HallusionBench GPT-4 judge (correct/incorrect/unclear)")
    print(f"Files: {len(result_files)}")
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

        evaluation = evaluate_hallusion_results(results, gt_data, judge)
        if evaluation is None:
            continue

        # Save detailed
        eval_path = os.path.join(output_dir, f"{ds_name}_evaluated.json")
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(evaluation, f, indent=2, ensure_ascii=False)

        s = evaluation["summary"]
        print(f"   ✅ Question Accuracy:      {s['question_accuracy']:.1f}%")
        print(f"      Question-Pair Accuracy: {s['question_pair_accuracy']:.1f}%")
        print(f"      Figure Accuracy:        {s['figure_accuracy']:.1f}%")
        print(f"      Easy / Hard:            {s['easy_accuracy']:.1f}% / {s['hard_accuracy']:.1f}%")

        if s.get("category_breakdown"):
            for cat, cm in s["category_breakdown"].items():
                print(f"      {cat}: {cm['accuracy']:.1f}% (n={cm['total']})")

        is_neutral = "NEUTRAL" in ds_name.upper()
        condition = "neutral" if is_neutral else emotion_category

        row = {
            "dataset": ds_name,
            "finding": finding,
            "condition": condition,
            "emotion_category": emotion_category,
            "subject": subject,
            "total_questions": s["total_questions"],
            "question_accuracy": s["question_accuracy"],
            "question_pair_accuracy": s["question_pair_accuracy"],
            "figure_accuracy": s["figure_accuracy"],
            "easy_accuracy": s["easy_accuracy"],
            "hard_accuracy": s["hard_accuracy"],
            "yes_ratio": s["yes_no_bias"]["yes_ratio"],
        }
        # Add per-category
        for cat in CATEGORIES:
            if cat in s.get("category_breakdown", {}):
                row[f"acc_{cat}"] = s["category_breakdown"][cat]["accuracy"]

        comparison_data.append(row)

    if not comparison_data:
        print("❌ No evaluations completed.")
        return None

    # Save comparison
    df = pd.DataFrame(comparison_data)
    df.to_csv(os.path.join(output_dir, f"{finding_name}_comparison.csv"), index=False)
    with open(os.path.join(output_dir, f"{finding_name}_comparison.json"), "w") as f:
        json.dump(comparison_data, f, indent=2)

    # Print comparison
    print(f"\n{'='*80}")
    print(f"{finding_name.upper()} COMPARISON (HallusionBench)")
    print(f"{'='*80}")
    print(f"{'Condition':<30} {'Q-Acc':>8} {'QP-Acc':>8} {'Fig-Acc':>8} {'Easy':>8} {'Hard':>8}")
    print(f"{'-'*80}")
    df_sorted = df.sort_values("question_accuracy", ascending=False)
    for _, row in df_sorted.iterrows():
        subj = f" [{row['subject']}]" if row.get("subject") else ""
        cond = f"{row['condition']}{subj}"
        print(f"{cond:<30} {row['question_accuracy']:>7.1f}% {row['question_pair_accuracy']:>7.1f}% "
              f"{row['figure_accuracy']:>7.1f}% {row['easy_accuracy']:>7.1f}% {row['hard_accuracy']:>7.1f}%")

    # Plot
    create_plot(df, finding_name, output_dir)

    # Summary
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "finding": finding_name,
        "dataset": "hallusionbench",
        "evaluator_model": judge.gpt_model,
        "evaluation_method": "hallusionbench_gpt4_judge",
        "evaluation_date": datetime.now().isoformat(),
        "files_evaluated": len(result_files),
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
    """Create comparison plot with multiple HallusionBench metrics."""
    sns.set_style("whitegrid")
    if len(df) == 0:
        return

    # Plot Question Accuracy + Question-Pair Accuracy side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(df) * 0.6)))

    df_sorted = df.sort_values("question_accuracy")

    labels = []
    for _, row in df_sorted.iterrows():
        lbl = row["condition"]
        if row.get("subject") and row["subject"] not in ("", "none", "unknown"):
            lbl += f" [{row['subject']}]"
        labels.append(lbl)

    colors = [COLOR_MAP.get(row["condition"], "#333") for _, row in df_sorted.iterrows()]

    # Left: Question Accuracy
    ax = axes[0]
    bars = ax.barh(range(len(df_sorted)), df_sorted["question_accuracy"], color=colors)
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Question Accuracy (%)", fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_title("Question-level Accuracy", fontsize=11)
    for bar, val in zip(bars, df_sorted["question_accuracy"]):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", ha="left", va="center", fontsize=8)

    # Right: Question-Pair Accuracy
    ax = axes[1]
    bars = ax.barh(range(len(df_sorted)), df_sorted["question_pair_accuracy"], color=colors)
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Question-Pair Accuracy (%)", fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_title("Question-Pair Accuracy (stricter)", fontsize=11)
    for bar, val in zip(bars, df_sorted["question_pair_accuracy"]):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", ha="left", va="center", fontsize=8)

    title_map = {
        "hallusion_finding1": "HallusionBench Finding 1: Emotion vs Neutral",
        "hallusion_finding2": "HallusionBench Finding 2: By Emotion Category",
        "hallusion_finding3": "HallusionBench Finding 3: Subject (I vs YOU)",
        "hallusion_finding4": "HallusionBench Finding 4: Visual Emotion",
    }
    fig.suptitle(title_map.get(finding_name, f"HallusionBench — {finding_name}"),
                 fontsize=13, y=1.02)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, f"{finding_name}_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 Plot saved: {plot_path}")


# ============================================================================
# TEST MODE
# ============================================================================
NUM_TEST_SAMPLES = 3


def run_test_mode(judge, gt_data, finding_name, model_dir=None):
    """Test mode: judge 3 samples with detailed output."""
    print(f"\n{'='*80}")
    print("TEST MODE — HallusionBench Evaluation (GPT-4 judge)")
    print(f"{'='*80}")
    print(f"  Finding:   {finding_name}")
    print(f"  GPT model: {judge.gpt_model}")
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

    # Step 3: Judge a few samples
    print(f"\n{'='*80}")
    print(f"STEP 3: JUDGING {NUM_TEST_SAMPLES} SAMPLES")
    print(f"{'='*80}")

    gt_index = build_gt_index(gt_data)
    scored = 0

    for sample in results:
        if scored >= NUM_TEST_SAMPLES:
            break

        key = (
            sample.get("category", ""),
            sample.get("subcategory", ""),
            str(sample.get("set_id", "")),
            str(sample.get("figure_id", "")),
            str(sample.get("question_id", "")),
        )
        gt_entry = gt_index.get(key)
        if gt_entry is None:
            continue

        question = gt_entry.get("question", "")
        gt_answer = gt_entry.get("gt_answer", gt_entry.get("answer", ""))
        prediction = sample.get("response", sample.get("prediction", ""))

        print(f"\n  [{scored+1}/{NUM_TEST_SAMPLES}]")
        print(f"  Category:   {sample.get('category')}/{sample.get('subcategory')}")
        print(f"  Set/Fig/Q:  {sample.get('set_id')}/{sample.get('figure_id')}/{sample.get('question_id')}")
        print(f"  Question:   {question[:100]}{'...' if len(question) > 100 else ''}")
        print(f"  GT:         {gt_answer[:100]}{'...' if len(gt_answer) > 100 else ''}")
        print(f"  Prediction: {prediction[:100]}{'...' if len(prediction) > 100 else ''}")

        result = judge.judge_single(question, gt_answer, prediction)
        print(f"  Judgment:   {result['judgment']} (code={result['correctness']})")
        print(f"  Raw output: {result['raw_output']}")

        scored += 1

    print(f"\n✅ Test complete. Run without --test_mode for full evaluation.")


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="HallusionBench Evaluation (GPT-4 judge)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
HallusionBench (Guan et al., CVPR 2024) diagnoses hallucination/illusion in LVLMs.
346 images × 1129 questions. GPT-4 judges: correct/incorrect/unclear.

Metrics:
  Question Accuracy (aQ):      % of individual questions correct
  Question-Pair Accuracy (aQP): % of question pairs where BOTH variants correct
  Figure Accuracy (aF):         % of figures where ALL questions correct (consistency)

Requires: OPENAI_API_KEY environment variable.

Examples:
  python eval_hallusion.py --finding1 --model_dir llava_1_5_7b --bench_path ./hallusion_bench
  python eval_hallusion.py --finding1 --finding2 --bench_path ./hallusion_bench
  python eval_hallusion.py --finding1 --bench_path ./hallusion_bench --test_mode
  python eval_hallusion.py --finding1 --bench_path ./hallusion_bench --gpt_model gpt-4o
        """,
    )

    parser.add_argument("--finding1", action="store_true")
    parser.add_argument("--finding2", action="store_true")
    parser.add_argument("--finding3", action="store_true")
    parser.add_argument("--finding4", action="store_true")

    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--bench_path", type=str, required=True,
                        help="Path to HallusionBench directory (contains HallusionBench.json)")
    parser.add_argument("--gpt_model", type=str, default="gpt-4-0613",
                        help="GPT model for judging (default: gpt-4-0613)")

    parser.add_argument("--skip_neutral", action="store_true")
    parser.add_argument("--test_mode", action="store_true",
                        help="Judge 3 samples with detailed output to verify pipeline")

    args = parser.parse_args()

    findings = {
        "hallusion_finding1": args.finding1,
        "hallusion_finding2": args.finding2,
        "hallusion_finding3": args.finding3,
        "hallusion_finding4": args.finding4,
    }
    if not any(findings.values()):
        parser.error("Specify at least one: --finding1, --finding2, --finding3, --finding4")

    eval_subdir = args.model_dir if args.model_dir else "all_models"

    print(f"\n{'='*80}")
    print("HALLUSIONBENCH EVALUATION (GPT-4 judge)")
    print(f"{'='*80}")
    print(f"GPT model:  {args.gpt_model}")
    print(f"Bench path: {args.bench_path}")
    print(f"Input:      {INFER_BASE_DIR}" + (f"/{args.model_dir}" if args.model_dir else " (all)"))
    print(f"Output:     {EVAL_BASE_DIR}/{eval_subdir}")

    # Load ground truth
    gt_data = load_hallusion_gt(args.bench_path)
    if gt_data is None:
        return

    # Initialize judge
    judge = GPT4Judge(gpt_model=args.gpt_model)

    # Test mode
    if args.test_mode:
        first_finding = next(name for name, sel in findings.items() if sel)
        run_test_mode(judge, gt_data, first_finding, model_dir=args.model_dir)
        return

    # Full evaluation
    for finding_name, selected in findings.items():
        if not selected:
            continue
        output_dir = os.path.join(EVAL_BASE_DIR, eval_subdir, finding_name)
        run_finding_evaluation(
            finding_name=finding_name,
            judge=judge,
            gt_data=gt_data,
            output_dir=output_dir,
            skip_neutral=args.skip_neutral,
            model_dir=args.model_dir,
        )

    print(f"\n{'='*80}")
    print("✅ ALL HALLUSIONBENCH EVALUATIONS COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
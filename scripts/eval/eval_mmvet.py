"""
MM-Vet Utility Evaluation Script

Faithful reproduction of official MM-Vet evaluator (Yu et al., ICML 2024):
  https://github.com/yuweihao/MM-Vet/blob/main/mm-vet_evaluator.py

Methodology:
  - LLM judge scores each prediction against GT on 0.0-1.0 scale
  - Few-shot prompt with AND/OR logic for multi-element answers
  - Multiple independent scoring runs averaged (default 5 for GPT-4, 1 for local)
  - Per-capability and per-capability-integration breakdowns
  - 6 core capabilities: rec, know, gen, spat, ocr, math

Evaluator options:
  - GPT-4 API (gpt-4-0613): Faithful to paper, recommended for final results
  - Local LLM (Llama-3-8B-Instruct): FREE, for pipeline testing

Usage:
    # Test with local model (FREE, no API needed):
    python eval_mmvet.py --finding1 --model_dir llava_1_5_7b \\
        --mmvet_path ./mm-vet --use_local_judge --load_in_4bit --test_mode

    # Full eval with GPT-4 (faithful to paper):
    python eval_mmvet.py --finding1 --model_dir llava_1_5_7b \\
        --mmvet_path ./mm-vet --gpt_model gpt-4-0613

    # Full eval with GPT-4o (cheaper):
    python eval_mmvet.py --finding1 --mmvet_path ./mm-vet --gpt_model gpt-4o --num_runs 3
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
# MM-VET GPT-4 SCORING PROMPT
# (Exact reproduction from official MM-Vet evaluator: mm-vet_evaluator.py)
# ============================================================================
MMVET_PROMPT = """Compare the ground truth and prediction from AI models, to give a correctness score for the prediction. <AND> in the ground truth means it is totally right only when all elements in the ground truth are present in the prediction, and <OR> means it is totally right when any one element in the ground truth is present in the prediction. The correctness score is 0.0 (totally wrong), 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, or 1.0 (totally right). Just complete the last space of the correctness score.\n\n"""

# Few-shot examples (from official MM-Vet evaluator notebook)
MMVET_FEW_SHOT_EXAMPLES = [
    {
        "question": "Are the butterflies the same or different?",
        "gt": "The butterflies are different.",
        "prediction": "The butterflies in the image are the same species.",
        "score": "0.0",
    },
    {
        "question": "How many tomatoes are there?",
        "gt": "5",
        "prediction": "There are 5 tomatoes in the image.",
        "score": "1.0",
    },
    {
        "question": "What is the name and year of this painting?",
        "gt": "erta de Milo <AND> 130 BC <OR> Venus de Milo <AND> 130 BC",
        "prediction": "This painting is called 'Venus de Milo' and it was created in 130 BC.",
        "score": "1.0",
    },
    {
        "question": "What is unusual about this image?",
        "gt": "The unusual thing about this image is that a man is ironing clothes on the back of a taxi.",
        "prediction": "The image shows a busy street scene with people and cars.",
        "score": "0.0",
    },
    {
        "question": "Can you explain this meme?",
        "gt": "This meme is poking fun at the fact that the names of the countries Iceland and Greenland are misleading. Despite its name, Iceland is known for its beautiful green landscapes, while Greenland is mostly covered in ice and snow. The meme is saying that the person has trust issues because the names of these countries do not accurately represent their landscapes.",
        "prediction": "The meme shows two images: one of a green, hilly landscape labeled 'Iceland' and one of a snowy, icy landscape labeled 'Greenland'. The joke is that the names of these two countries seem to be swapped, as Iceland appears green and Greenland appears icy. The text 'This is why I have trust issues' humorously expresses frustration at this misleading naming convention.",
        "score": "1.0",
    },
    {
        "question": "Can you explain this meme?",
        "gt": "This meme is poking fun at the fact that the names of the countries Iceland and Greenland are misleading. Despite its name, Iceland is known for its beautiful green landscapes, while Greenland is mostly covered in ice and snow. The meme is saying that the person has trust issues because the names of these countries do not accurately represent their landscapes.",
        "prediction": "The meme shows pictures of Iceland and Greenland. It's a funny image about geography.",
        "score": "0.3",
    },
    {
        "question": "What time is it and what is the weather?",
        "gt": "3:15 PM <AND> sunny",
        "prediction": "It is sunny in the image.",
        "score": "0.5",
    },
]

# ============================================================================
# MM-VET CAPABILITIES
# ============================================================================
CAPABILITIES = ["rec", "know", "gen", "spat", "ocr", "math"]
CAPABILITY_NAMES = {
    "rec": "Recognition",
    "know": "Knowledge",
    "gen": "Language Generation",
    "spat": "Spatial Awareness",
    "ocr": "OCR",
    "math": "Math",
}


def _build_few_shot_prompt():
    """Build the few-shot prompt string (shared by both evaluators)."""
    prompt = MMVET_PROMPT
    for ex in MMVET_FEW_SHOT_EXAMPLES:
        prompt += (
            f"Question | Ground truth | Prediction | Correctness\n"
            f"{ex['question']} | {ex['gt']} | {ex['prediction']} | {ex['score']}\n\n"
        )
    return prompt


def _extract_score(response_text):
    """Extract numeric score from LLM response. Shared by both evaluators."""
    if response_text is None:
        return 0.0

    # Priority 1: Match X.X format (e.g., "0.7", "1.0")
    match = re.search(r'(\d\.\d)', response_text)
    if match:
        score = float(match.group(1))
        return min(max(score, 0.0), 1.0)

    # Priority 2: Match single digit
    match = re.search(r'(\d)', response_text)
    if match:
        val = int(match.group(1))
        if val == 0:
            return 0.0
        elif val == 1:
            return 1.0
        else:
            return val / 10.0

    return 0.0


# ============================================================================
# LOCAL LLM EVALUATOR (for pipeline testing without API cost)
# ============================================================================
class LocalLLMEvaluator:
    """
    MM-Vet evaluator using local Llama-3-8B-Instruct.
    Same few-shot prompt as GPT-4, but runs locally.

    NOTE: For final paper results, use GPT-4 (GPT4Evaluator).
    Local model is for pipeline testing only.
    """

    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B-Instruct",
                 load_in_4bit=False, load_in_8bit=False, num_runs=1):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.model_name = model_name
        self.num_runs = num_runs
        self.gpt_model = f"local:{model_name.split('/')[-1]}"
        self.prompt = _build_few_shot_prompt()

        print(f"🔍 Loading local evaluator: {model_name}")
        if load_in_4bit:
            print("   Using 4-bit quantization")
        elif load_in_8bit:
            print("   Using 8-bit quantization")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.float16,
        }

        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
        elif load_in_8bit:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.device = self.model.device
        print(f"✅ Local evaluator loaded on {self.device}")

    def _build_query(self, question, gt, prediction):
        query = (
            self.prompt
            + f"Question | Ground truth | Prediction | Correctness\n"
            + f"{question} | {gt} | {prediction} | "
        )
        return query

    def _call_local(self, query):
        import torch
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a scoring assistant. Given a table of Question, Ground truth, "
                    "Prediction, and Correctness examples, you must complete the last row "
                    "by outputting ONLY a single correctness score (e.g. 0.0, 0.3, 0.5, 0.7, 1.0). "
                    "Do NOT repeat the table. Output ONLY the score number."
                ),
            },
            {"role": "user", "content": query},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=16,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return response

    def score_single(self, question, gt, prediction):
        query = self._build_query(question, gt, prediction)
        scores = []
        raw_outputs = []
        for _ in range(self.num_runs):
            raw = self._call_local(query)
            raw_outputs.append(raw)
            score = _extract_score(raw)
            scores.append(score)
        return {
            "mean_score": float(np.mean(scores)),
            "scores": scores,
            "raw_outputs": raw_outputs,
        }

    def score_batch(self, items, sleep_between=0):
        results = []
        for item in tqdm(items, desc=f"Local LLM scoring ({self.num_runs} run(s))"):
            result = self.score_single(item["question"], item["gt"], item["prediction"])
            results.append(result)
        return results


# ============================================================================
# GPT-4 EVALUATOR (faithful to official MM-Vet)
# ============================================================================
class GPT4Evaluator:
    """
    MM-Vet evaluator using GPT-4 API.
    Faithful reproduction of official mm-vet_evaluator.py methodology.
    """

    def __init__(self, gpt_model="gpt-4-0613", num_runs=5):
        self.gpt_model = gpt_model
        self.num_runs = num_runs
        self.prompt = _build_few_shot_prompt()

        try:
            from openai import OpenAI
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                print("⚠️  OPENAI_API_KEY not set! Set it with: export OPENAI_API_KEY='your-key'")
            self.client = OpenAI(api_key=api_key)
            print(f"✅ GPT-4 Evaluator initialized")
            print(f"   Model: {gpt_model}")
            print(f"   Runs: {num_runs}")
        except ImportError:
            raise ImportError("openai package required. Install: pip install openai>=1")

    def _build_query(self, question, gt, prediction):
        query = (
            self.prompt
            + f"Question | Ground truth | Prediction | Correctness\n"
            + f"{question} | {gt} | {prediction} | "
        )
        return query

    def _call_gpt4(self, query, max_retries=3, retry_sleep=10):
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.gpt_model,
                    messages=[{"role": "user", "content": query}],
                    temperature=0.0,
                    max_tokens=16,
                )
                content = response.choices[0].message.content.strip()
                return content
            except Exception as e:
                print(f"   ⚠️ API error (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_sleep)
        return None

    def score_single(self, question, gt, prediction):
        query = self._build_query(question, gt, prediction)
        scores = []
        raw_outputs = []
        for _ in range(self.num_runs):
            raw = self._call_gpt4(query)
            raw_outputs.append(raw)
            score = _extract_score(raw)
            scores.append(score)
        return {
            "mean_score": float(np.mean(scores)),
            "scores": scores,
            "raw_outputs": raw_outputs,
        }

    def score_batch(self, items, sleep_between=0.5):
        results = []
        for i, item in enumerate(tqdm(items, desc=f"GPT-4 scoring ({self.num_runs} runs)")):
            result = self.score_single(item["question"], item["gt"], item["prediction"])
            results.append(result)
            if sleep_between > 0 and i < len(items) - 1:
                time.sleep(sleep_between)
        return results



# ============================================================================
# MM-VET GROUND TRUTH LOADER
# ============================================================================
def load_mmvet_gt(mmvet_path):
    gt_file = os.path.join(mmvet_path, "mm-vet.json")
    if not os.path.exists(gt_file):
        print(f"❌ MM-Vet ground truth not found: {gt_file}")
        print(f"   Download from: https://github.com/yuweihao/MM-Vet/releases/download/v1/mm-vet.zip")
        return None
    with open(gt_file, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    print(f"✅ Loaded MM-Vet GT: {len(gt_data)} samples")
    return gt_data


def get_capability_integration(caps):
    return "+".join(sorted(caps))


# ============================================================================
# FILE DISCOVERY
# ============================================================================
def discover_result_files(finding_name, model_dir=None):
    found_files = []
    if model_dir:
        search_dirs = [os.path.join(INFER_BASE_DIR, model_dir, finding_name)]
    else:
        base = Path(INFER_BASE_DIR)
        if not base.exists():
            print(f"❌ Results base directory not found: {INFER_BASE_DIR}")
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
def evaluate_mmvet_results(results, gt_data, evaluator):
    items_to_score = []
    matched_results = []

    for r in results:
        sample_id = r.get("id", r.get("sample_id", ""))
        response = r.get("response", "")
        if sample_id in gt_data:
            gt_entry = gt_data[sample_id]
            items_to_score.append({
                "question": gt_entry.get("question", ""),
                "gt": gt_entry.get("answer", ""),
                "prediction": response,
            })
            matched_results.append({
                **r,
                "gt_question": gt_entry.get("question", ""),
                "gt_answer": gt_entry.get("answer", ""),
                "capabilities": gt_entry.get("capability", []),
            })
        else:
            print(f"   ⚠️ Sample '{sample_id}' not found in MM-Vet GT, skipping")

    if not items_to_score:
        print("❌ No samples matched with MM-Vet GT!")
        return None

    print(f"   Matched {len(items_to_score)} samples with GT")
    score_results = evaluator.score_batch(items_to_score)

    evaluated = []
    cap_scores = defaultdict(list)
    cap_int_scores = defaultdict(list)
    emotion_scores = defaultdict(list)
    all_scores = []

    for result, score_info in zip(matched_results, score_results):
        mean_score = score_info["mean_score"]
        capabilities = result.get("capabilities", [])
        emotion = result.get("emotion_category", "neutral")

        all_scores.append(mean_score)
        for cap in capabilities:
            cap_scores[cap].append(mean_score)
        cap_int = get_capability_integration(capabilities)
        cap_int_scores[cap_int].append(mean_score)
        emotion_scores[emotion].append(mean_score)

        evaluated.append({
            **result,
            "mmvet_score": mean_score,
            "mmvet_scores_per_run": score_info["scores"],
            "mmvet_raw_outputs": score_info["raw_outputs"],
        })

    overall_score = np.mean(all_scores) * 100

    cap_summary = {}
    for cap in CAPABILITIES:
        if cap in cap_scores:
            cap_summary[cap] = {
                "score": float(np.mean(cap_scores[cap]) * 100),
                "count": len(cap_scores[cap]),
                "name": CAPABILITY_NAMES.get(cap, cap),
            }

    cap_int_summary = {}
    for cap_int, scores in sorted(cap_int_scores.items(), key=lambda x: -len(x[1])):
        cap_int_summary[cap_int] = {
            "score": float(np.mean(scores) * 100),
            "count": len(scores),
        }

    emotion_summary = {}
    for emo, scores in sorted(emotion_scores.items()):
        emotion_summary[emo] = {
            "score": float(np.mean(scores) * 100),
            "count": len(scores),
        }

    return {
        "metadata": {
            "evaluator_model": evaluator.gpt_model,
            "evaluation_date": datetime.now().isoformat(),
            "evaluation_method": "mmvet_llm_scoring",
            "dataset": "mm-vet",
            "num_runs": evaluator.num_runs,
        },
        "summary": {
            "total_samples": len(evaluated),
            "overall_score": float(overall_score),
            "mean_raw_score": float(np.mean(all_scores)),
            "std_raw_score": float(np.std(all_scores)),
        },
        "capability_scores": cap_summary,
        "capability_integration_scores": cap_int_summary,
        "emotion_breakdown": emotion_summary,
        "detailed_results": evaluated,
    }


# ============================================================================
# FINDING-LEVEL EVALUATION
# ============================================================================
def run_finding_evaluation(finding_name, evaluator, gt_data, output_dir,
                           skip_neutral=False, model_dir=None):
    result_files = discover_result_files(finding_name, model_dir)
    if not result_files:
        return None

    if skip_neutral:
        result_files = [f for f in result_files if "NEUTRAL" not in f.name.upper()]

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"{finding_name.upper()} EVALUATION (MM-Vet)")
    print(f"{'='*80}")
    print(f"Evaluator: {evaluator.gpt_model} ({evaluator.num_runs} runs)")
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
        print(f"Samples: {len(results)}")

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

        evaluation = evaluate_mmvet_results(results, gt_data, evaluator)
        if evaluation is None:
            continue

        eval_path = os.path.join(output_dir, f"{ds_name}_evaluated.json")
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(evaluation, f, indent=2, ensure_ascii=False)

        s = evaluation["summary"]
        print(f"✅ Overall score: {s['overall_score']:.1f}%")

        cap_scores = evaluation.get("capability_scores", {})
        if cap_scores:
            print(f"   Per-capability:")
            for cap in CAPABILITIES:
                if cap in cap_scores:
                    cs = cap_scores[cap]
                    print(f"      {cs['name']:<25} {cs['score']:.1f}% (n={cs['count']})")

        is_neutral = "NEUTRAL" in ds_name.upper()
        condition = "neutral" if is_neutral else emotion_category

        row = {
            "dataset": ds_name,
            "finding": finding,
            "condition": condition,
            "emotion_category": emotion_category,
            "subject": subject,
            "total_samples": s["total_samples"],
            "overall_score": s["overall_score"],
        }
        for cap in CAPABILITIES:
            if cap in cap_scores:
                row[f"cap_{cap}"] = cap_scores[cap]["score"]
            else:
                row[f"cap_{cap}"] = None
        comparison_data.append(row)

    if not comparison_data:
        print("❌ No evaluations completed.")
        return None

    df = pd.DataFrame(comparison_data)
    df.to_csv(os.path.join(output_dir, f"{finding_name}_comparison.csv"), index=False)
    with open(os.path.join(output_dir, f"{finding_name}_comparison.json"), "w") as f:
        json.dump(comparison_data, f, indent=2)

    print(f"\n{'='*80}")
    print(f"{finding_name.upper()} COMPARISON (MM-Vet Score %)")
    print(f"{'='*80}")
    df_sorted = df.sort_values("overall_score", ascending=False)
    for _, row in df_sorted.iterrows():
        subj = f" [{row['subject']}]" if row.get("subject") else ""
        cap_str = "  ".join([
            f"{cap}:{row[f'cap_{cap}']:.0f}" for cap in CAPABILITIES
            if row.get(f'cap_{cap}') is not None
        ])
        print(f"  {row['condition']:<30}{subj:<6}  "
              f"Score: {row['overall_score']:.1f}%  | {cap_str}")

    create_plot(df, finding_name, output_dir)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "finding": finding_name,
        "dataset": "mm-vet",
        "evaluator_model": evaluator.gpt_model,
        "num_runs": evaluator.num_runs,
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
    "positive_high_arousal": "#2ecc71", "positive_low_arousal": "#27ae60",
    "negative_high_arousal": "#e74c3c", "negative_low_arousal": "#c0392b",
    "positive_high": "#2ecc71", "positive_low": "#27ae60",
    "negative_high": "#e74c3c", "negative_low": "#c0392b",
    "empathy": "#9b59b6", "psychological": "#3498db", "neutral": "#808080",
}

def create_plot(df, finding_name, output_dir):
    sns.set_style("whitegrid")
    if len(df) == 0:
        return
    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.6)))
    df_sorted = df.sort_values("overall_score")
    labels = []
    for _, row in df_sorted.iterrows():
        lbl = row["condition"]
        if row.get("subject") and row["subject"] not in ("", "none", "unknown"):
            lbl += f" [{row['subject']}]"
        labels.append(lbl)
    colors = [COLOR_MAP.get(row["condition"], "#333") for _, row in df_sorted.iterrows()]
    bars = ax.barh(range(len(df_sorted)), df_sorted["overall_score"], color=colors)
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("MM-Vet Score (%)", fontsize=11)
    ax.set_xlim(0, 100)
    for bar, val in zip(bars, df_sorted["overall_score"]):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", ha="left", va="center", fontsize=9)
    title_map = {
        "mmvet_finding1": "MM-Vet Finding 1: Utility — Emotion vs Neutral",
        "mmvet_finding2": "MM-Vet Finding 2: Utility by Emotion Category",
        "mmvet_finding3": "MM-Vet Finding 3: Subject Comparison (I vs YOU)",
        "mmvet_finding4": "MM-Vet Finding 4: Utility by Emotion Category",
    }
    ax.set_title(title_map.get(finding_name, f"MM-Vet — {finding_name}"), fontsize=13)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{finding_name}_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 Plot saved: {plot_path}")


# ============================================================================
# TEST MODE
# ============================================================================
NUM_TEST_SAMPLES = 5

def run_test_mode(evaluator, gt_data, finding_name, model_dir=None):
    print(f"\n{'='*80}")
    print("TEST MODE — MM-Vet Evaluation")
    print(f"{'='*80}")
    print(f"  Finding:   {finding_name}")
    print(f"  Evaluator: {evaluator.gpt_model}")
    print(f"  Runs:      {evaluator.num_runs}")
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
    ds_name = parse_dataset_name(test_file)
    print(f"\n  Using: {test_file.name}")
    print(f"  Parsed dataset name: {ds_name}")

    # Step 2: Load and inspect
    print(f"\n{'='*80}")
    print("STEP 2: DATA INSPECTION")
    print(f"{'='*80}")
    with open(test_file, "r", encoding="utf-8") as f:
        results = json.load(f)
    print(f"  Total samples in file: {len(results)}")
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
    matched = sum(1 for r in results if r.get("id", r.get("sample_id", "")) in gt_data)
    error_count = sum(1 for r in results if r.get("response", "").startswith("[ERROR"))
    print(f"\n  GT matched: {matched}/{len(results)}")
    print(f"  Error responses: {error_count}/{len(results)}")
    emotion_category = r0.get("emotion_category", "unknown")
    print(f"\n  Extracted metadata:")
    print(f"    emotion_category: {emotion_category}")
    print(f"    finding:          {r0.get('finding', finding_name)}")

    # Step 3: Score samples
    print(f"\n{'='*80}")
    print(f"STEP 3: SCORING {NUM_TEST_SAMPLES} SAMPLES")
    print(f"{'='*80}")
    test_results = []
    scored = 0
    for sample in results:
        if scored >= NUM_TEST_SAMPLES:
            break
        sample_id = sample.get("id", sample.get("sample_id", ""))
        response = sample.get("response", "")
        if sample_id not in gt_data:
            continue
        gt_entry = gt_data[sample_id]
        question = gt_entry.get("question", "")
        gt_answer = gt_entry.get("answer", "")
        capabilities = gt_entry.get("capability", [])
        print(f"\n  [{scored+1}/{NUM_TEST_SAMPLES}] ID: {sample_id}")
        print(f"  Question:     {question[:100]}{'...' if len(question) > 100 else ''}")
        print(f"  GT:           {gt_answer[:100]}{'...' if len(gt_answer) > 100 else ''}")
        print(f"  Prediction:   {response[:100]}{'...' if len(response) > 100 else ''}")
        print(f"  Capabilities: {capabilities}")
        score_info = evaluator.score_single(question, gt_answer, response)
        test_results.append({"id": sample_id, "capabilities": capabilities, **score_info})
        print(f"  Raw output:   {score_info['raw_outputs']}")
        print(f"  Score:        {score_info['mean_score']:.2f}")
        scored += 1

    # Save
    os.makedirs("./eval_tests/mmvet", exist_ok=True)
    test_out = f"./eval_tests/mmvet/test_mode_{model_dir or 'all'}_results.json"
    with open(test_out, "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2)
    print(f"\n💾 Test results saved: {test_out}")

    # Summary
    if test_results:
        mean_overall = np.mean([r["mean_score"] for r in test_results]) * 100
        print(f"\n{'='*80}")
        print("STEP 4: TEST SUMMARY")
        print(f"{'='*80}")
        print(f"  Files discovered:      {len(result_files)}")
        print(f"  Samples in first file: {len(results)}")
        print(f"  GT matched:            {matched}/{len(results)}")
        print(f"  Emotion category:      {emotion_category}")
        print(f"  Dataset name parsed:   {ds_name}")
        print(f"\n  Test scoring ({scored} samples):")
        print(f"    Average score:  {mean_overall:.1f}%")
        for tr in test_results:
            cap_str = "+".join(tr["capabilities"])
            print(f"    {tr['id']:<12} score={tr['mean_score']:.2f}  caps={cap_str}")

    print(f"\n{'='*80}")
    print("TEST MODE COMPLETE")
    print(f"{'='*80}")
    print(f"✅ Everything OK? Run without --test_mode for full evaluation.")
    print(f"❌ No files found? Check --model_dir and INFER_BASE_DIR path.")


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="MM-Vet Utility Evaluation (GPT-4 / Local LLM scoring)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
MM-Vet (Yu et al., ICML 2024) evaluates multimodal model UTILITY.

Examples:
  # Test pipeline with local model (FREE):
  python eval_mmvet.py --finding1 --model_dir llava_1_5_7b \\
      --mmvet_path ./mm-vet --use_local_judge --load_in_4bit --test_mode

  # Full eval with GPT-4 (faithful to paper):
  python eval_mmvet.py --finding1 --model_dir llava_1_5_7b --mmvet_path ./mm-vet

  # Full eval with GPT-4o (cheaper):
  python eval_mmvet.py --finding1 --mmvet_path ./mm-vet --gpt_model gpt-4o --num_runs 3
        """,
    )
    parser.add_argument("--finding1", action="store_true")
    parser.add_argument("--finding2", action="store_true")
    parser.add_argument("--finding3", action="store_true")
    parser.add_argument("--finding4", action="store_true")

    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--mmvet_path", type=str, required=True,
                        help="Path to MM-Vet dataset dir (contains mm-vet.json)")
    parser.add_argument("--gpt_model", type=str, default="gpt-4-0613",
                        help="GPT model for scoring (default: gpt-4-0613)")
    parser.add_argument("--num_runs", type=int, default=5,
                        help="Number of scoring runs to average (default: 5)")

    parser.add_argument("--use_local_judge", action="store_true",
                        help="Use local Llama-3-8B-Instruct instead of GPT-4")
    parser.add_argument("--local_model", type=str,
                        default="meta-llama/Meta-Llama-3-8B-Instruct",
                        help="Local model for scoring")
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="Load local model in 4-bit (saves VRAM)")
    parser.add_argument("--load_in_8bit", action="store_true",
                        help="Load local model in 8-bit")

    parser.add_argument("--skip_neutral", action="store_true")
    parser.add_argument("--test_mode", action="store_true",
                        help="Score 5 samples with detailed output")

    args = parser.parse_args()

    findings = {
        "mmvet_finding1": args.finding1,
        "mmvet_finding2": args.finding2,
        "mmvet_finding3": args.finding3,
        "mmvet_finding4": args.finding4,
    }
    if not any(findings.values()):
        parser.error("Specify at least one: --finding1, --finding2, --finding3, --finding4")

    eval_subdir = args.model_dir if args.model_dir else "all_models"

    print(f"\n{'='*80}")
    print("MM-VET UTILITY EVALUATION")
    print(f"{'='*80}")
    evaluator_type = "Local LLM" if args.use_local_judge else f"GPT-4 ({args.gpt_model})"
    print(f"Evaluator: {evaluator_type}")
    num_runs = args.num_runs if not args.use_local_judge else min(args.num_runs, 1)
    print(f"Scoring runs: {num_runs}")
    print(f"MM-Vet GT: {args.mmvet_path}")
    print(f"Input: {INFER_BASE_DIR}" + (f"/{args.model_dir}" if args.model_dir else " (all)"))
    print(f"Output: {EVAL_BASE_DIR}/{eval_subdir}")

    gt_data = load_mmvet_gt(args.mmvet_path)
    if gt_data is None:
        return

    if args.use_local_judge:
        evaluator = LocalLLMEvaluator(
            model_name=args.local_model,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            num_runs=num_runs,
        )
    else:
        evaluator = GPT4Evaluator(gpt_model=args.gpt_model, num_runs=num_runs)

    if args.test_mode:
        first_finding = next(name for name, sel in findings.items() if sel)
        run_test_mode(evaluator, gt_data, first_finding, model_dir=args.model_dir)
        return

    for finding_name, selected in findings.items():
        if not selected:
            continue
        output_dir = os.path.join(EVAL_BASE_DIR, eval_subdir, finding_name)
        run_finding_evaluation(
            finding_name=finding_name, evaluator=evaluator, gt_data=gt_data,
            output_dir=output_dir, skip_neutral=args.skip_neutral, model_dir=args.model_dir,
        )

    print(f"\n{'='*80}")
    print("✅ ALL MM-VET EVALUATIONS COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
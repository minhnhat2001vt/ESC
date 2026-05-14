"""
MathVista Data Preparation Script — BASELINE ONLY (testmini)

MathVista evaluates mathematical reasoning in visual contexts.
testmini has 1,000 examples total (English + Chinese).

Dataset structure:
    original_data/mathvista/
        mathvista_testmini.json      ← testmini split (1,000 examples)
        images/
            testmini/                ← images for testmini
                1.png, 2.png, ...

JSON format (per sample):
    {
        "pid": "3",
        "question": "...",
        "image": "images/testmini/3.png",
        "choices": ["135°", "140°", "145°", "150°"],   ← may be null for free_form
        "unit": null,
        "precision": null,
        "answer": "145°",
        "question_type": "multi_choice" | "free_form",
        "answer_type": "text" | "integer" | "float" | "list",
        "metadata": {
            "category": "math-targeted-vqa",
            "context": "geometry diagram",
            "grade": "high school",
            "img_height": 60, "img_width": 80,
            "language": "english" | "chinese",
            "skills": ["numeric commonsense", ...],
            "source": "TextVQA",
            "split": "testmini",
            "task": "visual question answering"
        },
        "query": "Hint: Please answer the question ..."
    }

The `query` field already contains the full prompt with hint, question, and choices.
We use it directly for inference.

Question format for inference:
    <image>
    <query field from dataset>

Output (English only — default):
    processed_data/mathvista_baseline_english/
        mathvista_full_baseline.json
        mathvista_baseline_summary.json

Output (English + Chinese — with --chinese flag):
    processed_data/mathvista_baseline_full/
        mathvista_full_baseline.json
        mathvista_baseline_summary.json

Usage:
    # English only (~900 samples)
    python prepare_mathvista.py

    # Full 1,000 samples (English + Chinese)
    python prepare_mathvista.py --chinese

    # Custom data dir
    python prepare_mathvista.py --data_dir /path/to/mathvista
    python prepare_mathvista.py --data_dir /path/to/mathvista --chinese

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
from collections import Counter


# ============================================================================
# CONSTANT PATHS
# ============================================================================
DATA_DIR        = "/workspace/original_data/mathvista"
OUTPUT_BASE_DIR = "/workspace/processed_data"


# ============================================================================
# DATA LOADING
# ============================================================================
def load_mathvista(json_path):
    """Load MathVista JSON file."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples from {os.path.basename(json_path)}")
    return data


def filter_english(data):
    """Filter to English-only samples."""
    english = [
        d for d in data
        if d.get("metadata", {}).get("language", "").lower() == "english"
    ]
    print(f"Filtered to {len(english)} English samples (from {len(data)} total)")
    return english


# ============================================================================
# CONVERSION
# ============================================================================
def convert_sample(entry):
    """
    Convert one MathVista entry to inference format.

    Uses the `query` field directly as the prompt (it already includes
    the hint, question, and choices formatted by the MathVista authors).
    """
    pid = str(entry["pid"])
    question = entry.get("question", "")
    image_path = entry.get("image", "")
    choices = entry.get("choices", None)
    answer = str(entry.get("answer", ""))
    question_type = entry.get("question_type", "")
    answer_type = entry.get("answer_type", "")
    query = entry.get("query", "")
    unit = entry.get("unit", None)
    precision = entry.get("precision", None)
    metadata = entry.get("metadata", {})

    # Use the query field as the formatted question (MathVista's own prompt)
    formatted_q = query if query else question

    # Build user message
    user_message = f"<image>\n{formatted_q}"

    sample_id = f"mathvista_{pid.zfill(4)}"

    # Parse choices into options dict for multi_choice
    options = {}
    options_raw = ""
    if choices and isinstance(choices, list) and len(choices) > 0:
        letters = [chr(ord("A") + i) for i in range(len(choices))]
        options = {l: c for l, c in zip(letters, choices)}
        options_raw = " ".join(f"({l}) {c}" for l, c in zip(letters, choices))

    # For multi_choice, find the GT letter
    gt_letter = ""
    if question_type == "multi_choice" and choices:
        for i, c in enumerate(choices):
            if str(c).strip() == str(answer).strip():
                gt_letter = chr(ord("A") + i)
                break

    return {
        "id": sample_id,
        "image": [image_path],
        "conversations": [
            {"from": "user", "value": user_message}
        ],
        "metadata": {
            "scenario":            "mathvista",
            "image_type":          metadata.get("context", "unknown"),
            "data_source":         "json",
            "question_id":         pid,
            "original_question":   question,
            "formatted_question":  formatted_q,
            "question_type":       question_type,
            "answer_type":         answer_type,
            "options":             options,
            "options_raw":         options_raw,
            "choices":             choices or [],
            "gt_answer":           answer,
            "gt_answer_letter":    gt_letter,
            "unit":                unit,
            "precision":           precision,
            "image_filename":      image_path,
            "query":               query,
            # MathVista-specific metadata
            "category":            metadata.get("category", ""),
            "context":             metadata.get("context", ""),
            "grade":               metadata.get("grade", ""),
            "language":            metadata.get("language", ""),
            "skills":              metadata.get("skills", []),
            "source":              metadata.get("source", ""),
            "split":               metadata.get("split", ""),
            "task":                metadata.get("task", ""),
            # Emotion fields (for compatibility)
            "emotion_category":    "neutral",
            "emotion_prompt_name": "",
            "emotion_prompt_text": "",
            "finding":             "baseline",
        },
    }


# ============================================================================
# SAVE / STATS
# ============================================================================
def save_dataset(samples, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    print(f"  ✅ Saved {len(samples)} samples → {os.path.basename(output_path)}")


def make_stats(samples):
    qtype_dist = Counter(s["metadata"]["question_type"] for s in samples)
    atype_dist = Counter(s["metadata"]["answer_type"] for s in samples)
    task_dist = Counter(s["metadata"]["task"] for s in samples)
    category_dist = Counter(s["metadata"]["category"] for s in samples)
    grade_dist = Counter(s["metadata"]["grade"] for s in samples)
    context_dist = Counter(s["metadata"]["context"] for s in samples)
    lang_dist = Counter(s["metadata"]["language"] for s in samples)

    return {
        "total": len(samples),
        "language_distribution": dict(lang_dist),
        "question_type_distribution": dict(qtype_dist),
        "answer_type_distribution": dict(atype_dist),
        "task_distribution": dict(task_dist),
        "category_distribution": dict(category_dist),
        "grade_distribution": dict(grade_dist),
        "context_distribution": dict(context_dist),
    }


# ============================================================================
# PREPARE
# ============================================================================
def prepare_baseline(output_dir, data_dir=DATA_DIR, include_chinese=False):
    lang_label = "English + Chinese" if include_chinese else "English only"

    print(f"\n{'='*80}")
    print(f"MATHVISTA BASELINE PREPARATION (testmini, {lang_label})")
    print(f"{'='*80}")

    os.makedirs(output_dir, exist_ok=True)

    # Load testmini
    json_path = os.path.join(data_dir, "mathvista_testmini.json")
    data = load_mathvista(json_path)

    # Filter language
    if include_chinese:
        print(f"Language filter: DISABLED → keeping all {len(data)} samples (English + Chinese)")
    else:
        data = filter_english(data)

    # Verify some images
    image_dir = data_dir
    missing = 0
    for entry in data[:10]:
        img_path = os.path.join(image_dir, entry.get("image", ""))
        if not os.path.exists(img_path):
            missing += 1
    if missing > 0:
        print(f"  ⚠️  {missing}/10 sample images not found in {image_dir}")
    else:
        print(f"  ✅ Sample images verified")

    # Convert
    samples = [convert_sample(e) for e in data]

    # Save
    output_path = os.path.join(output_dir, "mathvista_full_baseline.json")
    save_dataset(samples, output_path)

    # Stats
    stats = make_stats(samples)
    summary = {
        "dataset":      "MathVista",
        "reference":    "MathVista: Evaluating Mathematical Reasoning (ICLR 2024)",
        "mode":         f"baseline (testmini, {lang_label}, image + query)",
        "split":        "testmini",
        "language":     "english+chinese" if include_chinese else "english",
        "data_dir":     data_dir,
        "output_file":  "mathvista_full_baseline.json",
        **stats,
    }
    summary_path = os.path.join(output_dir, "mathvista_baseline_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n  Stats:")
    print(f"    Total samples:    {stats['total']}")
    print(f"    Languages:        {stats['language_distribution']}")
    print(f"    Question types:   {stats['question_type_distribution']}")
    print(f"    Answer types:     {stats['answer_type_distribution']}")
    print(f"    Tasks:            {stats['task_distribution']}")

    # Preview
    print(f"\n  Sample preview:")
    for s in samples[:3]:
        m = s["metadata"]
        print(f"    [{s['id']}] Q: {m['original_question'][:60]}...")
        print(f"       Type: {m['question_type']}/{m['answer_type']}")
        print(f"       Lang: {m['language']}  GT: {m['gt_answer']}  Image: {s['image'][0]}")

    return samples


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Prepare MathVista testmini baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
MathVista testmini: 1,000 examples total (English + Chinese).
Two question types: multi_choice and free_form.
Four answer types: text, integer, float, list.

Structure:
  original_data/mathvista/
      mathvista_testmini.json
      images/testmini/

Output (default — English only):
  processed_data/mathvista_baseline_english/
      mathvista_full_baseline.json
      mathvista_baseline_summary.json

Output (--chinese — full 1,000 samples):
  processed_data/mathvista_baseline_full/
      mathvista_full_baseline.json
      mathvista_baseline_summary.json

Examples:
  python prepare_mathvista.py                                  # English only
  python prepare_mathvista.py --chinese                        # Full 1,000 samples
  python prepare_mathvista.py --data_dir /path/to/mathvista    # Custom path
        """,
    )
    parser.add_argument("--data_dir",  type=str, default=DATA_DIR,
                        help="Path to mathvista data directory")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--chinese", action="store_true",
                        help="Include Chinese samples → full 1,000 samples "
                             "(output: mathvista_baseline_full). "
                             "Default (no flag): English only "
                             "(output: mathvista_baseline_english)")
    args = parser.parse_args()

    # ── Output folder name encodes language choice clearly ──────────────────
    if args.output_dir:
        output_dir = args.output_dir
    elif args.chinese:
        output_dir = os.path.join(OUTPUT_BASE_DIR, "mathvista_baseline_full")
    else:
        output_dir = os.path.join(OUTPUT_BASE_DIR, "mathvista_baseline_english")

    print(f"\n{'='*80}")
    print("MATHVISTA DATA PREPARATION — BASELINE")
    print(f"{'='*80}")
    print(f"Data dir:        {args.data_dir}")
    print(f"Output dir:      {output_dir}")
    print(f"Include Chinese: {args.chinese}")

    prepare_baseline(output_dir, data_dir=args.data_dir, include_chinese=args.chinese)

    print(f"\n{'='*80}")
    print("✅ MATHVISTA PREPARATION COMPLETE")
    print(f"{'='*80}")
    print(f"Output: {output_dir}")
    print(f"\nNext: add to inference.py:")
    print(f'  IMAGE_BASE_DIRS["mathvista_baseline"] = "{args.data_dir}"')
    if args.chinese:
        print(f'  BENCHMARK_FINDINGS["mathvista"] = ["mathvista_baseline_full"]')
    else:
        print(f'  BENCHMARK_FINDINGS["mathvista"] = ["mathvista_baseline_english"]')


if __name__ == "__main__":
    main()
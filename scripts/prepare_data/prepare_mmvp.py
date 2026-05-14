"""
MMVP (Multimodal Visual Patterns) Data Preparation Script — BASELINE ONLY

MMVP benchmark evaluates visual pattern understanding in VLMs.
Each question is paired with TWO images (odd/even index pairs) that test the same
visual concept but expect different answers.

Dataset structure:
    original_data/mmvp/
        MMVP Images/         ← images (1.jpg, 2.jpg, ..., 300.jpg)
        Questions.csv        ← CSV with columns: Index, Question, Options, Correct Answer

Questions.csv format:
    Index,Question,Options,Correct Answer
    1,Are the butterfly's wings closer to being open or closed?,(a) Open (b) Closed,(a)
    2,Are the butterfly's wings closer to being open or closed?,(a) Open (b) Closed,(b)
    ...

Each pair (odd, even) shares the same question and options but has different correct
answers and different images. This tests whether the model actually looks at the image
rather than relying on language priors.

Question format for inference:
    <image>
    <question>
    (a) <option_a>
    (b) <option_b>
    Answer with the option's letter from the given choices directly.

Output:
    processed_data/mmvp_baseline/
        mmvp_full_baseline.json       (all 300 samples)
        mmvp_baseline_summary.json

Usage:
    python prepare_mmvp.py
    python prepare_mmvp.py --output_dir /custom/path
    python prepare_mmvp.py --data_dir /path/to/mmvp

Reference:
    Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs (CVPR 2024)
    https://github.com/tsb0601/MMVP
"""

import json
import os
import csv
import re
import argparse
from pathlib import Path


# ============================================================================
# CONSTANT PATHS
# ============================================================================
DATA_DIR        = "/workspace/original_data/mmvp"
IMAGE_DIR       = os.path.join(DATA_DIR, "MMVP Images")
QUESTIONS_CSV   = os.path.join(DATA_DIR, "Questions.csv")
OUTPUT_BASE_DIR = "/workspace/processed_data"

ANSWER_INSTRUCTION = "Answer with the option's letter from the given choices directly."


# ============================================================================
# DATA LOADING
# ============================================================================
def load_questions(csv_path=QUESTIONS_CSV):
    """
    Load Questions.csv.

    Format: Index,Question,Options,Correct Answer
        1,Are the butterfly's wings closer to being open or closed?,(a) Open (b) Closed,(a)

    Returns: list of dicts with parsed fields
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Questions.csv not found: {csv_path}")

    data = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["Index"].strip())
            question = row["Question"].strip()
            options_raw = row["Options"].strip()
            correct_raw = row["Correct Answer"].strip()

            # Parse options: "(a) Open (b) Closed" → {"a": "Open", "b": "Closed"}
            options = parse_options(options_raw)

            # Parse correct answer: "(a)" → "a", "(b)" → "b"
            correct = parse_correct_answer(correct_raw)

            data.append({
                "index": idx,
                "question": question,
                "options_raw": options_raw,
                "options": options,
                "correct_answer": correct,
                "correct_answer_text": options.get(correct, ""),
                "image_filename": f"{idx}.jpg",
            })

    print(f"Loaded {len(data)} samples from Questions.csv")
    return data


def parse_options(options_str):
    """
    Parse options string into dict.

    Examples:
        "(a) Open (b) Closed"
            → {"a": "Open", "b": "Closed"}
        "(a) Towards the camera (b) Away from the camera"
            → {"a": "Towards the camera", "b": "Away from the camera"}
        "(a) Single (b) Multiple"
            → {"a": "Single", "b": "Multiple"}
    """
    # Pattern: (letter) text
    pattern = r'\(([a-z])\)\s*'
    parts = re.split(pattern, options_str)

    # parts will be: ['', 'a', 'Open ', 'b', 'Closed', ...]
    options = {}
    i = 1  # skip first empty element
    while i < len(parts) - 1:
        letter = parts[i].strip()
        text = parts[i + 1].strip()
        options[letter] = text
        i += 2

    return options


def parse_correct_answer(answer_str):
    """
    Parse correct answer string.

    "(a)" → "a"
    "(b)" → "b"
    "a"   → "a"
    """
    match = re.search(r'\(([a-z])\)', answer_str)
    if match:
        return match.group(1)
    # Fallback: just take the letter
    cleaned = answer_str.strip().lower().replace("(", "").replace(")", "")
    return cleaned


# ============================================================================
# QUESTION FORMATTING
# ============================================================================
def format_question(question, options):
    """
    Format question with options for inference.

    Output:
        <question>
        (a) <option_a>
        (b) <option_b>
        Answer with the option's letter from the given choices directly.
    """
    lines = [question]
    for letter in sorted(options.keys()):
        lines.append(f"({letter}) {options[letter]}")
    lines.append(ANSWER_INSTRUCTION)
    return "\n".join(lines)


# ============================================================================
# CONVERSION: MMVP entry → inference format
# ============================================================================
def convert_sample(entry):
    """
    Convert one MMVP entry to inference format.

    Output format (mirrors RWQA / figstep / pope baseline):
    {
        "id": "mmvp_0001",
        "image": ["/MMVP Images/1.jpg"],
        "conversations": [
            {"from": "user", "value": "<image>\n<formatted_question>"}
        ],
        "metadata": { ... }
    }
    """
    idx = entry["index"]
    question = entry["question"]
    options = entry["options"]
    correct = entry["correct_answer"]
    correct_text = entry["correct_answer_text"]
    image_filename = entry["image_filename"]

    formatted_q = format_question(question, options)
    image_path = f"/MMVP Images/{image_filename}"
    user_message = f"<image>\n{formatted_q}"
    sample_id = f"mmvp_{str(idx).zfill(4)}"

    # Determine pair info (odd/even pairs share same question)
    pair_id = (idx - 1) // 2  # 1,2 → 0; 3,4 → 1; etc.
    is_first_in_pair = (idx % 2 == 1)

    return {
        "id": sample_id,
        "image": [image_path],
        "conversations": [
            {"from": "user", "value": user_message}
        ],
        "metadata": {
            "scenario":            "mmvp",
            "image_type":          "visual_pattern",
            "data_source":         "csv",
            "question_id":         idx,
            "original_question":   question,
            "formatted_question":  formatted_q,
            "question_type":       "multiple_choice",
            "options":             options,
            "options_raw":         entry["options_raw"],
            "gt_answer":           correct,
            "gt_answer_text":      correct_text,
            "image_filename":      image_filename,
            "pair_id":             pair_id,
            "is_first_in_pair":    is_first_in_pair,
            "emotion_category":    "neutral",
            "emotion_prompt_name": "",
            "emotion_prompt_text": "",
            "finding":             "baseline",
        },
    }


# ============================================================================
# SAVE HELPER
# ============================================================================
def save_dataset(samples, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    print(f"  ✅ Saved {len(samples)} samples → {os.path.basename(output_path)}")


def make_stats(samples):
    """Compute statistics."""
    answer_dist = {}
    for s in samples:
        m = s["metadata"]
        ans = m.get("gt_answer", "")
        answer_dist[ans] = answer_dist.get(ans, 0) + 1

    # Pair analysis
    n_pairs = max(s["metadata"]["pair_id"] for s in samples) + 1 if samples else 0

    return {
        "total": len(samples),
        "num_pairs": n_pairs,
        "answer_distribution": answer_dist,
    }


# ============================================================================
# PREPARE
# ============================================================================
def prepare_baseline(output_dir, data_dir=DATA_DIR):
    """Prepare MMVP baseline dataset."""
    print(f"\n{'='*80}")
    print("MMVP BASELINE PREPARATION")
    print(f"{'='*80}")

    os.makedirs(output_dir, exist_ok=True)

    # Load questions
    csv_path = os.path.join(data_dir, "Questions.csv")
    data = load_questions(csv_path)

    # Verify images exist
    image_dir = os.path.join(data_dir, "MMVP Images")
    if not os.path.exists(image_dir):
        print(f"  ⚠️  Image dir not found: {image_dir}")
        print(f"      Trying alternate: {data_dir}/images/")
        image_dir_alt = os.path.join(data_dir, "images")
        if os.path.exists(image_dir_alt):
            image_dir = image_dir_alt
            print(f"      Found: {image_dir_alt}")

    # Check a few images
    missing = 0
    for entry in data[:10]:
        img_path = os.path.join(image_dir, entry["image_filename"])
        if not os.path.exists(img_path):
            missing += 1
    if missing > 0:
        print(f"  ⚠️  {missing}/10 sample images not found in {image_dir}")
    else:
        print(f"  ✅ Sample images verified in {image_dir}")

    # Convert samples
    samples = [convert_sample(e) for e in data]

    # Save
    output_path = os.path.join(output_dir, "mmvp_full_baseline.json")
    save_dataset(samples, output_path)

    # Stats
    stats = make_stats(samples)
    summary = {
        "dataset":      "MMVP",
        "reference":    "Eyes Wide Shut? CVPR 2024",
        "mode":         "baseline (image + question + options, no emotion)",
        "image_dir":    image_dir,
        "source_file":  csv_path,
        "output_file":  "mmvp_full_baseline.json",
        **stats,
    }
    summary_path = os.path.join(output_dir, "mmvp_baseline_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n  Stats:")
    print(f"    Total samples:        {stats['total']}")
    print(f"    Pairs:                {stats['num_pairs']}")
    print(f"    Answer distribution:  {stats['answer_distribution']}")

    # Show first few samples for verification
    print(f"\n  Sample preview:")
    for s in samples[:3]:
        m = s["metadata"]
        print(f"    [{s['id']}] Q: {m['original_question'][:60]}...")
        print(f"             Options: {m['options_raw']}")
        print(f"             GT: ({m['gt_answer']}) {m['gt_answer_text']}")
        print(f"             Image: {s['image'][0]}")

    return samples


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Prepare MMVP benchmark — baseline (image + question + options)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
MMVP (Multimodal Visual Patterns) tests visual pattern understanding.
300 samples in 150 pairs. Each pair has the same question but different
correct answers for different images.

Structure:
  original_data/mmvp/
      MMVP Images/       ← 1.jpg to 300.jpg
      Questions.csv      ← Index, Question, Options, Correct Answer

Output:
  processed_data/mmvp_baseline/
      mmvp_full_baseline.json
      mmvp_baseline_summary.json

Examples:
  python prepare_mmvp.py
  python prepare_mmvp.py --data_dir /path/to/mmvp
  python prepare_mmvp.py --output_dir /custom/output
        """,
    )

    parser.add_argument("--data_dir", type=str, default=DATA_DIR,
                        help=f"MMVP data directory (default: {DATA_DIR})")
    parser.add_argument("--output_dir", type=str, default="processed_data/mmvp_baseline",
                        help="Output directory (default: processed_data/mmvp_baseline)")

    args = parser.parse_args()
    output_dir = args.output_dir or os.path.join(OUTPUT_BASE_DIR, "mmvp_baseline")

    print(f"\n{'='*80}")
    print("MMVP DATA PREPARATION — BASELINE")
    print(f"{'='*80}")
    print(f"Data dir:   {args.data_dir}")
    print(f"CSV:        {os.path.join(args.data_dir, 'Questions.csv')}")
    print(f"Images:     {os.path.join(args.data_dir, 'MMVP Images')}")
    print(f"Output dir: {output_dir}")

    prepare_baseline(output_dir, data_dir=args.data_dir)

    print(f"\n{'='*80}")
    print("✅ MMVP PREPARATION COMPLETE")
    print(f"{'='*80}")
    print(f"Output: {output_dir}")
    print(f"\nNext: add to inference.py:")
    print(f'  IMAGE_BASE_DIRS["mmvp_baseline"] = "{os.path.join(args.data_dir)}"')
    print(f'  BENCHMARK_FINDINGS["mmvp"] = ["mmvp_baseline"]')


if __name__ == "__main__":
    main()
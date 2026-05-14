"""
MMStar (Multi-Modal Star) Data Preparation Script — BASELINE ONLY

MMStar evaluates multi-modal capabilities across 6 core competencies and 18 detailed axes.
It uses a 2-level categorization: coarse category (e.g. "coarse perception") and
fine-grained l2_category (e.g. "image scene and topic").

Dataset structure:
    original_data/mmstar/
        metadata.csv          ← CSV with all samples
        images/               ← images (00000.jpg, 00001.jpg, ...)

metadata.csv columns:
    index,question,answer,category,l2_category,meta_info,local_image

    index:        0, 1, 2, ...
    question:     "<image 1> Which of the following is the most ..." or plain text
                  May start with "<image 1>" tag; options embedded as "Options: A: ... B: ..."
    answer:       "A", "B", "C", "D"
    category:     "coarse perception", "fine-grained perception", "instance reasoning",
                  "logical reasoning", "science and technology", "math"
    l2_category:  "image scene and topic", "image emotion", "image style and quality", etc.
    meta_info:    JSON string with source/split/image_path metadata
    local_image:  "images/00000.jpg"

Question format — already contains choices, e.g.:
    "<image 1> Which of the following is the most prominent feature ..."
    "Options: A: The skyline, B: The golf course, C: The trees, D: The person"

Question format for inference:
    <image>
    <question text with options>
    Answer with the option's letter from the given choices directly.

Output:
    processed_data/mmstar_baseline/
        mmstar_full_baseline.json
        mmstar_baseline_summary.json

Usage:
    python prepare_mmstar.py
    python prepare_mmstar.py --data_dir /path/to/mmstar
    python prepare_mmstar.py --output_dir /custom/output
    python prepare_mmstar.py --split val

Reference:
    MMStar: Are We on the Right Way for Evaluating Large Vision-Language Models?
    https://github.com/MMStar-Benchmark/MMStar
"""

import json
import os
import csv
import re
import argparse
from pathlib import Path
from collections import Counter


# ============================================================================
# CONSTANT PATHS
# ============================================================================
DATA_DIR        = "/workspace/original_data/mmstar"
OUTPUT_BASE_DIR = "/workspace/processed_data"

ANSWER_INSTRUCTION = "Answer with the option's letter from the given choices directly."

# MMStar 6 coarse categories
MMSTAR_CATEGORIES = [
    "coarse perception",
    "fine-grained perception",
    "instance reasoning",
    "logical reasoning",
    "science and technology",
    "math",
]


# ============================================================================
# DATA LOADING
# ============================================================================
def load_metadata(csv_path):
    """
    Load metadata.csv.

    Format:
        index,question,answer,category,l2_category,meta_info,local_image
        0,"<image 1> Which of the following...",C,coarse perception,image scene and topic,"{...}",images/00000.jpg

    Note: question may contain embedded <image 1> tag and Options: A: ... B: ... etc.

    Returns: list of dicts with parsed fields
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"metadata.csv not found: {csv_path}")

    data = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["index"].strip())
            question = row["question"].strip()
            answer = row["answer"].strip().upper()
            category = row["category"].strip()
            l2_category = row["l2_category"].strip()
            meta_info_raw = row.get("meta_info", "").strip()
            local_image = row.get("local_image", "").strip()

            # Parse meta_info JSON
            meta_info = {}
            if meta_info_raw:
                try:
                    meta_info = json.loads(meta_info_raw)
                except (json.JSONDecodeError, TypeError):
                    meta_info = {"raw": meta_info_raw}

            # Parse options from question text
            options = parse_options_from_question(question)

            data.append({
                "index": idx,
                "question": question,
                "answer": answer,
                "category": category,
                "l2_category": l2_category,
                "meta_info": meta_info,
                "local_image": local_image,
                "options": options,
                "source": meta_info.get("source", ""),
                "split": meta_info.get("split", ""),
            })

    print(f"Loaded {len(data)} samples from {os.path.basename(csv_path)}")
    return data


def parse_options_from_question(question_text: str) -> dict:
    """
    Parse option letters and texts from question.

    MMStar questions embed options like:
        "Options: A: The skyline, B: The golf course, C: The trees, D: The person"
    or:
        "Options: A: The skyline  B: The golf course  C: The trees  D: The person"

    Returns: dict like {"A": "The skyline", "B": "The golf course", ...}
    """
    options = {}

    # Try "A: text, B: text" or "A: text B: text" pattern
    # Look after "Options:" if present
    text = question_text
    opt_idx = text.lower().find("options:")
    if opt_idx >= 0:
        text = text[opt_idx + len("options:"):]

    # Match "A: text" patterns — handle both comma and next-letter as delimiters
    matches = re.findall(r'([A-D])\s*[:.]\s*(.+?)(?=\s+[A-D]\s*[:.]\s|$)', text, re.DOTALL)
    for letter, val in matches:
        options[letter.upper()] = val.strip().rstrip(",").strip()

    return options


def clean_question_text(question: str) -> str:
    """
    Clean question text for inference prompt.

    - Remove leading "<image 1>" tag (we add our own <image> tag)
    - Keep the rest including Options
    """
    # Remove <image N> tags at the start
    cleaned = re.sub(r'^<image\s*\d*>\s*', '', question.strip())
    return cleaned.strip()


# ============================================================================
# CONVERSION: MMStar entry → inference format
# ============================================================================
def convert_sample(entry):
    """
    Convert one MMStar entry to inference format.

    Output format (mirrors pope / mmvp baseline pattern):
    {
        "id": "mmstar_00000",
        "image": ["images/00000.jpg"],
        "conversations": [
            {"from": "user", "value": "<image>\\n<question with options>\\n<instruction>"}
        ],
        "metadata": { ... }
    }
    """
    idx = entry["index"]
    question = entry["question"]
    answer = entry["answer"]
    category = entry["category"]
    l2_category = entry["l2_category"]
    local_image = entry["local_image"]
    options = entry["options"]

    # Clean question (remove <image 1> tag) and append instruction
    cleaned_q = clean_question_text(question)
    formatted_q = f"{cleaned_q}\n{ANSWER_INSTRUCTION}"

    # Build user message
    user_message = f"<image>\n{formatted_q}"

    # Image path — keep relative path as-is (images/00000.jpg)
    image_path = local_image if local_image else f"images/{str(idx).zfill(5)}.jpg"

    sample_id = f"mmstar_{str(idx).zfill(5)}"

    # Determine gt_answer_text from parsed options
    gt_answer_text = options.get(answer, "")

    return {
        "id": sample_id,
        "image": [image_path],
        "conversations": [
            {"from": "user", "value": user_message}
        ],
        "metadata": {
            "scenario":            "mmstar",
            "image_type":          l2_category or "multi_modal",
            "data_source":         "csv",
            "question_id":         idx,
            "original_question":   question,
            "formatted_question":  formatted_q,
            "question_type":       "multiple_choice",
            "options":             options,
            "gt_answer":           answer,
            "gt_answer_text":      gt_answer_text,
            "category":            category,
            "l2_category":         l2_category,
            "source":              entry.get("source", ""),
            "split":               entry.get("split", ""),
            "image_filename":      local_image,
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


def _make_stats(samples):
    """Compute statistics for summary."""
    answer_dist = Counter(s["metadata"]["gt_answer"] for s in samples)
    category_dist = Counter(s["metadata"]["category"] for s in samples)
    l2_dist = Counter(s["metadata"]["l2_category"] for s in samples)
    source_dist = Counter(s["metadata"]["source"] for s in samples if s["metadata"]["source"])

    return {
        "total":                 len(samples),
        "answer_distribution":   dict(sorted(answer_dist.items())),
        "category_distribution": dict(sorted(category_dist.items())),
        "l2_category_distribution": dict(sorted(l2_dist.items())),
        "source_distribution":   dict(sorted(source_dist.items())),
    }


# ============================================================================
# PREPARE
# ============================================================================
def prepare_full(data, output_dir, data_dir):
    """All samples from metadata.csv."""
    print(f"\n{'='*80}")
    print(f"MMSTAR BASELINE — Full ({len(data)} samples)")
    print(f"{'='*80}")

    os.makedirs(output_dir, exist_ok=True)

    # Verify images
    image_dir = os.path.join(data_dir, "images")
    if os.path.exists(image_dir):
        missing = 0
        for entry in data[:10]:
            img_path = os.path.join(data_dir, entry["local_image"])
            if not os.path.exists(img_path):
                missing += 1
        if missing > 0:
            print(f"  ⚠️  {missing}/10 sample images not found")
        else:
            print(f"  ✅ Sample images verified in {image_dir}")
    else:
        print(f"  ⚠️  Image dir not found: {image_dir}")

    samples = [convert_sample(e) for e in data]

    output_path = os.path.join(output_dir, "mmstar_full_baseline.json")
    save_dataset(samples, output_path)

    stats = _make_stats(samples)
    summary = {
        "dataset":      "MMStar",
        "reference":    "MMStar: Are We on the Right Way for Evaluating Large Vision-Language Models?",
        "mode":         "baseline (image + question + options, no emotion)",
        "data_dir":     data_dir,
        "source_file":  "metadata.csv",
        "output_file":  "mmstar_full_baseline.json",
        **stats,
    }
    summary_path = os.path.join(output_dir, "mmstar_baseline_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n  Stats:")
    print(f"    Total samples:       {stats['total']}")
    print(f"    Answer distribution: {stats['answer_distribution']}")
    print(f"    Categories:          {stats['category_distribution']}")

    # Preview
    print(f"\n  Sample preview:")
    for s in samples[:3]:
        m = s["metadata"]
        q_preview = m["original_question"][:80].replace("\n", " ")
        print(f"    [{s['id']}] Q: {q_preview}...")
        print(f"       GT: {m['gt_answer']}  Cat: {m['category']}  Image: {s['image'][0]}")

    return ["mmstar_full_baseline.json"]


def prepare_by_category(data, output_dir, data_dir):
    """Separate files per coarse category (6 categories)."""
    print(f"\n{'='*80}")
    print("MMSTAR BASELINE — By Category")
    print(f"{'='*80}")

    os.makedirs(output_dir, exist_ok=True)
    created_files = []

    # Group by category
    cat_groups = {}
    for e in data:
        cat = e["category"]
        if cat not in cat_groups:
            cat_groups[cat] = []
        cat_groups[cat].append(e)

    for cat_name in sorted(cat_groups.keys()):
        cat_data = cat_groups[cat_name]
        safe_name = cat_name.replace(" ", "_").replace("-", "_")

        samples = [convert_sample(e) for e in cat_data]

        fname = f"mmstar_{safe_name}_baseline.json"
        output_path = os.path.join(output_dir, fname)
        save_dataset(samples, output_path)

        print(f"  [{cat_name:<30}] {len(samples):>5} samples → {fname}")
        created_files.append(fname)

    return created_files


def prepare_by_split(data, output_dir, data_dir, split_name):
    """Filter to specific split (e.g. 'val') from meta_info."""
    filtered = [e for e in data if e.get("split", "").lower() == split_name.lower()]

    if not filtered:
        print(f"  ⚠️  No samples found with split='{split_name}'")
        print(f"     Available splits: {set(e.get('split','') for e in data)}")
        return []

    print(f"\n{'='*80}")
    print(f"MMSTAR BASELINE — Split: {split_name} ({len(filtered)} samples)")
    print(f"{'='*80}")

    os.makedirs(output_dir, exist_ok=True)

    samples = [convert_sample(e) for e in filtered]

    fname = f"mmstar_{split_name}_baseline.json"
    output_path = os.path.join(output_dir, fname)
    save_dataset(samples, output_path)

    stats = _make_stats(samples)
    summary = {
        "dataset":      "MMStar",
        "subset":       split_name,
        "mode":         "baseline (image + question + options, no emotion)",
        "output_file":  fname,
        **stats,
    }
    summary_path = os.path.join(output_dir, f"mmstar_{split_name}_baseline_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"  Stats: total={stats['total']}, answers={stats['answer_distribution']}")
    return [fname]


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Prepare MMStar benchmark — baseline (image + question + options)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
MMStar: Multi-Modal Star benchmark for evaluating LVLMs.
6 core competencies, 18 detailed axes, multiple-choice (A/B/C/D).

Structure:
  original_data/mmstar/
      metadata.csv       ← index, question, answer, category, l2_category, meta_info, local_image
      images/            ← 00000.jpg, 00001.jpg, ...

Output:
  processed_data/mmstar_baseline/
      mmstar_full_baseline.json
      mmstar_baseline_summary.json

Modes:
  --full       All samples
  --category   Separate files per coarse category (6 files)
  --split val  Filter by split field in meta_info
  --all        full + category

Examples:
  python prepare_mmstar.py --full
  python prepare_mmstar.py --all
  python prepare_mmstar.py --split val
  python prepare_mmstar.py --full --data_dir /path/to/mmstar
        """,
    )

    parser.add_argument("--full", action="store_true",
                        help="All samples from metadata.csv")
    parser.add_argument("--category", action="store_true",
                        help="Separate files per coarse category")
    parser.add_argument("--split", type=str, default=None,
                        help="Filter by split name (e.g. 'val')")
    parser.add_argument("--all", action="store_true",
                        help="Prepare full + category")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR,
                        help=f"MMStar data directory (default: {DATA_DIR})")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: processed_data/mmstar_baseline)")

    args = parser.parse_args()

    if not any([args.full, args.category, args.split, args.all]):
        # Default to --full if nothing specified
        args.full = True

    output_dir = args.output_dir or os.path.join(OUTPUT_BASE_DIR, "mmstar_baseline")

    print(f"\n{'='*80}")
    print("MMSTAR DATA PREPARATION — BASELINE")
    print(f"{'='*80}")
    print(f"Data dir:   {args.data_dir}")
    print(f"CSV:        {os.path.join(args.data_dir, 'metadata.csv')}")
    print(f"Images:     {os.path.join(args.data_dir, 'images')}")
    print(f"Output dir: {output_dir}")

    # Validate
    csv_path = os.path.join(args.data_dir, "metadata.csv")
    if not os.path.exists(csv_path):
        print(f"\n❌ metadata.csv not found: {csv_path}")
        return

    # Load data
    data = load_metadata(csv_path)

    # Quick stats
    categories = Counter(e["category"] for e in data)
    answers = Counter(e["answer"] for e in data)
    splits = Counter(e.get("split", "") for e in data if e.get("split"))
    print(f"  Categories: {dict(categories)}")
    print(f"  Answers:    {dict(answers)}")
    if splits:
        print(f"  Splits:     {dict(splits)}")

    all_created = []

    if args.full or args.all:
        files = prepare_full(data, output_dir, args.data_dir)
        all_created.extend(files)

    if args.category or args.all:
        files = prepare_by_category(data, output_dir, args.data_dir)
        all_created.extend(files)

    if args.split:
        files = prepare_by_split(data, output_dir, args.data_dir, args.split)
        all_created.extend(files)

    print(f"\n{'='*80}")
    print("✅ MMSTAR PREPARATION COMPLETE")
    print(f"{'='*80}")
    print(f"Output dir:    {output_dir}")
    print(f"Files created: {all_created}")
    print(f"\nNext: add to inference.py:")
    print(f'  IMAGE_BASE_DIRS["mmstar_baseline"] = "{args.data_dir}"')
    print(f'  BENCHMARK_FINDINGS["mmstar"] = ["mmstar_baseline"]')


if __name__ == "__main__":
    main()
"""
AI2D Data Preparation Script — BASELINE ONLY (test split)

AI2D evaluates diagram understanding and visual reasoning on scientific diagrams.
Test split has ~3,088 questions over 1,000 images.

Expected dataset structure (HuggingFace: lmms-lab/ai2d or local):
    original_data/ai2d/
        ai2d_test.json           ← test split
        ai2d_images/             ← diagram images
            1.png, 2.png, ...

JSON format (per sample):
    {
        "id": "0",
        "image": "ai2d_images/1234.png",
        "question": "What is the role of the sun in this diagram?",
        "options": ["Energy source", "Heat sink", "Reflector", "None"],
        "answer": "A"       ← ground-truth letter
    }

    Alternatively (some formats use "choices" or integer answer index).
    The script handles both conventions.

Question format for inference:
    <image>
    {question}
    (A) option0  (B) option1  (C) option2  (D) option3
    Answer with the option's letter from the given choices directly.

Output:
    processed_data/ai2d_baseline/
        ai2d_full_baseline.json
        ai2d_baseline_summary.json

Usage:
    python prepare_ai2d.py
    python prepare_ai2d.py --data_dir /path/to/ai2d

Reference:
    A Diagram Is Worth A Dozen Images (Kembhavi et al., 2016)
    https://allenai.org/data/diagrams
"""

import json
import os
import sys
import argparse
from pathlib import Path
from collections import Counter


# ============================================================================
# CONSTANT PATHS
# ============================================================================
_SCRIPT_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _SCRIPT_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from path_config import ORIGINAL_DATA_ROOT, PROCESSED_DATA_ROOT

DATA_DIR = str(ORIGINAL_DATA_ROOT / "ai2d")
OUTPUT_BASE_DIR = str(PROCESSED_DATA_ROOT)


# ============================================================================
# DATA DOWNLOAD + SETUP
# ============================================================================
def download_ai2d(data_dir):
    """
    Download AI2D test split from HuggingFace and set up local directory.

    Source: lmms-lab/ai2d (or equivalent)
    Creates:
        {data_dir}/ai2d_test.json
        {data_dir}/ai2d_images/
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError(
            "Please install the datasets library: pip install datasets"
        )

    print(f"\n  Downloading AI2D from HuggingFace...")
    ds = load_dataset("lmms-lab/ai2d", split="test")
    print(f"  Downloaded {len(ds)} samples")

    os.makedirs(data_dir, exist_ok=True)
    img_dir = os.path.join(data_dir, "ai2d_images")
    os.makedirs(img_dir, exist_ok=True)

    records = []
    for idx, sample in enumerate(ds):
        # Save image
        img_filename = f"{idx}.png"
        img_path = os.path.join(img_dir, img_filename)
        image = sample.get("image")
        if image is not None:
            image.save(img_path)

        # Build record
        record = {
            "id": str(sample.get("id", idx)),
            "image": f"ai2d_images/{img_filename}",
            "question": sample.get("question", ""),
            "options": sample.get("options", sample.get("choices", [])),
            "answer": sample.get("answer", ""),
        }
        records.append(record)

        if (idx + 1) % 500 == 0:
            print(f"    Processed {idx + 1}/{len(ds)} samples...")

    # Save JSON
    json_path = os.path.join(data_dir, "ai2d_test.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"  ✅ Saved {len(records)} samples → {json_path}")
    print(f"  ✅ Images saved → {img_dir}")
    return records


# ============================================================================
# DATA LOADING
# ============================================================================
def load_ai2d(json_path):
    """Load AI2D JSON file."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples from {os.path.basename(json_path)}")
    return data


# ============================================================================
# CONVERSION
# ============================================================================
def _resolve_answer_letter(entry):
    """
    Resolve the ground-truth answer letter from various AI2D format conventions.

    Handles:
      - answer is already a letter string: "A", "B", "C", "D"
      - answer is an integer index: 0 → "A", 1 → "B", etc.
      - answer is the text of the correct option → map to letter
    """
    answer_raw = entry.get("answer", "")
    options = entry.get("options", entry.get("choices", []))

    # Already a letter
    if isinstance(answer_raw, str) and len(answer_raw) == 1 and answer_raw.upper() in "ABCDEFGHIJ":
        return answer_raw.upper()

    # Integer index
    if isinstance(answer_raw, int):
        if 0 <= answer_raw < len(options):
            return chr(ord("A") + answer_raw)

    # Try parsing string-encoded integer
    if isinstance(answer_raw, str):
        try:
            idx = int(answer_raw)
            if 0 <= idx < len(options):
                return chr(ord("A") + idx)
        except ValueError:
            pass

    # Text match against options
    if isinstance(answer_raw, str) and options:
        answer_lower = answer_raw.strip().lower()
        for i, opt in enumerate(options):
            if str(opt).strip().lower() == answer_lower:
                return chr(ord("A") + i)

    return str(answer_raw).upper()


def convert_sample(entry):
    """Convert one AI2D entry to the standardized inference format."""
    sample_id_raw = str(entry.get("id", entry.get("question_id", "")))
    question = entry.get("question", "")
    image_path = entry.get("image", "")
    options = entry.get("options", entry.get("choices", []))
    gt_letter = _resolve_answer_letter(entry)

    # Build formatted question with options
    num_opts = len(options)
    letters = [chr(ord("A") + i) for i in range(num_opts)]
    options_dict = {l: str(o) for l, o in zip(letters, options)}
    options_lines = "\n".join(f"({l}) {o}" for l, o in zip(letters, options))

    formatted_q = (
        f"{question}\n"
        f"{options_lines}\n"
        f"Answer with the option's letter from the given choices directly."
    )

    user_message = f"<image>\n{formatted_q}"
    sample_id = f"ai2d_{sample_id_raw.zfill(5)}"

    # Ground-truth answer text
    gt_idx = ord(gt_letter) - ord("A") if gt_letter and gt_letter.isalpha() else -1
    gt_answer_text = str(options[gt_idx]) if 0 <= gt_idx < len(options) else ""

    return {
        "id": sample_id,
        "image": [image_path],
        "conversations": [
            {"from": "user", "value": user_message}
        ],
        "metadata": {
            "scenario":            "ai2d",
            "image_type":          "diagram",
            "data_source":         "json",
            "question_id":         sample_id_raw,
            "original_question":   question,
            "formatted_question":  formatted_q,
            "question_type":       "multi_choice",
            "answer_type":         "text",
            "options":             options_dict,
            "options_raw":         " ".join(f"({l}) {o}" for l, o in zip(letters, options)),
            "choices":             options,
            "num_choices":         num_opts,
            "gt_answer":           gt_answer_text,
            "gt_answer_letter":    gt_letter,
            "image_filename":      image_path,
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
    num_choices_dist = Counter(s["metadata"]["num_choices"] for s in samples)
    return {
        "total": len(samples),
        "num_choices_distribution": dict(num_choices_dist),
    }


# ============================================================================
# PREPARE
# ============================================================================
def prepare_baseline(output_dir, data_dir=DATA_DIR):
    print(f"\n{'='*80}")
    print(f"AI2D BASELINE PREPARATION (test split)")
    print(f"{'='*80}")

    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(data_dir, "ai2d_test.json")

    # Download if data doesn't exist
    if not os.path.exists(json_path):
        print(f"  Data not found at {json_path}")
        print(f"  Downloading from HuggingFace...")
        download_ai2d(data_dir)

    data = load_ai2d(json_path)

    # Verify some images
    missing = 0
    for entry in data[:10]:
        img_path = os.path.join(data_dir, entry.get("image", ""))
        if not os.path.exists(img_path):
            missing += 1
    if missing > 0:
        print(f"  ⚠️  {missing}/10 sample images not found in {data_dir}")
    else:
        print(f"  ✅ Sample images verified")

    # Convert
    samples = [convert_sample(e) for e in data]

    # Save
    output_path = os.path.join(output_dir, "ai2d_full_baseline.json")
    save_dataset(samples, output_path)

    # Stats
    stats = make_stats(samples)
    summary = {
        "dataset":      "AI2D",
        "reference":    "Kembhavi et al., A Diagram Is Worth A Dozen Images (2016)",
        "mode":         "baseline (test split, image + MCQ)",
        "split":        "test",
        "data_dir":     data_dir,
        "output_file":  "ai2d_full_baseline.json",
        **stats,
    }
    summary_path = os.path.join(output_dir, "ai2d_baseline_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n  Stats:")
    print(f"    Total samples:    {stats['total']}")
    print(f"    Choices dist:     {stats['num_choices_distribution']}")

    # Preview
    print(f"\n  Sample preview:")
    for s in samples[:3]:
        m = s["metadata"]
        print(f"    [{s['id']}] Q: {m['original_question'][:60]}...")
        print(f"       Choices: {m['num_choices']}  GT: {m['gt_answer_letter']}  Image: {s['image'][0]}")

    return samples


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Prepare AI2D test baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
AI2D test: ~3,088 MCQ questions over 1,000 science diagrams.
Variable number of choices per question.

Structure:
  original_data/ai2d/
      ai2d_test.json
      ai2d_images/

Output:
  processed_data/ai2d_baseline/
      ai2d_full_baseline.json
      ai2d_baseline_summary.json

Examples:
  python prepare_ai2d.py
  python prepare_ai2d.py --data_dir /path/to/ai2d
        """,
    )
    parser.add_argument("--data_dir", type=str, default=DATA_DIR,
                        help="Path to AI2D data directory")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(OUTPUT_BASE_DIR, "ai2d_baseline")

    print(f"\n{'='*80}")
    print("AI2D DATA PREPARATION — BASELINE")
    print(f"{'='*80}")
    print(f"Data dir:   {args.data_dir}")
    print(f"Output dir: {output_dir}")

    prepare_baseline(output_dir, data_dir=args.data_dir)

    print(f"\n{'='*80}")
    print("✅ AI2D PREPARATION COMPLETE")
    print(f"{'='*80}")
    print(f"Output: {output_dir}")
    print(f"\nNext: add to inference.py:")
    print(f'  AI2D_DATA_DIR = "{args.data_dir}"')
    print(f'  IMAGE_BASE_DIRS["ai2d_baseline"] = "{args.data_dir}"')


if __name__ == "__main__":
    main()
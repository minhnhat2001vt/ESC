"""
Docstring for scripts.prepare_data.prepare_RWQA
https://huggingface.co/datasets/xai-org/RealWorldQA
sẽ là:  [ảnh + question + choices]
question = entry["question"]
choices  = A, B, C, D  (multiple-choice)

RealWorldQA Data Preparation Script — BASELINE ONLY (no emotion).
Converts RealWorldQA dataset to inference format compatible with inference pipeline.

RealWorldQA là benchmark đánh giá spatial understanding của VLMs trong real-world settings.
Gồm ~700 samples, mỗi sample là multiple-choice (A/B/C/D) với ảnh thực tế.

Cấu trúc thư mục:
    original_data/RealWorldQA/
        images/              ← ảnh (0.jpg, 1.jpg, ...)
        RealWorldQA.json     ← list of dicts
        RealWorldQA.tsv      ← tabular form (fallback / alternative)

RealWorldQA.json fields (mỗi entry):
    index:    "0", "1", ...      (string index)
    image:    "images/0.jpg"     (path relative to DATA_DIR)
    question: câu hỏi
    A:        lựa chọn A (string, có thể rỗng "")
    B:        lựa chọn B
    C:        lựa chọn C
    D:        lựa chọn D (có thể rỗng "" nếu chỉ có 3 lựa chọn)
    answer:   "A" | "B" | "C" | "D"

Question format cho inference:
    <image>
    <question>
    A. <choice_A>
    B. <choice_B>
    C. <choice_C>
    D. <choice_D>        (chỉ hiển thị nếu D không rỗng)
    Answer with the option's letter from the given choices directly.

Modes:
    --json       → load từ RealWorldQA.json (preferred)
    --tsv        → load từ RealWorldQA.tsv  (fallback)
    --all        → cả hai (nếu muốn tạo song song để cross-check)

Output (processed_data/rwqa_baseline/):
    rwqa_baseline.json           (từ JSON source)
    rwqa_tsv_baseline.json       (từ TSV source)

Usage:
    python prepare_realworldqa.py --json
    python prepare_realworldqa.py --tsv
    python prepare_realworldqa.py --all
    python prepare_realworldqa.py --json --output_dir /custom/path
"""

import json
import os
import csv
import argparse
from pathlib import Path


# ============================================================================
# CONSTANT PATHS
# ============================================================================
DATA_DIR        = "/workspace/original_data/RealWorldQA"
IMAGE_DIR       = os.path.join(DATA_DIR, "images")
RWQA_JSON       = os.path.join(DATA_DIR, "RealWorldQA.json")
RWQA_TSV        = os.path.join(DATA_DIR, "RealWorldQA.tsv")
OUTPUT_BASE_DIR = "/workspace/processed_data"

# Instruction appended after choices
ANSWER_INSTRUCTION = "Answer with the option's letter from the given choices directly."


# ============================================================================
# DATA LOADING
# ============================================================================
def load_from_json(json_path=RWQA_JSON):
    """
    Load RealWorldQA.json.

    Format: list of dicts:
        [
          {
            "index": "0",
            "image": "images/0.jpg",
            "question": "...",
            "A": "Left",
            "B": "Straight",
            "C": "Right",
            "D": "",
            "answer": "C"
          },
          ...
        ]

    Returns:
        list of normalized dicts với "_index" (int), "_source" = "json"
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"RealWorldQA.json not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    data = []
    for i, entry in enumerate(raw):
        data.append(_normalize_entry(entry, i, source="json"))

    print(f"Loaded {len(data)} samples from RealWorldQA.json")
    return data


def load_from_tsv(tsv_path=RWQA_TSV):
    """
    Load RealWorldQA.tsv.

    Dự kiến columns (tab-separated):
        index  image  question  A  B  C  D  answer
    Có thể không có header — tự detect.

    Returns:
        list of normalized dicts với "_source" = "tsv"
    """
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"RealWorldQA.tsv not found: {tsv_path}")

    data = []
    csv.field_size_limit(10 * 1024 * 1024)
    with open(tsv_path, "r", encoding="utf-8") as f:
        # Peek first line để detect header
        first_line = f.readline().strip()
        f.seek(0)

        has_header = (
            "question" in first_line.lower()
            or "index" in first_line.lower()
        )

        reader = csv.DictReader(f, delimiter="\t") if has_header else None

        if has_header:
            reader = csv.DictReader(f, delimiter="\t")
            for i, row in enumerate(reader):
                data.append(_normalize_entry(dict(row), i, source="tsv"))
        else:
            # No header: assume order = index, image, question, A, B, C, D, answer
            reader = csv.reader(f, delimiter="\t")
            for i, row in enumerate(reader):
                if not row:
                    continue
                entry = {}
                cols = ["index", "image", "question", "A", "B", "C", "D", "answer"]
                for j, col in enumerate(cols):
                    entry[col] = row[j].strip() if j < len(row) else ""
                data.append(_normalize_entry(entry, i, source="tsv"))

    print(f"Loaded {len(data)} samples from RealWorldQA.tsv")
    return data


def _normalize_entry(entry, fallback_index, source):
    """
    Normalize một raw entry về chuẩn nội bộ.

    - index: convert sang int nếu có thể
    - image: normalize path → relative từ DATA_DIR
    - choices: strip whitespace, giữ rỗng nếu không có
    """
    # Index
    raw_index = entry.get("index", fallback_index)
    try:
        idx = int(raw_index)
    except (ValueError, TypeError):
        idx = fallback_index

    # Image path → chuẩn hóa relative từ DATA_DIR
    raw_image = entry.get("image", f"images/{idx}.jpg")
    image_relpath = _normalize_image_path(raw_image)

    # Choices — D có thể rỗng
    choices = {
        "A": entry.get("A", "").strip(),
        "B": entry.get("B", "").strip(),
        "C": entry.get("C", "").strip(),
        "D": entry.get("D", "").strip(),
    }

    return {
        "_index":       idx,
        "_source":      source,
        "index":        idx,
        "image_relpath": image_relpath,
        "question":     entry.get("question", "").strip(),
        "choices":      choices,
        "answer":       entry.get("answer", "").strip().upper(),
    }


def _normalize_image_path(raw_image):
    """
    Normalize image path sang relative từ DATA_DIR.

    Ví dụ:
        "images/0.jpg"                          → "images/0.jpg"
        "./images/0.jpg"                        → "images/0.jpg"
        "/absolute/path/images/0.jpg"           → "images/0.jpg"  (giữ images/filename)
        "RealWorldQA/images/0.jpg"              → "images/0.jpg"
    """
    p = raw_image.replace("\\", "/").strip()
    if p.startswith("./"):
        p = p[2:]
    # Strip absolute prefix nếu có — lấy phần từ "images/" trở đi
    if "images/" in p:
        p = "images/" + p.split("images/")[-1]
    return p


# ============================================================================
# QUESTION FORMATTING
# ============================================================================
def format_question(question, choices):
    """
    Format câu hỏi multiple-choice theo chuẩn inference:

        <question>
        A. <choice_A>
        B. <choice_B>
        C. <choice_C>
        D. <choice_D>     ← chỉ xuất hiện nếu D không rỗng
        Answer with the option's letter from the given choices directly.
    """
    lines = [question]
    for letter in ["A", "B", "C", "D"]:
        text = choices.get(letter, "").strip()
        if text:  # bỏ qua lựa chọn rỗng
            lines.append(f"{letter}. {text}")
    lines.append(ANSWER_INSTRUCTION)
    return "\n".join(lines)


# ============================================================================
# CONVERSION: RealWorldQA entry → inference format
# ============================================================================
def convert_sample(entry, subset_name):
    """
    Convert one RealWorldQA entry to inference format.

    Output format (mirrors figstep / mmvet / hallusion baseline):
    {
        "id": "rwqa_0000",
        "image": ["/images/0.jpg"],   # leading "/" — InferenceRunner strips và join
        "conversations": [
            {"from": "user", "value": "<image>\n<formatted_question>"}
        ],
        "metadata": { ... }
    }
    """
    idx         = entry["_index"]
    image_rp    = entry["image_relpath"]   # "images/0.jpg"
    question    = entry["question"]
    choices     = entry["choices"]
    answer      = entry["answer"]          # "A" | "B" | "C" | "D"
    source      = entry["_source"]

    # Format question với choices
    formatted_q = format_question(question, choices)

    # image path: leading "/" để InferenceRunner resolve đúng
    image_path   = f"/{image_rp}"
    user_message = f"<image>\n{formatted_q}"
    sample_id    = f"rwqa_{subset_name}_{str(idx).zfill(4)}"

    # GT answer letter → full text
    gt_answer_text = choices.get(answer, "")

    return {
        "id": sample_id,
        "image": [image_path],
        "conversations": [
            {"from": "user", "value": user_message}
        ],
        "metadata": {
            "scenario":            "realworldqa",
            "image_type":          "real_photo",
            "subset":              subset_name,
            "data_source":         source,
            "question_id":         idx,
            "original_question":   question,
            "formatted_question":  formatted_q,
            "used_question":       formatted_q,
            "question_type":       "multiple_choice",
            "choices":             choices,
            "gt_answer":           answer,           # "A" | "B" | "C" | "D"
            "gt_answer_text":      gt_answer_text,   # full text của đáp án đúng
            "image_relpath":       image_rp,
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
    """Thống kê answer distribution."""
    answer_dist = {}
    n_3choice = 0
    n_4choice = 0
    for s in samples:
        m = s["metadata"]
        ans = m.get("gt_answer", "")
        answer_dist[ans] = answer_dist.get(ans, 0) + 1
        d_text = m.get("choices", {}).get("D", "")
        if d_text:
            n_4choice += 1
        else:
            n_3choice += 1
    return {
        "total":            len(samples),
        "answer_distribution": answer_dist,
        "samples_with_3_choices": n_3choice,
        "samples_with_4_choices": n_4choice,
    }


# ============================================================================
# PREPARE FUNCTIONS
# ============================================================================
def prepare_from_json(output_dir):
    """Baseline từ RealWorldQA.json (preferred source)."""
    print(f"\n{'='*80}")
    print("REALWORLDQA BASELINE — from JSON")
    print(f"{'='*80}")

    os.makedirs(output_dir, exist_ok=True)

    data = load_from_json(RWQA_JSON)

    if not os.path.exists(IMAGE_DIR):
        print(f"  ⚠️  Image dir not found: {IMAGE_DIR}")

    samples = [convert_sample(e, subset_name="json") for e in data]

    output_path = os.path.join(output_dir, "rwqa_baseline.json")
    save_dataset(samples, output_path)

    stats = _make_stats(samples)
    summary = {
        "dataset":      "RealWorldQA",
        "subset":       "json",
        "mode":         "baseline (image + question + choices, no emotion)",
        "image_dir":    IMAGE_DIR,
        "source_file":  RWQA_JSON,
        "output_file":  "rwqa_baseline.json",
        **stats,
    }
    with open(os.path.join(output_dir, "rwqa_baseline_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"  Stats: {stats}")
    print(f"  Created: rwqa_baseline.json ({len(samples)} samples)")
    return ["rwqa_baseline.json"]


def prepare_from_tsv(output_dir):
    """Baseline từ RealWorldQA.tsv (fallback / alternative source)."""
    print(f"\n{'='*80}")
    print("REALWORLDQA BASELINE — from TSV")
    print(f"{'='*80}")

    os.makedirs(output_dir, exist_ok=True)

    data = load_from_tsv(RWQA_TSV)

    if not os.path.exists(IMAGE_DIR):
        print(f"  ⚠️  Image dir not found: {IMAGE_DIR}")

    samples = [convert_sample(e, subset_name="tsv") for e in data]

    output_path = os.path.join(output_dir, "rwqa_tsv_baseline.json")
    save_dataset(samples, output_path)

    stats = _make_stats(samples)
    summary = {
        "dataset":      "RealWorldQA",
        "subset":       "tsv",
        "mode":         "baseline (image + question + choices, no emotion)",
        "image_dir":    IMAGE_DIR,
        "source_file":  RWQA_TSV,
        "output_file":  "rwqa_tsv_baseline.json",
        **stats,
    }
    with open(os.path.join(output_dir, "rwqa_tsv_baseline_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"  Stats: {stats}")
    print(f"  Created: rwqa_tsv_baseline.json ({len(samples)} samples)")
    return ["rwqa_tsv_baseline.json"]


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Prepare RealWorldQA dataset — baseline only (image + question + choices, no emotion)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Constant paths:
  Input:   original_data/RealWorldQA/RealWorldQA.json  (or .tsv)
  Images:  original_data/RealWorldQA/images/
  Output:  processed_data/rwqa_baseline/

RealWorldQA.json format (list):
  [
    {
      "index":    "0",
      "image":    "images/0.jpg",
      "question": "In which direction is the front wheel of the car on the right side facing?",
      "A": "Left",
      "B": "Straight",
      "C": "Right",
      "D": "",
      "answer": "C"
    },
    ...
  ]

Question format sent to model:
  <image>
  <question>
  A. <choice_A>
  B. <choice_B>
  C. <choice_C>
  D. <choice_D>    (only shown if D is non-empty)
  Answer with the option's letter from the given choices directly.

Examples:
  python prepare_RWQA.py --json
  python prepare_RWQA.py --tsv
  python prepare_RWQA.py --all
  python prepare_RWQA.py --json --output_dir /custom/path
        """,
    )

    parser.add_argument("--json",       action="store_true",
                        help="Load from RealWorldQA.json (preferred)")
    parser.add_argument("--tsv",        action="store_true",
                        help="Load from RealWorldQA.tsv (fallback)")
    parser.add_argument("--all",        action="store_true",
                        help="Prepare both JSON and TSV sources")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: processed_data/rwqa_baseline)")

    args = parser.parse_args()

    if not any([args.json, args.tsv, args.all]):
        parser.error("Specify at least one source: --json, --tsv, or --all")

    output_dir = args.output_dir or os.path.join(OUTPUT_BASE_DIR, "rwqa_baseline")

    print(f"\n{'='*80}")
    print("REALWORLDQA DATA PREPARATION — BASELINE")
    print(f"{'='*80}")
    print(f"Data dir:   {DATA_DIR}")
    print(f"Image dir:  {IMAGE_DIR}")
    print(f"JSON file:  {RWQA_JSON}")
    print(f"TSV file:   {RWQA_TSV}")
    print(f"Output dir: {output_dir}")

    all_created = []

    # Auto-fallback: nếu --json nhưng JSON không tồn tại, thử TSV
    if (args.json or args.all):
        if os.path.exists(RWQA_JSON):
            files = prepare_from_json(output_dir)
            all_created.extend(files)
        else:
            print(f"\n  ❌ RealWorldQA.json not found: {RWQA_JSON}")
            if not args.tsv:
                print(f"     → Try --tsv to load from TSV instead.")

    if (args.tsv or args.all):
        if os.path.exists(RWQA_TSV):
            files = prepare_from_tsv(output_dir)
            all_created.extend(files)
        else:
            print(f"\n  ❌ RealWorldQA.tsv not found: {RWQA_TSV}")

    if not all_created:
        print("\n❌ No files created. Check that data files exist in DATA_DIR.")
        print(f"   DATA_DIR: {DATA_DIR}")
        print("   Download from: https://huggingface.co/datasets/xai-org/RealWorldQA")
        return

    print(f"\n{'='*80}")
    print("✅ REALWORLDQA PREPARATION COMPLETE")
    print(f"{'='*80}")
    print(f"Output dir:    {output_dir}")
    print(f"Files created: {all_created}")


if __name__ == "__main__":
    main()
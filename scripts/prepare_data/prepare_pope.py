"""
Docstring for scripts.prepare_data.prepare_pope
https://github.com/RUCAIBox/POPE  [EMNLP 2023 / arXiv:2305.10355]
sẽ là:  [ảnh + question]
question = entry["question"]    → "Is there a <object> in the image?"
answer   = entry["answer"]      → "yes" | "no"

POPE (Polling-based Object Probing Evaluation) Data Preparation Script — BASELINE ONLY.
Đánh giá object hallucination của LVLMs qua binary yes/no questions về sự tồn tại của object.

Paper: "Evaluating Object Hallucination in Large Vision-Language Models" (EMNLP 2023)
       Li et al., arXiv:2305.10355

Ba sampling strategies:
  - random:     random objects không tồn tại trong ảnh
  - popular:    objects phổ biến nhất trong dataset không tồn tại trong ảnh
  - adversarial: objects co-occur nhiều nhất với objects trong ảnh nhưng không tồn tại

Cấu trúc thư mục:
    original_data/pope/
        images/                   ← COCO val2014 images
        annotations.jsonl         ← tất cả samples (one JSON per line, có thể mixed splits)
        TinyVersion_ID_List.json  ← list IDs cho subset nhỏ hơn

annotations.jsonl fields (mỗi dòng là một JSON object):
    id:            "0", "1", ...        (string, sequential index)
    question_id:   "1", "2", ...        (string)
    question:      "Is there a snowboard in the image?"
    answer:        "yes" | "no"
    image_source:  "COCO_val2014_000000391895"  (filename WITHOUT extension)

TinyVersion_ID_List.json:
    list hoặc dict chứa IDs (question_id hoặc id) của TinyVersion subset

Image filename: "{image_source}.jpg"  →  images/COCO_val2014_000000391895.jpg

Modes:
    --full          → tất cả samples từ annotations.jsonl
    --tiny          → chỉ samples trong TinyVersion_ID_List.json
    --split         → tự detect & tách theo split (random/popular/adversarial) nếu có thể
    --all           → full + tiny + split

Output (processed_data/pope_baseline/):
    pope_full_baseline.json
    pope_tiny_baseline.json
    pope_split_random_baseline.json      (nếu split detect được)
    pope_split_popular_baseline.json
    pope_split_adversarial_baseline.json

Usage:
    python prepare_pope.py --full
    python prepare_pope.py --tiny
    python prepare_pope.py --split
    python prepare_pope.py --all
    python prepare_pope.py --full --output_dir /custom/path
"""

import json
import os
import argparse
import sys
from pathlib import Path

_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT))

from path_config import ORIGINAL_DATA_ROOT, PROCESSED_DATA_ROOT

# ============================================================================
# CONSTANT PATHS
# ============================================================================
DATA_DIR = str(ORIGINAL_DATA_ROOT / "pope")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
ANNOTATIONS_JSONL = os.path.join(DATA_DIR, "annotations.jsonl")
TINY_ID_LIST_JSON = os.path.join(DATA_DIR, "TinyVersion_ID_List.json")
OUTPUT_BASE_DIR = str(PROCESSED_DATA_ROOT)

# POPE 3 canonical split names
POPE_SPLITS = ["random", "popular", "adversarial"]

# Instruction appended to yes/no question
YESNO_INSTRUCTION = "Please answer yes or no."


# ============================================================================
# DATA LOADING
# ============================================================================
def load_annotations(jsonl_path=ANNOTATIONS_JSONL):
    """
    Load annotations.jsonl — one JSON object per line.

    Format (mỗi dòng):
        {
          "id": "0",
          "question_id": "1",
          "question": "Is there a snowboard in the image?",
          "answer": "yes",
          "image_source": "COCO_val2014_000000391895"
        }

    Có thể có thêm field "split" = "random"|"popular"|"adversarial" nếu file được tagged.

    Returns:
        list of dicts với "_index" (int), "_image_filename", "_split" (detected/inferred)
    """
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"annotations.jsonl not found: {jsonl_path}")

    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  ⚠️  Skipping malformed JSON at line {line_num + 1}: {e}")
                continue

            entry["_index"] = line_num

            # Image filename: image_source + ".jpg"
            image_source = entry.get("image_source", "")
            if image_source:
                entry["_image_filename"] = f"{image_source}.jpg"
            else:
                # Fallback: derive from id
                entry["_image_filename"] = ""

            # Detect split if field exists, otherwise mark as "unknown"
            entry["_split"] = _detect_split(entry)

            data.append(entry)

    print(f"Loaded {len(data)} samples from annotations.jsonl")

    # Thống kê nhanh
    split_counts = {}
    ans_counts = {"yes": 0, "no": 0}
    for e in data:
        s = e["_split"]
        split_counts[s] = split_counts.get(s, 0) + 1
        a = e.get("answer", "").lower()
        if a in ans_counts:
            ans_counts[a] += 1

    print(f"  Split distribution: {split_counts}")
    print(f"  Answer distribution: {ans_counts}")
    return data


def _detect_split(entry):
    """
    Detect split type từ entry.
    Ưu tiên: field "split" / "category" / "type" → detect từ image_source pattern
    """
    # 1. Explicit field
    for field in ["split", "category", "type", "pope_type"]:
        val = entry.get(field, "")
        if val:
            val_lower = val.lower()
            for split in POPE_SPLITS:
                if split in val_lower:
                    return split

    # 2. Detect từ image_source (một số version encode split vào filename)
    image_source = entry.get("image_source", "").lower()
    for split in POPE_SPLITS:
        if split in image_source:
            return split

    # 3. Detect từ question_id range (nếu data được concat theo thứ tự)
    #    POPE chuẩn: 3000 samples/split → random[1-3000], popular[3001-6000], adversarial[6001-9000]
    #    Tuy nhiên file này có thể khác → chỉ dùng khi không có cách nào khác
    try:
        qid = int(entry.get("question_id", -1))
        if 1 <= qid <= 3000:
            return "random"
        elif 3001 <= qid <= 6000:
            return "popular"
        elif 6001 <= qid <= 9000:
            return "adversarial"
    except (ValueError, TypeError):
        pass

    return "unknown"


def load_tiny_ids(json_path=TINY_ID_LIST_JSON):
    """
    Load TinyVersion_ID_List.json.

    Format có thể là:
        - list of IDs (int hoặc string):  [0, 1, 5, ...]  hoặc  ["0", "1", "5", ...]
        - list of question_ids:           [1, 2, 6, ...]
        - dict:                           {"0": True, "1": True, ...}

    Returns:
        set of string IDs (cả "id" và "question_id" đều sẽ được check)
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"TinyVersion_ID_List.json not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        id_set = {str(x) for x in raw}
    elif isinstance(raw, dict):
        id_set = {str(k) for k in raw.keys()}
    else:
        raise ValueError(f"Unexpected format in TinyVersion_ID_List.json: {type(raw)}")

    print(f"Loaded {len(id_set)} IDs from TinyVersion_ID_List.json")
    return id_set


# ============================================================================
# CONVERSION: POPE entry → inference format
# ============================================================================
def convert_sample(entry, subset_name):
    """
    Convert one POPE entry to inference format.

    POPE là binary yes/no question → question đơn giản, không cần format phức tạp.
    Instruction "Please answer yes or no." được append để guide model.

    Output format (mirrors figstep / mmvet / hallusion baseline):
    {
        "id": "pope_full_000000",
        "image": ["/COCO_val2014_000000391895.jpg"],
        "conversations": [
            {"from": "user", "value": "<image>\nIs there a snowboard in the image? Please answer yes or no."}
        ],
        "metadata": { ... }
    }
    """
    idx            = entry["_index"]
    entry_id       = str(entry.get("id", idx))
    question_id    = str(entry.get("question_id", idx))
    question       = entry.get("question", "").strip()
    answer         = entry.get("answer", "").strip().lower()    # "yes" | "no"
    image_source   = entry.get("image_source", "")
    image_filename = entry.get("_image_filename", "")
    split          = entry.get("_split", "unknown")

    # Format question + instruction
    formatted_q  = f"{question} {YESNO_INSTRUCTION}"
    user_message = f"<image>\n{formatted_q}"

    # image path: leading "/" để InferenceRunner resolve đúng
    image_list = [f"/{image_filename}"] if image_filename else []

    sample_id = f"pope_{subset_name}_{str(idx).zfill(6)}"

    return {
        "id": sample_id,
        "image": image_list,
        "conversations": [
            {"from": "user", "value": user_message}
        ],
        "metadata": {
            "scenario":            "pope",
            "image_type":          "real_photo",
            "subset":              subset_name,
            "pope_split":          split,           # "random" | "popular" | "adversarial" | "unknown"
            "entry_id":            entry_id,
            "question_id":         question_id,
            "original_question":   question,
            "formatted_question":  formatted_q,
            "used_question":       formatted_q,
            "question_type":       "yes_no",
            "gt_answer":           answer,           # "yes" | "no"
            "image_source":        image_source,
            "image_filename":      image_filename,
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
    """Thống kê distribution cho summary."""
    answer_dist = {"yes": 0, "no": 0}
    split_dist  = {}
    for s in samples:
        m = s["metadata"]
        ans = m.get("gt_answer", "")
        if ans in answer_dist:
            answer_dist[ans] += 1
        sp = m.get("pope_split", "unknown")
        split_dist[sp] = split_dist.get(sp, 0) + 1
    return {
        "total":              len(samples),
        "answer_distribution": answer_dist,
        "split_distribution":  split_dist,
    }


# ============================================================================
# PREPARE FUNCTIONS
# ============================================================================
def prepare_full(data, output_dir):
    """Tất cả samples từ annotations.jsonl."""
    print(f"\n{'='*80}")
    print(f"POPE BASELINE — Full ({len(data)} samples)")
    print(f"{'='*80}")

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(IMAGE_DIR):
        print(f"  ⚠️  Image dir not found: {IMAGE_DIR}")

    samples = [convert_sample(e, subset_name="full") for e in data]

    output_path = os.path.join(output_dir, "pope_full_baseline.json")
    save_dataset(samples, output_path)

    stats = _make_stats(samples)
    summary = {
        "dataset":      "POPE",
        "subset":       "full",
        "mode":         "baseline (image + question, no emotion)",
        "image_dir":    IMAGE_DIR,
        "source_file":  ANNOTATIONS_JSONL,
        "output_file":  "pope_full_baseline.json",
        **stats,
    }
    with open(os.path.join(output_dir, "pope_full_baseline_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"  Stats: {stats}")
    print(f"  Created: pope_full_baseline.json")
    return ["pope_full_baseline.json"]


def prepare_tiny(data, output_dir):
    """Chỉ samples có ID trong TinyVersion_ID_List.json."""
    print(f"\n{'='*80}")
    print("POPE BASELINE — TinyVersion Subset")
    print(f"{'='*80}")

    os.makedirs(output_dir, exist_ok=True)

    tiny_ids = load_tiny_ids(TINY_ID_LIST_JSON)

    # Match theo cả "id" lẫn "question_id" (vì không biết TinyVersion dùng field nào)
    filtered = [
        e for e in data
        if str(e.get("id", "")) in tiny_ids
        or str(e.get("question_id", "")) in tiny_ids
        or str(e.get("_index", "")) in tiny_ids
    ]

    if not filtered:
        print(f"  ❌ No samples matched TinyVersion IDs!")
        print(f"     Sample data IDs:     {[str(e.get('id','')) for e in data[:5]]}")
        print(f"     Sample question_ids: {[str(e.get('question_id','')) for e in data[:5]]}")
        print(f"     Sample tiny IDs:     {sorted(list(tiny_ids))[:10]}")
        return []

    if not os.path.exists(IMAGE_DIR):
        print(f"  ⚠️  Image dir not found: {IMAGE_DIR}")

    samples = [convert_sample(e, subset_name="tiny") for e in filtered]

    output_path = os.path.join(output_dir, "pope_tiny_baseline.json")
    save_dataset(samples, output_path)

    stats = _make_stats(samples)
    summary = {
        "dataset":           "POPE",
        "subset":            "tiny",
        "mode":              "baseline (image + question, no emotion)",
        "image_dir":         IMAGE_DIR,
        "source_file":       ANNOTATIONS_JSONL,
        "tiny_id_list_file": TINY_ID_LIST_JSON,
        "total_tiny_ids":    len(tiny_ids),
        "output_file":       "pope_tiny_baseline.json",
        **stats,
    }
    with open(os.path.join(output_dir, "pope_tiny_baseline_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"  Stats: {stats}")
    print(f"  Created: pope_tiny_baseline.json")
    return ["pope_tiny_baseline.json"]


def prepare_by_split(data, output_dir):
    """
    Tách và tạo file riêng cho mỗi split: random / popular / adversarial.
    Nếu split không detect được → báo cáo và bỏ qua.
    """
    print(f"\n{'='*80}")
    print("POPE BASELINE — By Split (random / popular / adversarial)")
    print(f"{'='*80}")

    os.makedirs(output_dir, exist_ok=True)

    # Group by split
    split_groups = {s: [] for s in POPE_SPLITS}
    unknown_samples = []
    for e in data:
        s = e.get("_split", "unknown")
        if s in split_groups:
            split_groups[s].append(e)
        else:
            unknown_samples.append(e)

    if unknown_samples:
        print(f"  ⚠️  {len(unknown_samples)} samples with unknown split (will be skipped in split mode)")
        print(f"     Tip: Check if annotations.jsonl has a 'split'/'category' field,")
        print(f"     or verify question_id range aligns with POPE standard (1-9000).")

    created_files = []

    for split_name in POPE_SPLITS:
        split_data = split_groups[split_name]
        if not split_data:
            print(f"  ⚠️  No samples found for split '{split_name}' — skipping")
            continue

        samples = [convert_sample(e, subset_name=f"split_{split_name}") for e in split_data]

        fname       = f"pope_split_{split_name}_baseline.json"
        fname_sum   = f"pope_split_{split_name}_baseline_summary.json"
        output_path = os.path.join(output_dir, fname)
        save_dataset(samples, output_path)

        stats = _make_stats(samples)
        summary = {
            "dataset":      "POPE",
            "subset":       f"split_{split_name}",
            "pope_split":   split_name,
            "description":  {
                "random":     "Randomly sampled non-existent objects",
                "popular":    "Most frequently occurring non-existent objects",
                "adversarial":"Most co-occurring non-existent objects (hardest)",
            }.get(split_name, ""),
            "mode":         "baseline (image + question, no emotion)",
            "image_dir":    IMAGE_DIR,
            "source_file":  ANNOTATIONS_JSONL,
            "output_file":  fname,
            **stats,
        }
        with open(os.path.join(output_dir, fname_sum), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"  [{split_name.upper():>11}] {len(samples):>5} samples → {fname}")
        created_files.append(fname)

    if not created_files:
        print(f"\n  ❌ No split files created.")
        print(f"     All samples have split='unknown'. This means split detection failed.")
        print(f"     Possible reasons:")
        print(f"       1. annotations.jsonl doesn't have a 'split'/'category' field")
        print(f"       2. question_id range doesn't follow POPE standard (1-9000)")
        print(f"     Suggestion: Use --full to get all data in one file instead.")

    return created_files


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Prepare POPE dataset — baseline only (image + yes/no question, no emotion)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
POPE: Polling-based Object Probing Evaluation [EMNLP 2023, arXiv:2305.10355]
Evaluates object hallucination via binary yes/no questions.

Constant paths:
  Input:   original_data/pope/annotations.jsonl
  TinySet: original_data/pope/TinyVersion_ID_List.json
  Images:  original_data/pope/images/   (COCO val2014)
  Output:  processed_data/pope_baseline/

annotations.jsonl format (one JSON per line):
  {"id": "0", "question_id": "1", "question": "Is there a snowboard in the image?",
   "answer": "yes", "image_source": "COCO_val2014_000000391895"}

Image filename: {image_source}.jpg  →  images/COCO_val2014_000000391895.jpg

Split detection (for --split mode):
  1. Explicit field: "split", "category", or "type" in each JSON entry
  2. image_source contains split name
  3. question_id range: 1-3000=random, 3001-6000=popular, 6001-9000=adversarial

Question format sent to model:
  <image>
  Is there a snowboard in the image? Please answer yes or no.

Examples:
  python prepare_pope.py --full
  python prepare_pope.py --tiny
  python prepare_pope.py --split
  python prepare_pope.py --all
  python prepare_pope.py --full --output_dir /custom/path
        """,
    )

    parser.add_argument("--full",       action="store_true",
                        help="All samples from annotations.jsonl")
    parser.add_argument("--tiny",       action="store_true",
                        help="TinyVersion subset (filtered by TinyVersion_ID_List.json)")
    parser.add_argument("--split",      action="store_true",
                        help="Separate files per split: random / popular / adversarial")
    parser.add_argument("--all",        action="store_true",
                        help="Prepare full + tiny + split")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: processed_data/pope_baseline)")

    args = parser.parse_args()

    if not any([args.full, args.tiny, args.split, args.all]):
        parser.error("Specify at least one mode: --full, --tiny, --split, or --all")

    output_dir = args.output_dir or os.path.join(OUTPUT_BASE_DIR, "pope_baseline")

    print(f"\n{'='*80}")
    print("POPE DATA PREPARATION — BASELINE")
    print(f"{'='*80}")
    print(f"Data dir:     {DATA_DIR}")
    print(f"Annotations:  {ANNOTATIONS_JSONL}")
    print(f"Tiny ID list: {TINY_ID_LIST_JSON}")
    print(f"Image dir:    {IMAGE_DIR}")
    print(f"Output dir:   {output_dir}")

    # Validate
    if not os.path.exists(ANNOTATIONS_JSONL):
        print(f"\n❌ annotations.jsonl not found: {ANNOTATIONS_JSONL}")
        print("   Download from: https://github.com/RUCAIBox/POPE")
        return

    # Load once, reuse
    data = load_annotations(ANNOTATIONS_JSONL)
    all_created = []

    if args.full or args.all:
        files = prepare_full(data, output_dir)
        all_created.extend(files)

    if args.tiny or args.all:
        if not os.path.exists(TINY_ID_LIST_JSON):
            print(f"\n  ⚠️  TinyVersion_ID_List.json not found: {TINY_ID_LIST_JSON} — skipping --tiny")
        else:
            files = prepare_tiny(data, output_dir)
            all_created.extend(files)

    if args.split or args.all:
        files = prepare_by_split(data, output_dir)
        all_created.extend(files)

    print(f"\n{'='*80}")
    print("✅ POPE PREPARATION COMPLETE")
    print(f"{'='*80}")
    print(f"Output dir:    {output_dir}")
    print(f"Files created: {all_created}")


if __name__ == "__main__":
    main()
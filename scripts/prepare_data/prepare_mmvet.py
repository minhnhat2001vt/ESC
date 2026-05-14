"""
Docstring for scripts.prepare_data.prepare_mmvet
https://github.com/yuweihao/MM-Vet
sẽ là:  [ảnh + question]
question = data[id]["question"]

MM-VET Data Preparation Script — BASELINE ONLY (no emotion).
Converts MM-VET dataset to inference format compatible with the inference pipeline.

MM-VET là benchmark đánh giá 6 khả năng VL tích hợp: recognition, OCR, knowledge,
language generation, spatial awareness, math. Có 218 samples (200 ảnh).

Cấu trúc thư mục:
    original_data/mm-vet/
        images/          ← ảnh (v1.png, v2.png, ...)
        mm-vet.json      ← toàn bộ 218 samples (dict keyed by "v1", "v2", ...)
        bard_set.json    ← subset IDs dùng để eval với Bard (list)

Hai mode được hỗ trợ:
    --full      → toàn bộ mm-vet.json (218 samples)
    --bardset   → chỉ subset trong bard_set.json
    --all       → cả hai

Output (processed_data/mmvet_baseline/):
    mmvet_full_baseline.json
    mmvet_bardset_baseline.json

Usage:
    python prepare_mmvet.py --full
    python prepare_mmvet.py --bardset
    python prepare_mmvet.py --all
    python prepare_mmvet.py --all --output_dir /custom/path
"""

import json
import os
import argparse
from pathlib import Path


# ============================================================================
# CONSTANT PATHS
# ============================================================================
DATA_DIR        = "/workspace/original_data/mm-vet"
IMAGE_DIR       = os.path.join(DATA_DIR, "images")
MMVET_JSON      = os.path.join(DATA_DIR, "mm-vet.json")
BARDSET_JSON    = os.path.join(DATA_DIR, "bard_set.json")
OUTPUT_BASE_DIR = "/workspace/processed_data"

# MM-VET capability labels (6 core)
CAPABILITIES = ["rec", "ocr", "know", "gen", "spat", "math"]


# ============================================================================
# DATA LOADING
# ============================================================================
def load_mmvet(json_path=MMVET_JSON):
    """
    Load mm-vet.json.

    Format:
        {
          "v1": {
            "imagename": "v1.png",
            "question":  "...",
            "answer":    "...",
            "capability": ["rec", "ocr"],
            "category":   "..."           # optional / may not exist in all versions
          },
          ...
        }

    Returns:
        list of dicts, each with key "_id" added.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"mm-vet.json not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    data = []
    for vid, value in raw.items():
        entry = dict(value)
        entry["_id"] = vid  # e.g. "v1", "v42"
        data.append(entry)

    # Sort by numeric index for reproducibility (v1, v2, ..., v218)
    def sort_key(e):
        try:
            return int(e["_id"].lstrip("v"))
        except ValueError:
            return 0

    data.sort(key=sort_key)
    print(f"Loaded {len(data)} samples from mm-vet.json")
    return data


def load_bardset(json_path=BARDSET_JSON):
    """
    Load bard_set.json.

    Format: list of IDs, e.g. ["v1", "v5", "v12", ...]

    Returns:
        set of ID strings.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"bard_set.json not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # bard_set.json có thể là list hoặc dict — chuẩn là list
    if isinstance(raw, list):
        id_set = set(raw)
    elif isinstance(raw, dict):
        # Một số phiên bản lưu dạng {id: True} hoặc {id: {...}}
        id_set = set(raw.keys())
    else:
        raise ValueError(f"Unexpected format in bard_set.json: {type(raw)}")

    print(f"Loaded {len(id_set)} IDs from bard_set.json")
    return id_set


# ============================================================================
# CONVERSION: MM-VET entry → inference format
# ============================================================================
def convert_sample(entry, subset_name):
    """
    Convert one MM-VET entry to the inference format.

    Output format (mirrors figstep / vlsafe baseline):
    {
        "id": "mmvet_v1",
        "image": ["/v1.png"],      # leading "/" — InferenceRunner strips và join với image_base_dir
        "conversations": [
            {"from": "user", "value": "<image>\n<question>"}
        ],
        "metadata": { ... }
    }
    """
    vid        = entry["_id"]              # "v1", "v42", ...
    imagename  = entry.get("imagename", f"{vid}.png")
    question   = entry["question"]
    answer     = entry.get("answer", "")
    capability = entry.get("capability", [])
    category   = entry.get("category", "")

    # image path: leading "/" để InferenceRunner resolve đúng
    image_path = f"/{imagename}"

    user_message = f"<image>\n{question}"

    sample_id = f"mmvet_{vid}"

    return {
        "id": sample_id,
        "image": [image_path],
        "conversations": [
            {"from": "user", "value": user_message}
        ],
        "metadata": {
            "scenario":            "mmvet",
            "image_type":          "natural",
            "subset":              subset_name,
            "mmvet_id":            vid,
            "imagename":           imagename,
            "question_id":         vid,
            "original_question":   question,
            "used_question":       question,
            "question_type":       "open_ended",
            "capability":          capability,
            "category":            category,
            "gt_answer":           answer,
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


# ============================================================================
# PREPARE FUNCTIONS
# ============================================================================
def prepare_full(output_dir):
    """
    Full MM-VET baseline: tất cả 218 samples từ mm-vet.json.
    """
    print(f"\n{'='*80}")
    print("MM-VET BASELINE — Full Set")
    print(f"{'='*80}")

    os.makedirs(output_dir, exist_ok=True)

    data = load_mmvet(MMVET_JSON)

    if not os.path.exists(IMAGE_DIR):
        print(f"  ⚠️  Image dir not found: {IMAGE_DIR}")

    samples = [convert_sample(e, subset_name="full") for e in data]

    output_path = os.path.join(output_dir, "mmvet_full_baseline.json")
    save_dataset(samples, output_path)

    # Thống kê capability distribution
    cap_dist = {}
    for e in data:
        for cap in e.get("capability", []):
            cap_dist[cap] = cap_dist.get(cap, 0) + 1

    summary = {
        "dataset":               "MM-VET",
        "subset":                "full",
        "mode":                  "baseline (image + question, no emotion)",
        "image_dir":             IMAGE_DIR,
        "source_file":           MMVET_JSON,
        "total_samples":         len(samples),
        "capability_distribution": cap_dist,
        "output_file":           "mmvet_full_baseline.json",
    }
    summary_path = os.path.join(output_dir, "mmvet_full_baseline_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"  Capability distribution: {cap_dist}")
    print(f"Created: mmvet_full_baseline.json ({len(samples)} samples)")
    return ["mmvet_full_baseline.json"]


def prepare_bardset(output_dir):
    """
    MM-VET BardSet baseline: chỉ những samples có ID trong bard_set.json.
    """
    print(f"\n{'='*80}")
    print("MM-VET BASELINE — BardSet Subset")
    print(f"{'='*80}")

    os.makedirs(output_dir, exist_ok=True)

    data    = load_mmvet(MMVET_JSON)
    bard_ids = load_bardset(BARDSET_JSON)

    # Lọc
    filtered = [e for e in data if e["_id"] in bard_ids]
    if not filtered:
        print(f"  ❌ No samples matched bard_set.json IDs! Check ID format.")
        print(f"     Example data IDs: {[e['_id'] for e in data[:5]]}")
        print(f"     Example bard IDs: {list(bard_ids)[:5]}")
        return []

    missed = bard_ids - {e["_id"] for e in filtered}
    if missed:
        print(f"  ⚠️  {len(missed)} bard_set IDs not found in mm-vet.json: {sorted(missed)[:10]}")

    if not os.path.exists(IMAGE_DIR):
        print(f"  ⚠️  Image dir not found: {IMAGE_DIR}")

    samples = [convert_sample(e, subset_name="bardset") for e in filtered]

    output_path = os.path.join(output_dir, "mmvet_bardset_baseline.json")
    save_dataset(samples, output_path)

    cap_dist = {}
    for e in filtered:
        for cap in e.get("capability", []):
            cap_dist[cap] = cap_dist.get(cap, 0) + 1

    summary = {
        "dataset":               "MM-VET",
        "subset":                "bardset",
        "mode":                  "baseline (image + question, no emotion)",
        "image_dir":             IMAGE_DIR,
        "source_file":           MMVET_JSON,
        "bardset_file":          BARDSET_JSON,
        "total_in_bardset":      len(bard_ids),
        "total_samples":         len(samples),
        "capability_distribution": cap_dist,
        "output_file":           "mmvet_bardset_baseline.json",
    }
    summary_path = os.path.join(output_dir, "mmvet_bardset_baseline_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"  Capability distribution: {cap_dist}")
    print(f"Created: mmvet_bardset_baseline.json ({len(samples)} samples)")
    return ["mmvet_bardset_baseline.json"]


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Prepare MM-VET dataset — baseline only (image + question, no emotion)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Constant paths:
  Input:    original_data/mm-vet/
  Images:   original_data/mm-vet/images/
  Output:   processed_data/mmvet_baseline/

mm-vet.json format (dict):
  {
    "v1": {"imagename": "v1.png", "question": "...", "answer": "...",
           "capability": ["rec", "ocr"], "category": "..."},
    ...
  }

bard_set.json format (list of IDs):
  ["v1", "v5", "v12", ...]

Examples:
  python prepare_mmvet.py --full
  python prepare_mmvet.py --bardset
  python prepare_mmvet.py --all
  python prepare_mmvet.py --all --output_dir /custom/path
        """,
    )

    parser.add_argument("--full",       action="store_true",
                        help="Prepare full MM-VET baseline (mm-vet.json, 218 samples)")
    parser.add_argument("--bardset",    action="store_true",
                        help="Prepare BardSet subset baseline (filtered by bard_set.json)")
    parser.add_argument("--all",        action="store_true",
                        help="Prepare both full and bardset")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: processed_data/mmvet_baseline)")

    args = parser.parse_args()

    if not any([args.full, args.bardset, args.all]):
        parser.error("Specify at least one mode: --full, --bardset, or --all")

    output_dir = args.output_dir or os.path.join(OUTPUT_BASE_DIR, "mmvet_baseline")

    print(f"\n{'='*80}")
    print("MM-VET DATA PREPARATION — BASELINE")
    print(f"{'='*80}")
    print(f"Data dir:    {DATA_DIR}")
    print(f"Image dir:   {IMAGE_DIR}")
    print(f"JSON file:   {MMVET_JSON}")
    print(f"BardSet:     {BARDSET_JSON}")
    print(f"Output dir:  {output_dir}")

    # Validate input files
    if not os.path.exists(MMVET_JSON):
        print(f"\n❌ mm-vet.json not found: {MMVET_JSON}")
        print("   Download from: https://github.com/yuweihao/MM-Vet")
        return

    all_created = []

    if args.full or args.all:
        files = prepare_full(output_dir)
        all_created.extend(files)

    if args.bardset or args.all:
        if not os.path.exists(BARDSET_JSON):
            print(f"\n  ⚠️  bard_set.json not found: {BARDSET_JSON} — skipping --bardset")
        else:
            files = prepare_bardset(output_dir)
            all_created.extend(files)

    print(f"\n{'='*80}")
    print("✅ MM-VET PREPARATION COMPLETE")
    print(f"{'='*80}")
    print(f"Output dir:    {output_dir}")
    print(f"Files created: {all_created}")


if __name__ == "__main__":
    main()
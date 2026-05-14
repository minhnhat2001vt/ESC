"""
Docstring for scripts.prepare_data.prepare_hallusion
https://github.com/tianyi-lab/HallusionBench [CVPR'24]
sẽ là:  [ảnh + question]  hoặc  [question]  (nếu visual_input == "0")
question = entry["question"]

HallusionBench Data Preparation Script — BASELINE ONLY (no emotion).
Converts HallusionBench dataset to inference format compatible with inference pipeline.

HallusionBench là benchmark CVPR'24 đánh giá visual hallucination và language hallucination
của LVLMs, gồm Yes/No questions với binary ground-truth (0/1).

Cấu trúc thư mục:
    original_data/hallusion_bench/
        VD/                    ← Visual Dependent images (illusion, misleading, ...)
            illusion/
            misleading/
            ...
        VS/                    ← Visual Supplement images (chart, map, table, ...)
            chart/
            map/
            table/
            ...
        HallusionBench.json    ← tất cả samples (list of dicts)

HallusionBench.json fields (mỗi entry):
    category:         "VD" | "VS"
    subcategory:      "illusion" | "misleading" | "chart" | "map" | ...
    visual_input:     "0" = text-only (không cần ảnh)
                      "1" = easy (original image)
                      "2" = hard (edited/manipulated image)
    set_id:           ID nhóm câu hỏi cùng figure
    figure_id:        "0" = original image, "1"+ = edited image
    question_id:      ID câu hỏi trong set
    sample_note:      label ngắn mô tả nội dung
    question:         câu hỏi (Yes/No question)
    gt_answer:        "0" (No) | "1" (Yes)
    gt_answer_details: giải thích chi tiết đáp án
    filename:         đường dẫn ảnh tương đối, dạng "./VD/illusion/0_0.png"
                      hoặc "./hallusion_bench/VD/illusion/0_0.png"

Modes được hỗ trợ:
    --full        → tất cả samples (kể cả text-only visual_input=0)
    --visual      → chỉ samples có ảnh (visual_input="1" hoặc "2")
    --vd          → chỉ Visual Dependent (category="VD")
    --vs          → chỉ Visual Supplement (category="VS")
    --all         → --full + --visual + --vd + --vs

Output (processed_data/hallusion_baseline/):
    hallusion_full_baseline.json
    hallusion_visual_baseline.json
    hallusion_vd_baseline.json
    hallusion_vs_baseline.json

Usage:
    python prepare_hallusion.py --full
    python prepare_hallusion.py --visual
    python prepare_hallusion.py --vd
    python prepare_hallusion.py --vs
    python prepare_hallusion.py --all
    python prepare_hallusion.py --visual --output_dir /custom/path
"""

import json
import os
import argparse
from pathlib import Path


# ============================================================================
# CONSTANT PATHS
# ============================================================================
DATA_DIR        = "/workspace/original_data/hallusion_bench"
IMAGE_DIR       = DATA_DIR                          # ảnh nằm ngay trong DATA_DIR: VD/, VS/
HALLUSION_JSON  = os.path.join(DATA_DIR, "HallusionBench.json")
OUTPUT_BASE_DIR = "/workspace/processed_data"

# visual_input values
VISUAL_INPUT_TEXT_ONLY = "0"   # không cần ảnh
VISUAL_INPUT_EASY      = "1"   # original / easy image
VISUAL_INPUT_HARD      = "2"   # edited / hard image


# ============================================================================
# DATA LOADING
# ============================================================================
def load_hallusion(json_path=HALLUSION_JSON):
    """
    Load HallusionBench.json.

    Format: list of dicts, each with:
        category, subcategory, visual_input, set_id, figure_id, question_id,
        sample_note, question, gt_answer, gt_answer_details, filename

    filename có thể là:
        "./VD/illusion/0_0.png"
        "./hallusion_bench/VD/illusion/0_0.png"
    → ta sẽ normalize về dạng "VD/illusion/0_0.png"

    Returns:
        list of dicts, mỗi entry có thêm "_index" và "_image_relpath"
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"HallusionBench.json not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    data = []
    for idx, entry in enumerate(raw):
        e = dict(entry)
        e["_index"] = idx

        # Normalize filename → relative path from IMAGE_DIR
        # e.g. "./VD/illusion/0_0.png"        → "VD/illusion/0_0.png"
        #       "./hallusion_bench/VD/..."     → "VD/illusion/0_0.png"
        raw_filename = e.get("filename", "")
        relpath = _normalize_filename(raw_filename)
        e["_image_relpath"] = relpath  # e.g. "VD/illusion/0_0.png" or "" if text-only

        data.append(e)

    print(f"Loaded {len(data)} samples from HallusionBench.json")

    # Thống kê nhanh
    n_vd  = sum(1 for e in data if e.get("category") == "VD")
    n_vs  = sum(1 for e in data if e.get("category") == "VS")
    n_vis = sum(1 for e in data if e.get("visual_input") != VISUAL_INPUT_TEXT_ONLY)
    n_txt = sum(1 for e in data if e.get("visual_input") == VISUAL_INPUT_TEXT_ONLY)
    print(f"  VD: {n_vd}  |  VS: {n_vs}")
    print(f"  With image (visual_input 1/2): {n_vis}  |  Text-only (visual_input 0): {n_txt}")

    return data


def _normalize_filename(raw_filename):
    """
    Normalize filename sang đường dẫn tương đối từ IMAGE_DIR.
    
    Ví dụ:
        "./VD/illusion/0_0.png"             → "VD/illusion/0_0.png"
        "./hallusion_bench/VD/illusion/..."  → "VD/illusion/0_0.png"
        "VD/illusion/0_0.png"               → "VD/illusion/0_0.png"
        ""                                  → ""
    """
    if not raw_filename or raw_filename.strip().lower() == "none":
        return ""

    # Dùng PurePosixPath để strip prefix an toàn
    p = raw_filename.replace("\\", "/")

    # Strip leading "./"
    if p.startswith("./"):
        p = p[2:]

    # Strip "hallusion_bench/" prefix nếu có
    if p.startswith("hallusion_bench/"):
        p = p[len("hallusion_bench/"):]

    return p


# ============================================================================
# CONVERSION: HallusionBench entry → inference format
# ============================================================================
def convert_sample(entry, subset_name):
    """
    Convert one HallusionBench entry to inference format.

    - Nếu visual_input == "0" (text-only): KHÔNG có <image> token, image list rỗng
    - Nếu visual_input == "1" hoặc "2": có <image> token, image list có 1 entry

    Output format (mirrors figstep / mmvet baseline):
    {
        "id": "hallusion_full_000042",
        "image": ["/VD/illusion/0_0.png"],   # leading "/" — InferenceRunner strips và join
        "conversations": [
            {"from": "user", "value": "<image>\n<question>"}
        ],
        "metadata": { ... }
    }
    """
    idx           = entry["_index"]
    category      = entry.get("category", "")
    subcategory   = entry.get("subcategory", "")
    visual_input  = entry.get("visual_input", "1")
    set_id        = entry.get("set_id", "")
    figure_id     = entry.get("figure_id", "")
    question_id   = entry.get("question_id", "")
    sample_note   = entry.get("sample_note", "")
    question      = entry.get("question", "")
    gt_answer     = entry.get("gt_answer", "")
    gt_answer_details = entry.get("gt_answer_details", "")
    relpath       = entry.get("_image_relpath", "")

    has_image = (visual_input != VISUAL_INPUT_TEXT_ONLY) and bool(relpath)

    # image list
    if has_image:
        image_list   = [f"/{relpath}"]   # leading "/" để InferenceRunner resolve đúng
        user_message = f"<image>\n{question}"
        image_type   = "easy" if visual_input == VISUAL_INPUT_EASY else "hard"
    else:
        image_list   = []
        user_message = question
        image_type   = "text_only"

    # Unique sample ID: category_subcategory_set_figure_question
    uid = f"{category}_{subcategory}_{set_id}_{figure_id}_{question_id}"
    sample_id = f"hallusion_{subset_name}_{str(idx).zfill(5)}_{uid}"

    return {
        "id": sample_id,
        "image": image_list,
        "conversations": [
            {"from": "user", "value": user_message}
        ],
        "metadata": {
            "scenario":            "hallusion_bench",
            "image_type":          image_type,
            "subset":              subset_name,
            "category":            category,        # "VD" | "VS"
            "subcategory":         subcategory,     # "illusion", "chart", ...
            "visual_input":        visual_input,    # "0", "1", "2"
            "set_id":              set_id,
            "figure_id":           figure_id,
            "question_id":         question_id,
            "sample_note":         sample_note,
            "original_question":   question,
            "used_question":       question,
            "question_type":       "yes_no",        # HallusionBench là binary yes/no
            "gt_answer":           gt_answer,       # "0" (No) | "1" (Yes)
            "gt_answer_details":   gt_answer_details,
            "image_relpath":       relpath,
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
    stats = {
        "total": len(samples),
        "by_category":     {},
        "by_subcategory":  {},
        "by_visual_input": {"0": 0, "1": 0, "2": 0},
    }
    for s in samples:
        m = s["metadata"]
        cat  = m.get("category", "")
        sub  = m.get("subcategory", "")
        vi   = m.get("visual_input", "")
        stats["by_category"][cat]    = stats["by_category"].get(cat, 0) + 1
        stats["by_subcategory"][sub] = stats["by_subcategory"].get(sub, 0) + 1
        if vi in stats["by_visual_input"]:
            stats["by_visual_input"][vi] += 1
    return stats


# ============================================================================
# PREPARE FUNCTIONS
# ============================================================================
def prepare_subset(data, output_dir, subset_name, filter_fn=None, description=""):
    """
    Generic prepare function: áp filter_fn (nếu có) rồi convert và lưu.
    
    Args:
        data:        toàn bộ data đã load
        output_dir:  thư mục output
        subset_name: tên subset ("full", "visual", "vd", "vs")
        filter_fn:   lambda/function lọc entries, None = lấy tất cả
        description: mô tả cho summary
    """
    os.makedirs(output_dir, exist_ok=True)

    filtered = [e for e in data if filter_fn(e)] if filter_fn else data

    print(f"\n{'='*80}")
    print(f"HALLUSIONBENCH BASELINE — {subset_name.upper()} ({len(filtered)} samples)")
    print(f"{'='*80}")
    if not filtered:
        print(f"  ❌ No samples matched filter for subset '{subset_name}'!")
        return []

    if not os.path.exists(IMAGE_DIR):
        print(f"  ⚠️  Image dir not found: {IMAGE_DIR}")

    samples = [convert_sample(e, subset_name=subset_name) for e in filtered]

    fname       = f"hallusion_{subset_name}_baseline.json"
    fname_sum   = f"hallusion_{subset_name}_baseline_summary.json"
    output_path = os.path.join(output_dir, fname)
    save_dataset(samples, output_path)

    stats = _make_stats(samples)
    summary = {
        "dataset":       "HallusionBench",
        "subset":        subset_name,
        "description":   description,
        "mode":          "baseline (image + question, no emotion)",
        "image_dir":     IMAGE_DIR,
        "source_file":   HALLUSION_JSON,
        "output_file":   fname,
        **stats,
    }
    with open(os.path.join(output_dir, fname_sum), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"  Stats: {stats}")
    print(f"  Created: {fname}")
    return [fname]


# ============================================================================
# PUBLIC PREPARE ENTRYPOINTS
# ============================================================================
def prepare_full(data, output_dir):
    """Tất cả samples, kể cả text-only (visual_input=0)."""
    return prepare_subset(
        data, output_dir,
        subset_name="full",
        filter_fn=None,
        description="All samples including text-only (visual_input=0)",
    )


def prepare_visual(data, output_dir):
    """Chỉ samples có ảnh: visual_input='1' hoặc '2'."""
    return prepare_subset(
        data, output_dir,
        subset_name="visual",
        filter_fn=lambda e: e.get("visual_input") in (VISUAL_INPUT_EASY, VISUAL_INPUT_HARD),
        description="Visual samples only (visual_input=1 or 2, excludes text-only)",
    )


def prepare_vd(data, output_dir):
    """Visual Dependent subset (category='VD')."""
    return prepare_subset(
        data, output_dir,
        subset_name="vd",
        filter_fn=lambda e: e.get("category") == "VD",
        description="Visual Dependent (VD) — questions requiring visual context",
    )


def prepare_vs(data, output_dir):
    """Visual Supplement subset (category='VS')."""
    return prepare_subset(
        data, output_dir,
        subset_name="vs",
        filter_fn=lambda e: e.get("category") == "VS",
        description="Visual Supplement (VS) — questions with visual as supplement",
    )


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Prepare HallusionBench dataset — baseline only (image + question, no emotion)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Constant paths:
  Input:   original_data/hallusion_bench/HallusionBench.json
  Images:  original_data/hallusion_bench/VD/  |  original_data/hallusion_bench/VS/
  Output:  processed_data/hallusion_baseline/

HallusionBench.json format (list):
  [
    {
      "category": "VD",            # Visual Dependent
      "subcategory": "illusion",
      "visual_input": "1",         # "0"=text-only, "1"=easy/orig, "2"=hard/edited
      "set_id": "0",
      "figure_id": "0",            # "0"=original image, "1+"=edited image
      "question_id": "0",
      "sample_note": "circle",
      "question": "Is the right orange circle the same size as the left orange circle?",
      "gt_answer": "1",            # "0"=No, "1"=Yes
      "gt_answer_details": "...",
      "filename": "./VD/illusion/0_0.png"
    },
    ...
  ]

visual_input note:
  "0" → text-only, không có ảnh → output KHÔNG có <image> token
  "1" → easy (original image)   → output CÓ <image> token
  "2" → hard (edited image)     → output CÓ <image> token

Examples:
  python prepare_hallusion.py --full
  python prepare_hallusion.py --visual
  python prepare_hallusion.py --vd
  python prepare_hallusion.py --vs
  python prepare_hallusion.py --all
  python prepare_hallusion.py --visual --output_dir /custom/path
        """,
    )

    parser.add_argument("--full",       action="store_true",
                        help="All samples including text-only (visual_input=0)")
    parser.add_argument("--visual",     action="store_true",
                        help="Visual samples only (visual_input=1 or 2)")
    parser.add_argument("--vd",         action="store_true",
                        help="Visual Dependent subset (category=VD)")
    parser.add_argument("--vs",         action="store_true",
                        help="Visual Supplement subset (category=VS)")
    parser.add_argument("--all",        action="store_true",
                        help="Prepare all modes: full + visual + vd + vs")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: processed_data/hallusion_baseline)")

    args = parser.parse_args()

    if not any([args.full, args.visual, args.vd, args.vs, args.all]):
        parser.error("Specify at least one mode: --full, --visual, --vd, --vs, or --all")

    output_dir = args.output_dir or os.path.join(OUTPUT_BASE_DIR, "hallusion_baseline")

    print(f"\n{'='*80}")
    print("HALLUSIONBENCH DATA PREPARATION — BASELINE")
    print(f"{'='*80}")
    print(f"Data dir:   {DATA_DIR}")
    print(f"JSON file:  {HALLUSION_JSON}")
    print(f"Image dir:  {IMAGE_DIR}")
    print(f"Output dir: {output_dir}")

    # Validate
    if not os.path.exists(HALLUSION_JSON):
        print(f"\n❌ HallusionBench.json not found: {HALLUSION_JSON}")
        print("   Download from: https://github.com/tianyi-lab/HallusionBench")
        return

    # Load once, reuse
    data = load_hallusion(HALLUSION_JSON)
    all_created = []

    if args.full or args.all:
        files = prepare_full(data, output_dir)
        all_created.extend(files)

    if args.visual or args.all:
        files = prepare_visual(data, output_dir)
        all_created.extend(files)

    if args.vd or args.all:
        files = prepare_vd(data, output_dir)
        all_created.extend(files)

    if args.vs or args.all:
        files = prepare_vs(data, output_dir)
        all_created.extend(files)

    print(f"\n{'='*80}")
    print("✅ HALLUSIONBENCH PREPARATION COMPLETE")
    print(f"{'='*80}")
    print(f"Output dir:    {output_dir}")
    print(f"Files created: {all_created}")


if __name__ == "__main__":
    main()
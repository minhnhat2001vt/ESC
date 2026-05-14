"""
BLINK (Multimodal Large Language Models Can See but Not Perceive) Data Preparation Script — BASELINE ONLY

BLINK benchmark evaluates core visual perception abilities in VLMs across 14 classic
computer vision tasks, reformatted as 3,807 multiple-choice questions paired with
single or multiple images and visual prompting.

Multi-image handling (following the BLINK paper, ECCV 2024):
    "For the models that do not support multiple images as input,
     we concatenate the images as input."
    Images are placed horizontally with a black margin in between,
    producing a SINGLE concatenated image per sample. This ensures
    compatibility with single-image VLMs (e.g., LLaVA-1.5) while
    preserving all visual information.

Dataset structure (downloaded from HuggingFace BLINK-Benchmark/BLINK):
    original_data/blink/
        dataset/
            Art_Style/
                test-00000-of-00001.parquet
                val-00000-of-00001.parquet
            Counting/
                test-00000-of-00001.parquet
                val-00000-of-00001.parquet
            Forensic_Detection/
            Functional_Correspondence/
            IQ_Test/
            Jigsaw/
            Multi-view_Reasoning/
            Object_Localization/
            Relative_Depth/
            Relative_Reflectance/
            Semantic_Correspondence/
            Spatial_Relation/
            Visual_Correspondence/
            Visual_Similarity/

Parquet columns:
    idx          — string, e.g. "val_Art_Style_1"
    question     — string, short question text
    sub_task     — string, e.g. "Art Style"
    image_1      — PIL Image (always present)
    image_2      — PIL Image (often present)
    image_3      — PIL Image (sometimes present)
    image_4      — PIL Image (rarely present)
    choices      — list of strings, e.g. ["the second image", "the third image"]
    answer       — string, e.g. "(A)", "(B)", "(C)", "(D)"
    prompt       — string, the full prompt with context and options
    explanation  — string (may be empty)

Each sample may have 1–4 images. Multi-image samples are concatenated
horizontally into a single image. The prompt column contains the complete
prompt used for evaluation (with context, question, and options).

Question format for inference (using the dataset's own prompt):
    <image>
    <prompt from dataset>
    Answer with the option's letter from the given choices directly.

Output:
    processed_data/blink_baseline/
        blink_full_baseline.json              (all val samples)
        blink_baseline_summary.json
        images/                               (individual + concatenated images)
            val_Art_Style_1_img1.jpg           (individual)
            val_Art_Style_1_img2.jpg           (individual)
            val_Art_Style_1_concat.jpg         (concatenated — used for inference)
            ...

Usage:
    python prepare_blink.py
    python prepare_blink.py --output_dir /custom/path
    python prepare_blink.py --data_dir /path/to/blink
    python prepare_blink.py --split val          # default: val (has answers)
    python prepare_blink.py --split test         # test set (no answers for most)
    python prepare_blink.py --subtasks Art_Style Counting  # specific subtasks only

Reference:
    BLINK: Multimodal Large Language Models Can See but Not Perceive (ECCV 2024)
    https://github.com/zeyofu/BLINK_Benchmark
    https://huggingface.co/datasets/BLINK-Benchmark/BLINK
"""

import json
import os
import re
import argparse
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    raise ImportError("pandas is required. Install with: pip install pandas")

try:
    from PIL import Image
except ImportError:
    raise ImportError("Pillow is required. Install with: pip install Pillow")


# ============================================================================
# CONSTANT PATHS
# ============================================================================
DATA_DIR        = "/workspace/original_data/blink/hf_dataset_repo"
DATASET_DIR_NAME = "dataset"  # subfolder containing parquet files
OUTPUT_BASE_DIR = "/workspace/processed_data"

ANSWER_INSTRUCTION = "Answer with the option's letter from the given choices directly."

# All 14 BLINK subtasks
ALL_SUBTASKS = [
    "Art_Style",
    "Counting",
    "Forensic_Detection",
    "Functional_Correspondence",
    "IQ_Test",
    "Jigsaw",
    "Multi-view_Reasoning",
    "Object_Localization",
    "Relative_Depth",
    "Relative_Reflectance",
    "Semantic_Correspondence",
    "Spatial_Relation",
    "Visual_Correspondence",
    "Visual_Similarity",
]

# Image columns in the parquet
IMAGE_COLUMNS = ["image_1", "image_2", "image_3", "image_4"]


# ============================================================================
# DATA LOADING
# ============================================================================
def find_parquet_file(subtask_dir, split):
    """
    Find the parquet file for a given split in a subtask directory.

    Handles patterns like:
        val-00000-of-00001.parquet
        test-00000-of-00001.parquet
    """
    if not os.path.exists(subtask_dir):
        return None

    for fname in os.listdir(subtask_dir):
        if fname.startswith(f"{split}-") and fname.endswith(".parquet"):
            return os.path.join(subtask_dir, fname)

    return None


def load_subtask(subtask_dir, subtask_name, split="val"):
    """
    Load a single subtask's parquet file.

    Returns: pandas DataFrame or None if not found
    """
    parquet_path = find_parquet_file(subtask_dir, split)
    if parquet_path is None:
        print(f"  ⚠️  No {split} parquet found for {subtask_name} in {subtask_dir}")
        return None

    df = pd.read_parquet(parquet_path)
    print(f"  Loaded {len(df)} samples from {subtask_name}/{split}")
    return df


def load_all_subtasks(data_dir, split="val", subtasks=None):
    """
    Load all (or specified) subtasks.

    Returns: dict of {subtask_name: DataFrame}
    """
    dataset_dir = os.path.join(data_dir, DATASET_DIR_NAME)
    if not os.path.exists(dataset_dir):
        # Try data_dir directly (maybe dataset/ is not a subfolder)
        dataset_dir = data_dir
        print(f"  ℹ️  No '{DATASET_DIR_NAME}' subfolder found, using {data_dir} directly")

    subtask_list = subtasks or ALL_SUBTASKS
    results = {}

    for subtask in subtask_list:
        subtask_dir = os.path.join(dataset_dir, subtask)
        if not os.path.exists(subtask_dir):
            print(f"  ⚠️  Subtask directory not found: {subtask_dir}")
            continue

        df = load_subtask(subtask_dir, subtask, split)
        if df is not None and len(df) > 0:
            results[subtask] = df

    total = sum(len(df) for df in results.values())
    print(f"\nLoaded {total} total samples from {len(results)} subtasks ({split} split)")
    return results


# ============================================================================
# IMAGE EXTRACTION
# ============================================================================
def _is_none_safe(val):
    """
    Safely check if a value is None, handling numpy arrays and other types
    that don't support direct truth-value comparison.
    """
    if val is None:
        return True
    try:
        import numpy as np
        if isinstance(val, np.ndarray):
            return False  # numpy array is not None
    except ImportError:
        pass
    # For pandas NA / NaT
    try:
        if pd.isna(val):
            return True
    except (ValueError, TypeError):
        # pd.isna can raise ValueError for arrays
        pass
    return False


def extract_images(row, idx_str, image_output_dir):
    """
    Extract images from a parquet row and save to disk.

    HuggingFace parquet stores images as dicts with {"bytes": <bytes>, "path": <str>}
    or as raw bytes/numpy arrays depending on the pyarrow version.

    Returns: list of relative image paths (relative to blink data dir)
    """
    import io

    image_paths = []
    pil_images = []

    for i, col in enumerate(IMAGE_COLUMNS, start=1):
        if col not in row:
            continue

        img_data = row[col]

        # Safe None check (handles numpy arrays, pandas NA, etc.)
        if _is_none_safe(img_data):
            continue

        # Handle different image formats from parquet
        try:
            img = None

            if isinstance(img_data, Image.Image):
                img = img_data
            elif isinstance(img_data, dict):
                # HuggingFace format: {"bytes": b"...", "path": "..."}
                if "bytes" in img_data and img_data["bytes"] is not None:
                    raw_bytes = img_data["bytes"]
                    # bytes might be numpy array or python bytes
                    if hasattr(raw_bytes, 'tobytes'):
                        raw_bytes = raw_bytes.tobytes()
                    elif not isinstance(raw_bytes, bytes):
                        raw_bytes = bytes(raw_bytes)
                    img = Image.open(io.BytesIO(raw_bytes))
                elif "path" in img_data and img_data["path"]:
                    img = Image.open(img_data["path"])
            elif isinstance(img_data, bytes):
                img = Image.open(io.BytesIO(img_data))
            elif hasattr(img_data, 'tobytes'):
                # numpy array of bytes
                img = Image.open(io.BytesIO(img_data.tobytes()))

            if img is None:
                continue

        except Exception as e:
            print(f"    ⚠️  Failed to load {col} for {idx_str}: {e}")
            continue

        # Save image
        img_filename = f"{idx_str}_img{i}.jpg"
        img_path = os.path.join(image_output_dir, img_filename)

        # Convert to RGB if necessary (handle RGBA, palette, etc.)
        if img.mode != "RGB":
            img = img.convert("RGB")

        img.save(img_path, "JPEG", quality=95)
        image_paths.append(f"images/{img_filename}")
        pil_images.append(img)

    return image_paths, pil_images


# ============================================================================
# IMAGE CONCATENATION (Following BLINK paper: horizontal + black margin)
# ============================================================================
CONCAT_MARGIN_PX = 10  # black margin between images (pixels)


def concatenate_images_horizontal(pil_images, margin=CONCAT_MARGIN_PX):
    """
    Concatenate multiple PIL images into a single image, placed horizontally
    with a black margin in between.

    Following the BLINK paper (ECCV 2024):
        "For the models that do not support multiple images as input,
         we concatenate the images as input."
        "we place the images horizontally, with a black margin in between."

    All images are resized to the same height (the max height among them)
    before concatenation, preserving aspect ratio.

    Args:
        pil_images: list of PIL.Image.Image (already RGB)
        margin: black margin width in pixels between images

    Returns:
        Single concatenated PIL.Image.Image (RGB)
    """
    if len(pil_images) == 0:
        return None
    if len(pil_images) == 1:
        return pil_images[0]

    # Resize all images to the same height (max height), preserving aspect ratio
    target_h = max(img.height for img in pil_images)
    resized = []
    for img in pil_images:
        if img.height != target_h:
            scale = target_h / img.height
            new_w = int(img.width * scale)
            img = img.resize((new_w, target_h), Image.LANCZOS)
        resized.append(img)

    # Calculate total width: sum of widths + margins between images
    total_w = sum(img.width for img in resized) + margin * (len(resized) - 1)

    # Create black canvas and paste images
    concat = Image.new("RGB", (total_w, target_h), (0, 0, 0))
    x_offset = 0
    for img in resized:
        concat.paste(img, (x_offset, 0))
        x_offset += img.width + margin

    return concat


# ============================================================================
# ANSWER PARSING
# ============================================================================
def parse_answer(answer_str):
    """
    Parse BLINK answer string.

    "(A)" → "A"
    "(B)" → "B"
    "A"   → "A"
    """
    if not answer_str:
        return ""
    match = re.search(r'\(([A-Z])\)', answer_str)
    if match:
        return match.group(1)
    # Fallback
    cleaned = answer_str.strip().upper().replace("(", "").replace(")", "")
    return cleaned


def format_choices(choices):
    """
    Format choices list into options dict and string.

    ["the second image", "the third image"]
    → options: {"A": "the second image", "B": "the third image"}
    → options_str: "(A) the second image (B) the third image"
    """
    letters = [chr(ord("A") + i) for i in range(len(choices))]
    options = {letter: choice for letter, choice in zip(letters, choices)}
    options_str = " ".join(f"({letter}) {choice}" for letter, choice in zip(letters, choices))
    return options, options_str


# ============================================================================
# QUESTION FORMATTING
# ============================================================================
def format_question_from_prompt(prompt):
    """
    Use the dataset's own prompt directly (it already includes context + options).
    Append the answer instruction.

    Returns the formatted question string.
    """
    return f"{prompt}\n{ANSWER_INSTRUCTION}"


def format_question_from_parts(question, options):
    """
    Build question from parts (fallback if prompt column is missing).

    Output:
        <question>
        (A) <option_a>
        (B) <option_b>
        Answer with the option's letter from the given choices directly.
    """
    lines = [question]
    for letter in sorted(options.keys()):
        lines.append(f"({letter}) {options[letter]}")
    lines.append(ANSWER_INSTRUCTION)
    return "\n".join(lines)


# ============================================================================
# CONVERSION: BLINK row → inference format
# ============================================================================
def convert_sample(row, subtask_name, image_output_dir, use_prompt_column=True):
    """
    Convert one BLINK parquet row to inference format.

    Following the BLINK paper (ECCV 2024): for models that do not support
    multiple images as input, images are concatenated horizontally with a
    black margin in between. This produces a SINGLE concatenated image per
    sample so the inference pipeline (which is single-image) works correctly.

    Individual images are still saved for reference/debugging.

    Output format (mirrors MMVP / RWQA / figstep / pope baseline):
    {
        "id": "blink_val_Art_Style_1",
        "image": ["images/val_Art_Style_1_concat.jpg"],
        "conversations": [
            {"from": "user", "value": "<image>\n<formatted_question>"}
        ],
        "metadata": { ... }
    }
    """
    idx_str = str(row.get("idx", ""))
    question = str(row.get("question", ""))
    sub_task = str(row.get("sub_task", subtask_name))
    answer_raw = str(row.get("answer", ""))
    prompt = str(row.get("prompt", ""))
    explanation = str(row.get("explanation", "") or "")

    # Safely convert choices (may be numpy array, list, or None)
    raw_choices = row.get("choices", [])
    if _is_none_safe(raw_choices):
        choices = []
    else:
        choices = list(raw_choices)  # convert numpy array to list if needed

    # Parse answer
    answer_letter = parse_answer(answer_raw)

    # Format choices
    options, options_str = format_choices(choices) if len(choices) > 0 else ({}, "")

    # Get answer text
    answer_text = options.get(answer_letter, "")

    # Extract and save individual images (also returns PIL objects)
    individual_image_paths, pil_images = extract_images(row, idx_str, image_output_dir)
    num_images_original = len(individual_image_paths)

    # Concatenate images horizontally (BLINK paper protocol)
    if len(pil_images) > 1:
        concat_img = concatenate_images_horizontal(pil_images)
        concat_filename = f"{idx_str}_concat.jpg"
        concat_path = os.path.join(image_output_dir, concat_filename)
        concat_img.save(concat_path, "JPEG", quality=95)
        final_image_path = f"images/{concat_filename}"
    elif len(pil_images) == 1:
        # Single image — use the individual image directly (no concatenation needed)
        final_image_path = individual_image_paths[0]
    else:
        final_image_path = None

    # Build formatted question
    if use_prompt_column and prompt:
        formatted_q = format_question_from_prompt(prompt)
    else:
        formatted_q = format_question_from_parts(question, options)

    # Build image fields:
    # - "image": concat image (for single-image models like LLaVA)
    # - "image_list": individual images (for multi-image models like Qwen2-VL)
    if final_image_path:
        image_field = [final_image_path]
        # If only 1 original image, image_list == image_field (no difference)
        image_list_field = individual_image_paths if individual_image_paths else [final_image_path]
    else:
        image_field = []
        image_list_field = []

    # Number of <image> tokens matches individual images (for multi-image models)
    num_image_tokens = len(image_list_field)
    if num_image_tokens > 1:
        image_tokens = " ".join(["<image>"] * num_image_tokens)
        multi_user_message = f"{image_tokens}\n{formatted_q}"
    else:
        multi_user_message = f"<image>\n{formatted_q}" if image_list_field else formatted_q

    # Single <image> tag message (for single-image models)
    single_user_message = f"<image>\n{formatted_q}" if final_image_path else formatted_q

    # Sample ID
    sample_id = f"blink_{idx_str}"

    return {
        "id": sample_id,
        "image": image_field,           # concat image path (single-image models)
        "image_list": image_list_field, # individual image paths (multi-image models)
        "conversations": [
            {"from": "user", "value": single_user_message}   # single-image format
        ],
        "conversations_multi": [
            {"from": "user", "value": multi_user_message}    # multi-image format
        ],
        "metadata": {
            "scenario":            "blink",
            "image_type":          "visual_perception",
            "data_source":         "parquet",
            "question_id":         idx_str,
            "subtask":             subtask_name,
            "sub_task_display":    sub_task,
            "original_question":   question,
            "formatted_question":  formatted_q,
            "question_type":       "multiple_choice",
            "options":             options,
            "options_raw":         options_str,
            "choices":             choices,
            "gt_answer":           answer_letter,
            "gt_answer_raw":       answer_raw,
            "gt_answer_text":      answer_text,
            "num_images":          1 if final_image_path else 0,
            "num_images_original": num_images_original,
            "image_filenames":     individual_image_paths,
            "image_list":          image_list_field,   # individual paths for multi-image models
            "concat_image":        final_image_path or "",
            "prompt":              prompt,
            "explanation":         explanation,
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
    subtask_dist = {}
    num_images_original_dist = {}
    concat_count = 0

    for s in samples:
        m = s["metadata"]
        ans = m.get("gt_answer", "")
        answer_dist[ans] = answer_dist.get(ans, 0) + 1

        subtask = m.get("subtask", "unknown")
        subtask_dist[subtask] = subtask_dist.get(subtask, 0) + 1

        n_img = m.get("num_images_original", m.get("num_images", 0))
        num_images_original_dist[n_img] = num_images_original_dist.get(n_img, 0) + 1
        if n_img > 1:
            concat_count += 1

    return {
        "total": len(samples),
        "num_subtasks": len(subtask_dist),
        "concatenated_samples": concat_count,
        "answer_distribution": dict(sorted(answer_dist.items())),
        "subtask_distribution": dict(sorted(subtask_dist.items())),
        "num_images_original_distribution": dict(sorted(num_images_original_dist.items())),
    }


# ============================================================================
# PREPARE
# ============================================================================
def prepare_baseline(output_dir, data_dir=DATA_DIR, split="val", subtasks=None):
    """Prepare BLINK baseline dataset."""
    print(f"\n{'='*80}")
    print("BLINK BASELINE PREPARATION")
    print(f"{'='*80}")

    os.makedirs(output_dir, exist_ok=True)

    # Create images output directory
    image_output_dir = os.path.join(output_dir, "images")
    os.makedirs(image_output_dir, exist_ok=True)

    # Load all subtasks
    all_data = load_all_subtasks(data_dir, split=split, subtasks=subtasks)

    if not all_data:
        print("  ❌ No data loaded! Check data_dir and split.")
        return []

    # Convert all samples
    all_samples = []
    for subtask_name, df in sorted(all_data.items()):
        print(f"\n  Processing {subtask_name} ({len(df)} samples)...")
        subtask_samples = []
        for _, row in df.iterrows():
            try:
                sample = convert_sample(
                    row, subtask_name, image_output_dir,
                    use_prompt_column=True
                )
                subtask_samples.append(sample)
            except Exception as e:
                idx = row.get("idx", "unknown")
                print(f"    ⚠️  Error processing {idx}: {e}")
                continue

        all_samples.extend(subtask_samples)
        print(f"    ✅ Converted {len(subtask_samples)} samples for {subtask_name}")

    # Save full dataset
    output_path = os.path.join(output_dir, "blink_full_baseline.json")
    save_dataset(all_samples, output_path)

    # Also save per-subtask files
    subtask_groups = {}
    for s in all_samples:
        st = s["metadata"]["subtask"]
        subtask_groups.setdefault(st, []).append(s)

    for st_name, st_samples in sorted(subtask_groups.items()):
        st_path = os.path.join(output_dir, f"blink_{st_name.lower()}_baseline.json")
        save_dataset(st_samples, st_path)

    # Stats
    stats = make_stats(all_samples)
    summary = {
        "dataset":          "BLINK",
        "reference":        "BLINK: Multimodal Large Language Models Can See but Not Perceive (ECCV 2024)",
        "mode":             f"baseline (concatenated images + prompt, no emotion) — {split} split",
        "split":            split,
        "data_dir":         data_dir,
        "output_file":      "blink_full_baseline.json",
        "per_subtask_files": [f"blink_{st.lower()}_baseline.json" for st in sorted(subtask_groups.keys())],
        **stats,
    }
    summary_path = os.path.join(output_dir, "blink_baseline_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n  Stats:")
    print(f"    Total samples:          {stats['total']}")
    print(f"    Subtasks:               {stats['num_subtasks']}")
    print(f"    Concatenated (multi→1): {stats['concatenated_samples']}")
    print(f"    Answer distribution:    {stats['answer_distribution']}")
    print(f"    Original images dist:   {stats['num_images_original_distribution']}")
    print(f"\n    Per-subtask counts:")
    for st, cnt in sorted(stats["subtask_distribution"].items()):
        print(f"      {st:30s} {cnt}")

    # Show first few samples for verification
    print(f"\n  Sample preview:")
    for s in all_samples[:3]:
        m = s["metadata"]
        print(f"    [{s['id']}]")
        print(f"       Subtask: {m['subtask']}")
        print(f"       Q: {m['original_question'][:60]}...")
        print(f"       Options: {m['options_raw'][:80]}...")
        print(f"       GT: ({m['gt_answer']}) {m['gt_answer_text'][:40]}")
        print(f"       Original images: {m['num_images_original']} → {m['image_filenames'][:2]}...")
        print(f"       Concat image:    {m['concat_image']}")

    return all_samples


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Prepare BLINK benchmark — baseline (image + prompt + options)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
BLINK (Multimodal LLMs Can See but Not Perceive) tests visual perception.
3,807 samples across 14 subtasks. Each sample has 1-4 images and a
multiple-choice question.

14 subtasks:
  Art_Style, Counting, Forensic_Detection, Functional_Correspondence,
  IQ_Test, Jigsaw, Multi-view_Reasoning, Object_Localization,
  Relative_Depth, Relative_Reflectance, Semantic_Correspondence,
  Spatial_Relation, Visual_Correspondence, Visual_Similarity

Structure:
  original_data/blink/
      dataset/
          Art_Style/
              val-00000-of-00001.parquet
              test-00000-of-00001.parquet
          Counting/
              ...
          ...

Output:
  processed_data/blink_baseline/
      blink_full_baseline.json          (all samples)
      blink_art_style_baseline.json     (per-subtask)
      blink_counting_baseline.json
      ...
      blink_baseline_summary.json
      images/                           (extracted images)

Examples:
  python prepare_blink.py
  python prepare_blink.py --data_dir /path/to/blink
  python prepare_blink.py --output_dir /custom/output
  python prepare_blink.py --split test
  python prepare_blink.py --subtasks Art_Style Counting Jigsaw
        """,
    )

    parser.add_argument("--data_dir", type=str, default=DATA_DIR,
                        help=f"BLINK data directory (default: {DATA_DIR})")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: processed_data/blink_baseline)")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"],
                        help="Data split to process (default: val)")
    parser.add_argument("--subtasks", type=str, nargs="+", default=None,
                        help="Specific subtasks to process (default: all 14)")

    args = parser.parse_args()
    output_dir = args.output_dir or os.path.join(OUTPUT_BASE_DIR, "blink_baseline")

    # Validate subtasks if specified
    if args.subtasks:
        invalid = [s for s in args.subtasks if s not in ALL_SUBTASKS]
        if invalid:
            print(f"⚠️  Invalid subtasks: {invalid}")
            print(f"   Valid subtasks: {ALL_SUBTASKS}")
            return

    print(f"\n{'='*80}")
    print("BLINK DATA PREPARATION — BASELINE")
    print(f"{'='*80}")
    print(f"Data dir:    {args.data_dir}")
    print(f"Dataset dir: {os.path.join(args.data_dir, DATASET_DIR_NAME)}")
    print(f"Split:       {args.split}")
    print(f"Subtasks:    {args.subtasks or 'ALL (14)'}")
    print(f"Output dir:  {output_dir}")

    prepare_baseline(
        output_dir,
        data_dir=args.data_dir,
        split=args.split,
        subtasks=args.subtasks,
    )

    print(f"\n{'='*80}")
    print("✅ BLINK PREPARATION COMPLETE")
    print(f"{'='*80}")
    print(f"Output: {output_dir}")
    print(f"\nNext: add to inference.py:")
    print(f'  IMAGE_BASE_DIRS["blink_baseline"] = "{output_dir}"')
    print(f'  BENCHMARK_FINDINGS["blink"] = ["blink_baseline"]')


if __name__ == "__main__":
    main()
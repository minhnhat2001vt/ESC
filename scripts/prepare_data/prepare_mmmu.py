"""
MMMU Data Preparation Script — BASELINE ONLY (validation split)

MMMU evaluates multimodal models on college-level multi-discipline tasks.
Val split has 900 samples across 30 subjects and 6 disciplines.

Expected dataset structure (from HuggingFace: MMMU/MMMU, or local):
    original_data/mmmu/
        mmmu_val.json            ← validation split (900 samples)
        images/                  ← all images
            image_001.png, ...

JSON format (per sample):
    {
        "id": "validation_Accounting_1",
        "question": "Which of the following <image 1> ...",
        "options": ["optA", "optB", "optC", "optD"],
        "answer": "A",
        "image_1": "images/image_001.png",     ← up to image_7
        "image_2": null,
        ...
        "question_type": "multi-choice",       ← or "open"
        "topic_difficulty": "Hard",
        "subfield": "Financial Accounting",
        "subject": "Accounting"
    }

    Note: MMMU uses "multi-choice" (with hyphen), not "multi_choice".
    Images are referenced in question text as <image 1>, <image 2>, etc.
    For single-image samples, we replace <image 1> with <image>.

Question format for inference (multi-choice):
    <image>
    {question with <image N> placeholders resolved}
    (A) optA  (B) optB  (C) optC  (D) optD
    Answer with the option's letter from the given choices directly.

Question format for inference (open):
    <image>
    {question}
    Answer the question directly with a short response.

Output:
    processed_data/mmmu_baseline/
        mmmu_full_baseline.json
        mmmu_baseline_summary.json

Usage:
    python prepare_mmmu.py                       # single-image only (default, ~857 samples)
    python prepare_mmmu.py --multi_image         # all 900 samples
    python prepare_mmmu.py --data_dir /path/to/mmmu

Reference:
    MMMU: A Massive Multi-discipline Multimodal Understanding and
    Reasoning Benchmark for Expert AGI (CVPR 2024)
    https://mmmu-benchmark.github.io/
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

DATA_DIR = str(ORIGINAL_DATA_ROOT / "mmmu")
OUTPUT_BASE_DIR = str(PROCESSED_DATA_ROOT)

# MMMU 6 disciplines mapped from 30 subjects
SUBJECT_TO_DISCIPLINE = {
    "Accounting":                   "Business",
    "Economics":                    "Business",
    "Finance":                      "Business",
    "Manage":                       "Business",
    "Marketing":                    "Business",
    "Art":                          "Art & Design",
    "Art_Theory":                   "Art & Design",
    "Design":                       "Art & Design",
    "Music":                        "Art & Design",
    "Biology":                      "Science",
    "Chemistry":                    "Science",
    "Geography":                    "Science",
    "Math":                         "Science",
    "Physics":                      "Science",
    "Basic_Medical_Science":        "Health & Medicine",
    "Clinical_Medicine":            "Health & Medicine",
    "Diagnostics_and_Laboratory_Medicine": "Health & Medicine",
    "Pharmacy":                     "Health & Medicine",
    "Public_Health":                "Health & Medicine",
    "History":                      "Humanities & Social Science",
    "Literature":                   "Humanities & Social Science",
    "Psychology":                   "Humanities & Social Science",
    "Sociology":                    "Humanities & Social Science",
    "Architecture_and_Engineering": "Tech & Engineering",
    "Computer_Science":             "Tech & Engineering",
    "Electronics":                  "Tech & Engineering",
    "Energy_and_Power":             "Tech & Engineering",
    "Materials":                    "Tech & Engineering",
    "Mechanical_Engineering":       "Tech & Engineering",
}


# ============================================================================
# DATA DOWNLOAD + SETUP
# ============================================================================
def download_mmmu(data_dir):
    """
    Download MMMU validation split from HuggingFace and set up local directory.

    Source: MMMU/MMMU
    Creates:
        {data_dir}/mmmu_val.json
        {data_dir}/images/
    """
    try:
        from datasets import load_dataset, concatenate_datasets
    except ImportError:
        raise RuntimeError(
            "Please install the datasets library: pip install datasets"
        )

    print(f"\n  Downloading MMMU validation split from HuggingFace...")

    # Enumerate available configs dynamically instead of hardcoding
    from datasets import get_dataset_config_names
    try:
        all_configs = get_dataset_config_names("MMMU/MMMU")
        subjects = [c for c in all_configs if c not in ("default",)]
        print(f"  Found {len(subjects)} subject configs on HuggingFace")
    except Exception:
        # Fallback to hardcoded list
        subjects = list(SUBJECT_TO_DISCIPLINE.keys())
        print(f"  Using hardcoded {len(subjects)} subjects (could not enumerate configs)")

    all_samples = []

    for subj in subjects:
        try:
            ds = load_dataset("MMMU/MMMU", name=subj, split="validation")
            # Inject subject name since dataset rows don't contain it
            for sample in ds:
                sample_dict = dict(sample)
                if not sample_dict.get("subject"):
                    sample_dict["subject"] = subj
                all_samples.append(sample_dict)
            print(f"    {subj}: {len(ds)} samples")
        except Exception as e:
            print(f"    ⚠️  {subj}: failed ({e})")

    print(f"  Total downloaded: {len(all_samples)} samples")

    os.makedirs(data_dir, exist_ok=True)
    img_dir = os.path.join(data_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    records = []
    for idx, sample in enumerate(all_samples):
        record = {
            "id": sample.get("id", f"val_{idx}"),
            "question": sample.get("question", ""),
            "options": sample.get("options", []),
            "answer": sample.get("answer", ""),
            "question_type": sample.get("question_type", "multi-choice"),
            "subject": sample.get("subject", ""),
            "subfield": sample.get("subfield", ""),
            "topic_difficulty": sample.get("topic_difficulty", ""),
        }

        # Parse options if stored as string
        if isinstance(record["options"], str):
            import ast
            try:
                record["options"] = ast.literal_eval(record["options"])
            except (ValueError, SyntaxError):
                try:
                    record["options"] = json.loads(record["options"])
                except (json.JSONDecodeError, TypeError):
                    record["options"] = []

        # Save images (image_1 through image_7)
        for i in range(1, 8):
            img_key = f"image_{i}"
            image = sample.get(img_key, None)
            if image is not None:
                img_filename = f"{record['id']}_{i}.png"
                img_path = os.path.join(img_dir, img_filename)
                try:
                    image.save(img_path)
                    record[img_key] = f"images/{img_filename}"
                except Exception:
                    record[img_key] = None
            else:
                record[img_key] = None

        records.append(record)

        if (idx + 1) % 100 == 0:
            print(f"    Processed {idx + 1}/{len(all_samples)} samples...")

    # Save JSON
    json_path = os.path.join(data_dir, "mmmu_val.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"  ✅ Saved {len(records)} samples → {json_path}")
    print(f"  ✅ Images saved → {img_dir}")
    return records


# ============================================================================
# DATA LOADING
# ============================================================================
def load_mmmu(json_path):
    """Load MMMU JSON file."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples from {os.path.basename(json_path)}")
    return data


# ============================================================================
# CONVERSION
# ============================================================================
def _collect_image_paths(entry):
    """Collect all non-null image paths from image_1 through image_7."""
    paths = []
    for i in range(1, 8):
        img = entry.get(f"image_{i}", None)
        if img and str(img).strip() and str(img).lower() != "none":
            paths.append(str(img))
    return paths


def _resolve_image_placeholders(question, image_paths):
    """
    Replace <image N> placeholders in question text.

    For single-image samples: replace <image 1> with <image>.
    For multi-image: keep <image 1>, <image 2>, etc. as-is
    (the inference script handles multi-image models).
    """
    if len(image_paths) == 1:
        # Single image: replace <image 1> with just <image>
        question = question.replace("<image 1>", "<image>")
    # For multi-image, placeholders remain for the inference script to handle
    return question


def convert_sample(entry):
    """Convert one MMMU entry to the standardized inference format."""
    sample_id_raw = str(entry.get("id", ""))
    question = entry.get("question", "")
    options = entry.get("options", [])
    answer = entry.get("answer", "")
    question_type_raw = entry.get("question_type", "multi-choice")
    subject = entry.get("subject", "")
    subfield = entry.get("subfield", "")
    topic_difficulty = entry.get("topic_difficulty", "")

    # Parse options: HuggingFace MMMU stores them as a JSON-encoded string
    # e.g. "['$6.00', '$5.00', '$4.00', '$3.00']" instead of an actual list
    if isinstance(options, str):
        import ast
        try:
            options = ast.literal_eval(options)
        except (ValueError, SyntaxError):
            try:
                options = json.loads(options)
            except (json.JSONDecodeError, TypeError):
                options = []

    # Normalize question_type: "multi-choice" → "multi_choice"
    question_type = question_type_raw.replace("-", "_")

    # Collect image paths
    image_paths = _collect_image_paths(entry)

    # Resolve image placeholders in question text
    question_resolved = _resolve_image_placeholders(question, image_paths)

    # Build formatted question
    if question_type in ("multi_choice", "multiple_choice") and options:
        num_opts = len(options)
        letters = [chr(ord("A") + i) for i in range(num_opts)]
        options_dict = {l: str(o) for l, o in zip(letters, options)}
        options_lines = "\n".join(f"({l}) {o}" for l, o in zip(letters, options))

        formatted_q = (
            f"{question_resolved}\n"
            f"{options_lines}\n"
            f"Answer with the option's letter from the given choices directly."
        )
    else:
        # Open-ended question
        options_dict = {}
        formatted_q = (
            f"{question_resolved}\n"
            f"Answer the question directly with a short response."
        )

    # Build user message
    if len(image_paths) <= 1:
        # Single image or no image
        if "<image>" not in formatted_q:
            user_message = f"<image>\n{formatted_q}"
        else:
            user_message = formatted_q
    else:
        # Multi-image: don't prepend extra <image>
        user_message = formatted_q

    sample_id = f"mmmu_{sample_id_raw}"
    discipline = SUBJECT_TO_DISCIPLINE.get(subject, "Unknown")

    # GT answer letter for multi-choice, or text for open
    gt_letter = answer.upper() if question_type in ("multi_choice", "multiple_choice") and len(answer) == 1 else ""
    gt_answer = answer

    return {
        "id": sample_id,
        "image": image_paths if image_paths else [""],
        "conversations": [
            {"from": "user", "value": user_message}
        ],
        "metadata": {
            "scenario":            "mmmu",
            "image_type":          "mixed",
            "data_source":         "json",
            "question_id":         sample_id_raw,
            "original_question":   question,
            "formatted_question":  formatted_q,
            "question_type":       question_type,
            "answer_type":         "text",
            "options":             options_dict,
            "options_raw":         " ".join(f"({l}) {o}" for l, o in zip(
                [chr(ord("A")+i) for i in range(len(options))], options
            )) if options else "",
            "choices":             options,
            "num_images":          len(image_paths),
            "gt_answer":           gt_answer,
            "gt_answer_letter":    gt_letter,
            "image_filenames":     image_paths,
            "subject":             subject,
            "subfield":            subfield,
            "discipline":          discipline,
            "topic_difficulty":    topic_difficulty,
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
    subject_dist = Counter(s["metadata"]["subject"] for s in samples)
    discipline_dist = Counter(s["metadata"]["discipline"] for s in samples)
    difficulty_dist = Counter(s["metadata"]["topic_difficulty"] for s in samples)
    num_images_dist = Counter(s["metadata"]["num_images"] for s in samples)

    return {
        "total": len(samples),
        "question_type_distribution": dict(qtype_dist),
        "subject_distribution": dict(subject_dist),
        "discipline_distribution": dict(discipline_dist),
        "difficulty_distribution": dict(difficulty_dist),
        "num_images_distribution": dict(num_images_dist),
    }


# ============================================================================
# PREPARE
# ============================================================================
def prepare_baseline(output_dir, data_dir=DATA_DIR, single_image_only=True):
    img_label = "single-image only" if single_image_only else "all samples"

    print(f"\n{'='*80}")
    print(f"MMMU BASELINE PREPARATION (val split, {img_label})")
    print(f"{'='*80}")

    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(data_dir, "mmmu_val.json")

    # Download if data doesn't exist
    if not os.path.exists(json_path):
        print(f"  Data not found at {json_path}")
        print(f"  Downloading from HuggingFace...")
        download_mmmu(data_dir)

    data = load_mmmu(json_path)

    # Filter to single-image samples if requested
    if single_image_only:
        original_count = len(data)
        data = [e for e in data if len(_collect_image_paths(e)) <= 1]
        filtered_count = original_count - len(data)
        print(f"  Filtered to single-image: {len(data)} samples "
              f"({filtered_count} multi-image samples removed from {original_count})")

    # Verify some images
    missing = 0
    for entry in data[:10]:
        paths = _collect_image_paths(entry)
        for p in paths[:1]:
            img_path = os.path.join(data_dir, p)
            if not os.path.exists(img_path):
                missing += 1
    if missing > 0:
        print(f"  ⚠️  {missing} sample images not found in {data_dir}")
    else:
        print(f"  ✅ Sample images verified")

    # Convert
    samples = [convert_sample(e) for e in data]

    # Save
    output_path = os.path.join(output_dir, "mmmu_full_baseline.json")
    save_dataset(samples, output_path)

    # Stats
    stats = make_stats(samples)
    summary = {
        "dataset":      "MMMU",
        "reference":    "Yue et al., MMMU: A Massive Multi-discipline Multimodal "
                        "Understanding and Reasoning Benchmark (CVPR 2024)",
        "mode":         f"baseline (val split, {img_label})",
        "split":        "validation",
        "single_image_only": single_image_only,
        "data_dir":     data_dir,
        "output_file":  "mmmu_full_baseline.json",
        **stats,
    }
    summary_path = os.path.join(output_dir, "mmmu_baseline_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n  Stats:")
    print(f"    Total samples:       {stats['total']}")
    print(f"    Question types:      {stats['question_type_distribution']}")
    print(f"    Disciplines:         {stats['discipline_distribution']}")
    print(f"    Difficulty:          {stats['difficulty_distribution']}")
    print(f"    Images per sample:   {stats['num_images_distribution']}")

    # Preview
    print(f"\n  Sample preview:")
    for s in samples[:3]:
        m = s["metadata"]
        print(f"    [{s['id']}]")
        print(f"       Subject: {m['subject']} ({m['discipline']})")
        print(f"       Type: {m['question_type']}  GT: {m['gt_answer']}  Images: {m['num_images']}")

    return samples


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Prepare MMMU val baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
MMMU val: 900 samples across 30 subjects and 6 disciplines.
Two question types: multi_choice (4 options) and open.
Samples may contain 1-7 images interleaved with text.

By default, filters to single-image samples only (~857 samples),
which is standard practice for models without multi-image support
(see e.g. Cache-of-Thought, Geo-R1). Use --multi_image to keep all.

Structure:
  original_data/mmmu/
      mmmu_val.json
      images/

Output:
  processed_data/mmmu_baseline/
      mmmu_full_baseline.json
      mmmu_baseline_summary.json

Examples:
  python prepare_mmmu.py                       # single-image only (default)
  python prepare_mmmu.py --multi_image         # include multi-image samples
  python prepare_mmmu.py --data_dir /path/to/mmmu
        """,
    )
    parser.add_argument("--data_dir", type=str, default=DATA_DIR,
                        help="Path to MMMU data directory")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--multi_image", action="store_true",
                        help="Include multi-image samples. Default: single-image only "
                             "(filters out ~5%% of samples that reference multiple images, "
                             "following common practice for models without multi-image support)")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(OUTPUT_BASE_DIR, "mmmu_baseline")
    single_image_only = not args.multi_image

    print(f"\n{'='*80}")
    print("MMMU DATA PREPARATION — BASELINE")
    print(f"{'='*80}")
    print(f"Data dir:          {args.data_dir}")
    print(f"Output dir:        {output_dir}")
    print(f"Single-image only: {single_image_only}")

    prepare_baseline(output_dir, data_dir=args.data_dir, single_image_only=single_image_only)

    print(f"\n{'='*80}")
    print("✅ MMMU PREPARATION COMPLETE")
    print(f"{'='*80}")
    print(f"Output: {output_dir}")
    print(f"\nNext: add to inference.py:")
    print(f'  MMMU_DATA_DIR = "{args.data_dir}"')
    print(f'  IMAGE_BASE_DIRS["mmmu_baseline"] = "{args.data_dir}"')


if __name__ == "__main__":
    main()
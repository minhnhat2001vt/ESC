"""
Method 1: Detect-then-Regenerate (5-Step Method)
Data Preparation Script

Prepares the neutral dataset for Model A's initial generation.
If Model A results already exist, this step can be skipped entirely — 
inference_method1.py accepts pre-computed results directly via --model_a_results.

This script only needs to run if you do NOT already have neutral inference results.

Constant paths:
- Input:   original_data/vlsafe/harmlessness_examine.jsonl
- Images:  original_data/vlsafe/imgs/
- Output:  processed_data/vlsafe_method1/

Usage:
    # Prepare neutral dataset for Model A
    python prepare_method1.py

    # With custom output directory
    python prepare_method1.py --output_dir /path/to/output

    # Limit samples (for testing)
    python prepare_method1.py --max_samples 50
"""

import json
import os
import argparse
from pathlib import Path


# ============================================================================
# CONSTANT PATHS
# ============================================================================
DATA_DIR = "/home/emogenai4e/emo/cleaned/original_data/vlsafe"
JSONL_FILE = os.path.join(DATA_DIR, "harmlessness_examine.jsonl")
IMAGE_DIR = os.path.join(DATA_DIR, "imgs")
OUTPUT_DIR = "/home/emogenai4e/emo/cleaned/processed_data/vlsafe_method1"


# ============================================================================
# DATA LOADING
# ============================================================================
def load_vlsafe(jsonl_path=JSONL_FILE):
    """Load VLSafe JSONL. Each line: {image_id, caption, query}"""
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            entry["_index"] = line_num
            data.append(entry)
    print(f"Loaded {len(data)} samples from VLSafe")
    return data


# ============================================================================
# CONVERSION
# ============================================================================
def convert_to_neutral_sample(entry):
    """
    Convert one VLSafe entry to the inference format for Model A neutral generation.
    Format is compatible with inference.py's InferenceRunner.
    """
    image_id = entry["image_id"]
    query = entry["query"]
    caption = entry.get("caption", "")
    idx = entry["_index"]

    return {
        "id": f"vlsafe_{idx}_method1_neutral",
        "image": [f"/{image_id}"],
        "conversations": [
            {"from": "user", "value": f"<image>\n{query}"}
        ],
        "metadata": {
            "scenario": "vlsafe",
            "image_type": "real_photo",
            "question_id": idx,
            "original_question": query,
            "emotion_category": "neutral",
            "emotion_prompt_name": "",
            "emotion_prompt_text": "",
            "finding": "vlsafe_method1",
            "subject": "",
            "caption": caption,
            "image_id": image_id,
        },
    }


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Method 1: Prepare neutral dataset for Model A",
    )
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    # Validate
    if not os.path.exists(JSONL_FILE):
        print(f"VLSafe data not found: {JSONL_FILE}")
        return
    if not os.path.exists(IMAGE_DIR):
        print(f"Image directory not found: {IMAGE_DIR}")
        return

    # Load
    data = load_vlsafe()
    if args.max_samples:
        data = data[:args.max_samples]

    # Convert
    samples = [convert_to_neutral_sample(e) for e in data]

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "vlsafe_method1_neutral.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(samples)} samples → {output_path}")

    # Summary
    summary = {
        "method": "method1_detect_then_regenerate",
        "phase": "neutral_preparation",
        "total_samples": len(data),
        "output_file": os.path.basename(output_path),
        "note": "Run inference_method1.py with --model_a_results to use existing results, or run inference.py on this dataset first.",
    }
    summary_path = os.path.join(args.output_dir, "method1_prepare_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
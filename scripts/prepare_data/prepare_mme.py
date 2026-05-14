#!/usr/bin/env python3
"""
MME Data Preparation Script — BASELINE ONLY (no emotion).

Converts MME benchmark to inference format: [image + question] only.
Output is saved under processed_data/mme_baseline/ similar to other benchmark prepare scripts.

Expected input folder:
    original_data/mme/MME_Benchmark_release_version/
        artwork/
        celebrity/
        ...
Each task usually contains:
    images/
    questions_answers_YN/*.txt

Each QA txt typically contains 2 lines (one yes, one no):
    <question>\t<yes|no>

This script is robust to minor layout variants:
- QA txt files can be in `questions_answers_YN/` or directly inside task dir
- images can be in `images/` or directly inside task dir

Usage:
    python prepare_mme.py --full
    python prepare_mme.py --split
    python prepare_mme.py --all
    python prepare_mme.py --task artwork celebrity --split
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ============================================================================
# CONSTANT PATHS
# ============================================================================
DATA_DIR = "/workspace/original_data/mme"
MME_ROOT_DIR = os.path.join(DATA_DIR, "MME_Benchmark_release_version")
OUTPUT_BASE_DIR = "/workspace/processed_data"

# Canonical MME tasks (14)
CANONICAL_MME_TASKS = [
    "artwork",
    "celebrity",
    "code_reasoning",
    "color",
    "commonsense_reasoning",
    "count",
    "existence",
    "landmark",
    "numerical_calculation",
    "OCR",
    "position",
    "posters",
    "scene",
    "text_translation",
]

YESNO_INSTRUCTION = "Please answer yes or no."
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


# ============================================================================
# SMALL UTILS
# ============================================================================
def _safe_id(text: str) -> str:
    text = str(text)
    text = re.sub(r"[^a-zA-Z0-9_-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "sample"


def _norm_answer(ans: str) -> str:
    s = (ans or "").strip().lower()
    if s.startswith("yes"):
        return "yes"
    if s.startswith("no"):
        return "no"
    if s in {"y", "1", "true"}:
        return "yes"
    if s in {"n", "0", "false"}:
        return "no"
    return s


def _format_question(q: str) -> str:
    q = (q or "").strip()
    if not q:
        return YESNO_INSTRUCTION
    q_lower = q.lower()
    if "please answer yes or no" in q_lower:
        return q
    if not q.endswith((".", "?", "!", ":", ";")):
        q += "?"
    return f"{q} {YESNO_INSTRUCTION}"


def _choose_best_image_rel(candidates_rel: List[str]) -> str:
    if not candidates_rel:
        return ""

    def score(p: str):
        p_norm = p.replace("\\", "/")
        in_images = 0 if "/images/" in p_norm else 1
        return (in_images, len(p_norm), p_norm)

    return sorted(candidates_rel, key=score)[0]


# ============================================================================
# DISCOVERY / LOADING
# ============================================================================
def discover_tasks(mme_root_dir: str = MME_ROOT_DIR) -> List[str]:
    if not os.path.exists(mme_root_dir):
        raise FileNotFoundError(f"MME root dir not found: {mme_root_dir}")

    all_dirs = sorted([
        p.name for p in Path(mme_root_dir).iterdir()
        if p.is_dir() and not p.name.startswith(".")
    ])

    ordered = [t for t in CANONICAL_MME_TASKS if t in all_dirs]
    extras = [t for t in all_dirs if t not in ordered]
    return ordered + extras


def _build_image_index_for_task(task_dir: Path, data_root: Path) -> Dict[str, List[str]]:
    image_map: Dict[str, List[str]] = {}

    for p in task_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMAGE_EXTS:
            continue

        lower_parts = [x.lower() for x in p.parts]
        if any("question" in part and "answer" in part for part in lower_parts):
            continue

        rel = p.relative_to(data_root).as_posix()
        image_map.setdefault(p.stem, []).append(rel)

    return image_map


def _find_qa_txt_files(task_dir: Path) -> List[Path]:
    qa_files: List[Path] = []

    preferred_dirs = []
    for cand in [
        "questions_answers_YN",
        "questions_answers_yn",
        "question_answers_YN",
        "Questions_Answers_YN",
        "questions_answers",
    ]:
        d = task_dir / cand
        if d.exists() and d.is_dir():
            preferred_dirs.append(d)

    if preferred_dirs:
        for d in preferred_dirs:
            qa_files.extend(sorted(d.rglob("*.txt")))
        return sorted(set(qa_files))

    for p in task_dir.rglob("*.txt"):
        if p.name.lower() in {"readme.txt", "license.txt"}:
            continue
        qa_files.append(p)
    return sorted(set(qa_files))


def _parse_qaline(line: str) -> Optional[Tuple[str, str]]:
    line = line.strip()
    if not line:
        return None

    if "\t" in line:
        parts = [x.strip() for x in line.split("\t")]
        parts = [x for x in parts if x != ""]
        if len(parts) >= 2:
            answer = parts[-1]
            question = " ".join(parts[:-1]).strip()
            return question, answer

    m = re.match(r"^(.*?)[,\s]+(yes|no)\s*$", line, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip(), m.group(2).strip()

    return None


def load_mme_task(task_name: str,
                  mme_root_dir: str = MME_ROOT_DIR,
                  data_dir: str = DATA_DIR) -> List[dict]:
    data_root = Path(data_dir)
    task_dir = Path(mme_root_dir) / task_name
    if not task_dir.exists():
        raise FileNotFoundError(f"Task dir not found: {task_dir}")

    qa_files = _find_qa_txt_files(task_dir)
    if not qa_files:
        raise FileNotFoundError(f"No QA txt files found under task dir: {task_dir}")

    image_map = _build_image_index_for_task(task_dir, data_root)

    raw_entries: List[dict] = []
    skipped_no_image = 0
    skipped_bad_line = 0

    for qa_path in qa_files:
        qa_stem = qa_path.stem

        image_rel_candidates = image_map.get(qa_stem, [])
        if not image_rel_candidates:
            qa_stem_lower = qa_stem.lower()
            for stem, cands in image_map.items():
                if stem.lower() == qa_stem_lower:
                    image_rel_candidates = cands
                    break

        if not image_rel_candidates:
            skipped_no_image += 1
            continue

        image_rel = _choose_best_image_rel(image_rel_candidates)

        with open(qa_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        q_idx_for_image = 0
        for line_no, line in enumerate(lines, start=1):
            parsed = _parse_qaline(line)
            if parsed is None:
                if line.strip():
                    skipped_bad_line += 1
                continue

            question, answer = parsed
            q_idx_for_image += 1

            raw_entries.append({
                "_task": task_name,
                "_qa_file": qa_path.name,
                "_qa_relpath": qa_path.relative_to(data_root).as_posix(),
                "_qa_stem": qa_stem,
                "_line_no": line_no,
                "_q_idx_for_image": q_idx_for_image,
                "image_rel": image_rel,
                "image_filename": Path(image_rel).name,
                "image_stem": Path(image_rel).stem,
                "question": question.strip(),
                "answer": _norm_answer(answer),
            })

    print(
        f"[MME] Task '{task_name}': {len(raw_entries)} samples "
        f"(qa files: {len(qa_files)}, no-image skipped: {skipped_no_image}, bad-line skipped: {skipped_bad_line})"
    )
    return raw_entries


# ============================================================================
# CONVERSION: raw MME entry -> inference format (baseline)
# ============================================================================
def convert_sample(entry: dict, subset_name: str, running_idx: int) -> dict:
    task = entry["_task"]
    q = entry["question"]
    ans = entry["answer"]
    image_rel = entry["image_rel"]
    qa_stem = entry["_qa_stem"]

    formatted_q = _format_question(q)
    user_msg = f"<image>\n{formatted_q}"
    image_list = [f"/{image_rel}"]

    sample_id = f"mme_{_safe_id(task)}_{_safe_id(qa_stem)}_{entry['_q_idx_for_image']:02d}_{running_idx:06d}"

    return {
        "id": sample_id,
        "image": image_list,
        "conversations": [
            {"from": "user", "value": user_msg}
        ],
        "metadata": {
            "scenario": "mme",
            "image_type": "real_image",
            "subset": subset_name,
            "mme_task": task,
            "pair_group_id": f"{task}/{qa_stem}",
            "question_index_per_image": entry["_q_idx_for_image"],
            "qa_file": entry["_qa_file"],
            "qa_relpath": entry["_qa_relpath"],
            "qa_line_no": entry["_line_no"],
            "image_relpath": image_rel,
            "image_filename": entry["image_filename"],
            "image_stem": entry["image_stem"],
            "original_question": q,
            "formatted_question": formatted_q,
            "used_question": formatted_q,
            "question_type": "yes_no",
            "gt_answer": ans,
            "emotion_category": "neutral",
            "emotion_prompt_name": "",
            "emotion_prompt_text": "",
            "finding": "baseline",
        },
    }


# ============================================================================
# SAVE / SUMMARY
# ============================================================================
def save_dataset(samples: List[dict], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    print(f"  ✅ Saved {len(samples)} samples → {os.path.basename(output_path)}")


def build_stats(samples: List[dict]) -> dict:
    task_dist: Dict[str, int] = {}
    ans_dist: Dict[str, int] = {"yes": 0, "no": 0}
    pair_size_dist: Dict[str, int] = {}
    pair_sizes: Dict[str, int] = {}

    for s in samples:
        m = s["metadata"]
        task = m.get("mme_task", "unknown")
        ans = (m.get("gt_answer", "") or "").lower()
        pair_id = m.get("pair_group_id", "")

        task_dist[task] = task_dist.get(task, 0) + 1
        ans_dist[ans] = ans_dist.get(ans, 0) + 1
        if pair_id:
            pair_sizes[pair_id] = pair_sizes.get(pair_id, 0) + 1

    for _, sz in pair_sizes.items():
        k = str(sz)
        pair_size_dist[k] = pair_size_dist.get(k, 0) + 1

    return {
        "total_samples": len(samples),
        "total_unique_pairs": len(pair_sizes),
        "answer_distribution": ans_dist,
        "task_distribution": task_dist,
        "pair_size_distribution": pair_size_dist,
    }


# ============================================================================
# PREPARE FUNCTIONS
# ============================================================================
def prepare_full(task_entries_map: Dict[str, List[dict]], output_dir: str) -> List[str]:
    print(f"\n{'='*80}")
    print("MME BASELINE — Full")
    print(f"{'='*80}")

    all_entries: List[dict] = []
    for task in task_entries_map:
        all_entries.extend(task_entries_map[task])

    samples: List[dict] = []
    for i, e in enumerate(all_entries, start=1):
        samples.append(convert_sample(e, subset_name="full", running_idx=i))

    fname = "mme_full_baseline.json"
    fsum = "mme_full_baseline_summary.json"
    save_dataset(samples, os.path.join(output_dir, fname))

    summary = {
        "dataset": "MME",
        "subset": "full",
        "mode": "baseline (image + yes/no question, no emotion)",
        "data_dir": DATA_DIR,
        "mme_root_dir": MME_ROOT_DIR,
        "output_file": fname,
        **build_stats(samples),
    }
    with open(os.path.join(output_dir, fsum), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"  Created: {fname}")
    return [fname]


def prepare_by_task(task_entries_map: Dict[str, List[dict]], output_dir: str) -> List[str]:
    print(f"\n{'='*80}")
    print("MME BASELINE — By Task")
    print(f"{'='*80}")

    created: List[str] = []

    for task, entries in task_entries_map.items():
        if not entries:
            print(f"  ⚠️  Task '{task}' has 0 samples — skip")
            continue

        samples: List[dict] = []
        for i, e in enumerate(entries, start=1):
            samples.append(convert_sample(e, subset_name=f"task_{task}", running_idx=i))

        safe_task = _safe_id(task)
        fname = f"mme_{safe_task}_baseline.json"
        fsum = f"mme_{safe_task}_baseline_summary.json"

        save_dataset(samples, os.path.join(output_dir, fname))

        summary = {
            "dataset": "MME",
            "subset": f"task_{task}",
            "mme_task": task,
            "mode": "baseline (image + yes/no question, no emotion)",
            "data_dir": DATA_DIR,
            "mme_root_dir": MME_ROOT_DIR,
            "output_file": fname,
            **build_stats(samples),
        }
        with open(os.path.join(output_dir, fsum), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"  [{task}] {len(samples)} samples → {fname}")
        created.append(fname)

    return created


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Prepare MME dataset — baseline only (image + yes/no question, no emotion)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Constant paths:
  Input root:  {MME_ROOT_DIR}
  Output root: {os.path.join(OUTPUT_BASE_DIR, 'mme_baseline')}

Examples:
  python prepare_mme.py --full
  python prepare_mme.py --split
  python prepare_mme.py --all
  python prepare_mme.py --task artwork celebrity --split
  python prepare_mme.py --all --output_dir /custom/path
"""
    )

    parser.add_argument("--full", action="store_true",
                        help="Create one merged file: mme_full_baseline.json")
    parser.add_argument("--split", action="store_true",
                        help="Create one file per MME task")
    parser.add_argument("--all", action="store_true",
                        help="Create both full + per-task files")
    parser.add_argument("--task", nargs="+", default=None,
                        help="Only prepare selected task(s), e.g. --task artwork count OCR")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: processed_data/mme_baseline)")

    args = parser.parse_args()

    if not any([args.full, args.split, args.all]):
        parser.error("Specify at least one mode: --full, --split, or --all")

    output_dir = args.output_dir or os.path.join(OUTPUT_BASE_DIR, "mme_baseline")

    print(f"\n{'='*80}")
    print("MME DATA PREPARATION — BASELINE")
    print(f"{'='*80}")
    print(f"Data dir:      {DATA_DIR}")
    print(f"MME root dir:  {MME_ROOT_DIR}")
    print(f"Output dir:    {output_dir}")

    if not os.path.exists(MME_ROOT_DIR):
        print(f"\n❌ MME root directory not found: {MME_ROOT_DIR}")
        print("   Expected after unzip:")
        print("   original_data/mme/MME_Benchmark_release_version/<task>/...")
        return

    tasks = discover_tasks(MME_ROOT_DIR)
    if args.task:
        requested = args.task
        missing = [t for t in requested if t not in tasks]
        if missing:
            print(f"\n❌ Requested task(s) not found: {missing}")
            print(f"Available tasks: {tasks}")
            return
        tasks = requested

    print(f"Tasks to process ({len(tasks)}): {tasks}")

    task_entries_map: Dict[str, List[dict]] = {}
    for task in tasks:
        try:
            task_entries_map[task] = load_mme_task(task, MME_ROOT_DIR, DATA_DIR)
        except Exception as e:
            print(f"  ❌ Failed task '{task}': {e}")
            task_entries_map[task] = []

    task_entries_map = {k: v for k, v in task_entries_map.items() if v}
    if not task_entries_map:
        print("\n❌ No valid MME samples were loaded.")
        print("   Check folder structure inside each task (QA txt + images).")
        return

    os.makedirs(output_dir, exist_ok=True)
    created_files: List[str] = []

    if args.full or args.all:
        created_files.extend(prepare_full(task_entries_map, output_dir))

    if args.split or args.all:
        created_files.extend(prepare_by_task(task_entries_map, output_dir))

    print(f"\n{'='*80}")
    print("✅ MME PREPARATION COMPLETE")
    print(f"{'='*80}")
    print(f"Output dir:    {output_dir}")
    print(f"Files created: {created_files}")


if __name__ == "__main__":
    main()

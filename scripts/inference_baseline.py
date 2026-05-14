"""
Multi-Model Inference for Safety Benchmarks (v7 - Enhanced)
Features:
- InternVL2.5-8B support (replaces Gemini)
- Multi-GPU parallel processing
- Original v6 functionality preserved

New capabilities:
- Run multiple models in parallel on different GPUs
- Optimal batch size auto-detection per GPU
- GPU assignment with CUDA_VISIBLE_DEVICES

Usage:
    # Single model, single GPU (original behavior)
    python inference.py --model llava --finding finding3
    
    # Parallel execution on GPUs 0 and 1
    python inference.py --parallel --gpus 0 1 \
        --models llava internvl --finding finding3 --batch_size 32
    
    # Custom batch sizes per GPU
    python inference.py --parallel --gpus 0 1 \
        --models llava qwen2-vl --finding finding3 \
        --batch_sizes 32 24

    ## vlsafe
    --benchmark vlsafe
    python3 scripts/inference.py --model qwen2-vl --finding vlsafe_finding3 --batch_size 50  --test_mode

    ## mmsafety
    python inference.py --model internvl --finding finding3 --benchmark mmsafety --test_mode
"""

import json
import os
import re
import time
import torch
import multiprocessing as mp
from abc import ABC, abstractmethod
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import argparse
from datetime import datetime
import warnings
from model.base import BaseMLLM
from model.llava import LLaVA15Model, LLaVAModel
from model.cogvlm import CogVLM2Model
from model.internvl import InternVLModel
from model.llama import LLaMAVisionModel
from model.minicpm import MiniCPMModel
from model.pixtral import PixtralModel
from model.qwen import Qwen2VLModel
from model.llava_onevision import LLaVAOneVisionModel
from model.qwen3 import Qwen3VLModel
from model.gemma4 import Gemma4Model
from model.internvl3 import InternVL3Model
from model.qwen3_thinking import Qwen3VLThinkingModel
from utils import *
warnings.filterwarnings('ignore')


# ============================================================================
# CONSTANT PATHS
# ============================================================================
MMSAFETY_DATA_DIR = "/workspace/original_data/MMSafety"
MMSAFETY_IMAGE_DIR = "/workspace/original_data/MMSafety/imgs"
DATA_DIR = "/workspace/original_data/vlsafe"
VLSAFE_IMAGE_DIR = "/workspace/original_data/vlsafe/imgs"
PROCESSED_DIR = "/workspace/processed_data"
OUTPUT_BASE_DIR = "/workspace/results/infer"

FIGSTEP_IMAGE_DIR = "/workspace/original_data/figstep/images"
HALLUSION_DATA_DIR= "/workspace/original_data/hallusion_bench"
MMVET_IMAGE_DIR   = "/workspace/original_data/mm-vet/images"
RWQA_DATA_DIR     = "/workspace/original_data/RealWorldQA"
POPE_IMAGE_DIR    = "/workspace/original_data/pope/images"
MME_DATA_DIR = "/workspace/original_data/mme"
MMVP_BASELINE_DIR = "/workspace/original_data/mmvp"
BLINK_BASELINE_DIR = "/workspace/processed_data/blink_baseline"
MATHVISTA_DATA_DIR = "/workspace/original_data/mathvista"
MMSTAR_DATA_DIR = "/workspace/original_data/mmstar"

# Image base directories per finding
IMAGE_BASE_DIRS = {
    "finding1": MMSAFETY_DATA_DIR,
    "finding2": MMSAFETY_DATA_DIR,
    "finding3": MMSAFETY_DATA_DIR,
    "finding4": os.path.join(PROCESSED_DIR, "finding4"),
    "vlsafe_finding1": VLSAFE_IMAGE_DIR,
    "vlsafe_finding2": VLSAFE_IMAGE_DIR,
    "vlsafe_finding3": VLSAFE_IMAGE_DIR,
    "vlsafe_method": VLSAFE_IMAGE_DIR,
    # Benchmarks mới
    "figstep_baseline":   FIGSTEP_IMAGE_DIR,   
    "hallusion_baseline": HALLUSION_DATA_DIR, 
    "mmvet_baseline":     MMVET_IMAGE_DIR,    
    "rwqa_baseline":      RWQA_DATA_DIR,      
    "pope_baseline":      POPE_IMAGE_DIR, 
    "mme_baseline":       MME_DATA_DIR,    
    "mmvp_baseline":     MMVP_BASELINE_DIR,
    "blink_baseline":     BLINK_BASELINE_DIR,
    "mathvista_baseline_full": MATHVISTA_DATA_DIR,
    "mmstar_baseline":    MMSTAR_DATA_DIR,
}

# Benchmark → list of findings
BENCHMARK_FINDINGS = {
    "mmsafety":   ["finding1", "finding2", "finding3", "finding4"],
    "vlsafe":     ["vlsafe_finding1", "vlsafe_finding2", "vlsafe_finding3"],
    "figstep":    ["figstep_baseline"],
    "hallusion":  ["hallusion_baseline"],
    "mmvet":      ["mmvet_baseline"],
    "rwqa":       ["rwqa_baseline"],
    "pope":       ["pope_baseline"],
    "mme":        ["mme_baseline"],
    "mmvp":       ["mmvp_baseline"],
    "blink":      ["blink_baseline"],
    "mathvista": ["mathvista_baseline_full"],
    "mmstar":     ["mmstar_baseline"],
}


# ============================================================================
# MODEL REGISTRY (Updated with InternVL2.5-8B)
# ============================================================================
MODEL_REGISTRY = {
    "llava": {
        "name": "LLaVA-1.6-Vicuna-7B",
        "hf_id": "llava-hf/llava-v1.6-vicuna-7b-hf",
        "type": "llava",
        "max_tokens": 512,
    },
    "llava_1.5": {
        "name": "LLaVA-1.5-7B",
        "hf_id": "llava-hf/llava-1.5-7b-hf",
        "type": "llava15",
        "max_tokens": 512,
    },
    "llama-vision": {
        "name": "LLaMA-3.2-11B-Vision",
        "hf_id": "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "type": "llama_vision",
        "max_tokens": 512,
    },
    "qwen2-vl": {
        "name": "Qwen2-VL-7B",
        "hf_id": "Qwen/Qwen2-VL-7B-Instruct",
        "type": "qwen2_vl",
        "max_tokens": 512,
    },
    "internvl": {
        "name": "InternVL2.5-8B",
        "hf_id": "OpenGVLab/InternVL2_5-8B",
        "type": "internvl",
        "max_tokens": 512,
    },
    "pixtral": {
        "name": "Pixtral-12B",
        "hf_id": "mistral-community/pixtral-12b",
        "type": "pixtral",
        "max_tokens": 512,
    },
    "cogvlm2": {
        "name": "CogVLM2-LLaMA3-19B",
        "hf_id": "THUDM/cogvlm2-llama3-chat-19B",
        "type": "cogvlm2",
        "max_tokens": 512,
    },
    "minicpm": {
        "name": "MiniCPM-V-2.6",
        "hf_id": "openbmb/MiniCPM-V-2_6",
        "type": "minicpm",
        "max_tokens": 512,
    },
    "llava-ov": {
    "name": "LLaVA-OneVision-7B",
    "hf_id": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
    "type": "llava_onevision",
    "max_tokens": 512,
    },
    "qwen3-vl-8b": {
        "name": "Qwen3-VL-8B-Instruct",
        "hf_id": "Qwen/Qwen3-VL-8B-Instruct",
        "type": "qwen3_vl",
        "max_tokens": 512,
    },
    # T5 verifiers (small, ESC verifier role)
    "qwen3-vl-4b": {
        "name": "Qwen3-VL-4B-Instruct",
        "hf_id": "Qwen/Qwen3-VL-4B-Instruct",
        "type": "qwen3_vl",
        "max_tokens": 512,
    },
    "gemma4-e4b": {
        "name": "Gemma-4-E4B-it",
        "hf_id": "google/gemma-4-E4B-it",
        "type": "gemma4",
        "max_tokens": 512,
    },
    "internvl3-2b": {
        "name": "InternVL3-2B",
        "hf_id": "OpenGVLab/InternVL3-2B-hf",
        "type": "internvl3",
        "max_tokens": 512,
    },

    # T4 backbones (target model role)
    "internvl3-8b": {
        "name": "InternVL3-8B",
        "hf_id": "OpenGVLab/InternVL3-8B-hf",
        "type": "internvl3",
        "max_tokens": 512,
    },
    "qwen3-vl-8b-thinking": {
    "name": "Qwen3-VL-8B-Thinking",
    "hf_id": "Qwen/Qwen3-VL-8B-Thinking",
    "type": "qwen3_vl_thinking",
    "max_tokens": 8192,
    },
}

# ============================================================================
# MODEL FACTORY
# ============================================================================
class ModelFactory:
    """Create model instances from registry short names."""

    _model_classes = {
        "llava": LLaVAModel,
        "llava15": LLaVA15Model,
        "llama_vision": LLaMAVisionModel,
        "qwen2_vl": Qwen2VLModel,
        "internvl": InternVLModel,
        "pixtral": PixtralModel,
        "cogvlm2": CogVLM2Model,
        "minicpm": MiniCPMModel,
        "llava_onevision": LLaVAOneVisionModel,
        "qwen3_vl": Qwen3VLModel,
        "gemma4": Gemma4Model,
        "internvl3": InternVL3Model,
        "qwen3_vl_thinking": Qwen3VLThinkingModel,
    }

    @classmethod
    def create(cls, model_name: str, load_4bit: bool = True, device: str = "auto") -> BaseMLLM:
        if model_name not in MODEL_REGISTRY:
            available = list(MODEL_REGISTRY.keys())
            raise ValueError(f"Unknown model: {model_name}. Available: {available}")

        config = MODEL_REGISTRY[model_name]
        model_type = config["type"]

        if model_type not in cls._model_classes:
            raise ValueError(f"No implementation for model type: {model_type}")

        model_class = cls._model_classes[model_type]
        return model_class(config, load_4bit=load_4bit, device=device)

    @classmethod
    def list_available(cls) -> List[str]:
        return list(MODEL_REGISTRY.keys())


# ============================================================================
# INFERENCE RUNNER
# ============================================================================
class InferenceRunner:
    """
    Runs inference on prepared datasets.

    Features:
    - Batch processing with automatic fallback to sequential on error
    - Checkpointing: saves every N samples so progress survives crashes
    - Resume: skips already-processed sample IDs on restart
    - Builds result dicts with all metadata from prepare script
    """

    def __init__(
        self,
        model: BaseMLLM,
        image_base_dir: str,
        output_dir: str,
        batch_size: int = 4,
        checkpoint_interval: int = 50,
        max_samples: Optional[int] = None,
    ):
        self.model = model
        self.image_base_dir = Path(image_base_dir)
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.checkpoint_interval = checkpoint_interval
        self.max_samples = max_samples

        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------
    def _load_image(self, image_path: str) -> Image.Image:
        """Load a single image, resolving relative path against image_base_dir."""
        rel = image_path.lstrip("/")
        full_path = self.image_base_dir / rel

        if full_path.exists():
            try:
                return Image.open(full_path).convert("RGB")
            except Exception as e:
                print(f"   ⚠️ Error opening {full_path}: {e}")

        print(f"   ⚠️ Image not found: {full_path}")
        return Image.new("RGB", (224, 224), "white")

    def _load_images(self, image_paths: list) -> List[Image.Image]:
        """Load multiple images for multi-image models (e.g. Qwen2-VL)."""
        return [self._load_image(p) for p in image_paths]

    def _is_multi_image_model(self) -> bool:
        """Return True if the model natively supports multiple image inputs."""
        return "qwen" in self.model.name.lower()

    # ------------------------------------------------------------------
    # Sample helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_question(sample: Dict, multi_image: bool = False) -> str:
        """Extract question text, optionally using the multi-image conversation format."""
        if multi_image:
            conv_key = "conversations_multi"
        else:
            conv_key = "conversations"
        conversation = sample.get(conv_key, sample.get("conversations", [{}]))[0]
        # Strip all <image> tokens (single or repeated)
        text = conversation.get("value", "")
        text = re.sub(r'<image>\s*', '', text).strip()
        return text

    @staticmethod
    def _build_result(sample: Dict, response: str, model_name: str) -> Dict:
        """Build result dict preserving all metadata from prepare script."""
        metadata = sample.get("metadata", {})
        return {
            "id": sample["id"],
            "model": model_name,
            "scenario": metadata.get("scenario", "unknown"),
            "category": metadata.get("category", metadata.get("category_name", "")),
            "image_type": metadata.get("image_type", "unknown"),
            "question_id": metadata.get("question_id", "unknown"),
            "question_type": metadata.get("question_type", ""),
            "finding": metadata.get("finding", ""),
            "gt_answer": metadata.get("gt_answer", ""),
            "pope_split": metadata.get("pope_split", ""),
            "original_question": metadata.get("original_question", ""),
            "full_question": sample.get("conversations", [{}])[0].get("value", ""),
            "emotion_category": metadata.get("emotion_category", "neutral"),
            "emotion_prompt_name": metadata.get("emotion_prompt_name", ""),
            "emotion_prompt_text": metadata.get("emotion_prompt_text", ""),
            "emotion_location": metadata.get("emotion_location", "text"),
            "image_path": (sample.get("image_list") or sample.get("image", [""]))[0] if (sample.get("image_list") or sample.get("image")) else "",
            "image_list": sample.get("image_list", []),
            "response": response,
            "response_length": len(response),
        }

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------
    def _save_results(self, results: List[Dict], output_path: Path) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Single dataset processing (with resume + checkpoint)
    # ------------------------------------------------------------------
    def run_single_dataset(self, dataset_path: Path) -> List[Dict]:
        """
        Run inference on one JSON dataset file.
        Supports resuming from partial results and periodic checkpointing.
        """
        dataset_name = dataset_path.stem
        output_path = self.output_dir / f"results_{dataset_name}.json"

        print(f"\n{'='*60}")
        print(f"📂 Dataset: {dataset_name}")
        print(f"{'='*60}")

        # Load dataset
        with open(dataset_path, "r", encoding="utf-8") as f:
            samples = json.load(f)

        if self.max_samples:
            samples = samples[: self.max_samples]

        print(f"   Total samples: {len(samples)}")

        # Print metadata from first sample
        if samples and "metadata" in samples[0]:
            meta = samples[0]["metadata"]
            print(f"   Image type: {meta.get('image_type', 'N/A')}")
            print(f"   Question type: {meta.get('question_type', 'N/A')}")
            print(f"   Emotion category: {meta.get('emotion_category', 'N/A')}")
            if meta.get("emotion_location", "text") != "text":
                print(f"   Emotion location: {meta['emotion_location']}")

        # Resume: load existing results and skip processed IDs
        results: List[Dict] = []
        if output_path.exists():
            with open(output_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            print(f"   Resuming from {len(results)} already-processed samples")

        processed_ids = {r["id"] for r in results}
        remaining = [s for s in samples if s["id"] not in processed_ids]

        if not remaining:
            print("   ✅ All samples already processed!")
            return results

        print(f"   Processing {len(remaining)} remaining samples...")

        # Process in batches
        use_multi_image = self._is_multi_image_model()
        for i in tqdm(range(0, len(remaining), self.batch_size), desc=f"Inference ({self.model.name})"):
            batch = remaining[i : i + self.batch_size]

            try:
                if use_multi_image:
                    # Qwen2-VL: pass list of individual images per sample
                    images = [
                        self._load_images(s.get("image_list") or s.get("image", []))
                        if s.get("image_list") or s.get("image")
                        else [Image.new("RGB", (224, 224), "white")]
                        for s in batch
                    ]
                else:
                    images = [
                        self._load_image(s["image"][0]) if s.get("image") else Image.new("RGB", (224, 224), "white")
                        for s in batch
                    ]
                questions = [self._extract_question(s, multi_image=use_multi_image) for s in batch]

                if use_multi_image:
                    responses = self.model.generate_batch_multi(images, questions)
                else:
                    responses = self.model.generate_batch(images, questions)

                for sample, response in zip(batch, responses):
                    result = self._build_result(sample, response, self.model.name)
                    results.append(result)

            except Exception as e:
                print(f"\n   ⚠️ Batch error: {e}  — falling back to sequential")
                for sample in batch:
                    try:
                        if use_multi_image:
                            imgs = self._load_images(sample.get("image_list") or sample.get("image", []))
                            question = self._extract_question(sample, multi_image=True)
                            response = self.model.generate_multi(imgs, question)
                        else:
                            image = self._load_image(sample["image"][0]) if sample.get("image") else Image.new("RGB", (224, 224), "white")
                            question = self._extract_question(sample)
                            response = self.model.generate(image, question)
                        result = self._build_result(sample, response, self.model.name)
                        results.append(result)
                    except Exception as e2:
                        print(f"      Sample {sample['id']} error: {e2}")
                        result = self._build_result(sample, f"[Error: {e2}]", self.model.name)
                        results.append(result)

            # Periodic checkpoint
            if len(results) % self.checkpoint_interval == 0:
                self._save_results(results, output_path)

        # Final save
        self._save_results(results, output_path)

        successful = sum(1 for r in results if not r["response"].startswith("[Error"))
        print(f"   ✅ Complete: {successful}/{len(results)} successful")
        print(f"   💾 Saved to: {output_path}")

        return results


# ============================================================================
# FILE DISCOVERY
# ============================================================================
# def discover_finding_files(finding_name: str) -> List[Path]:
#     """
#     Discover all JSON data files for a given finding.
#     Looks in PROCESSED_DIR/finding{n}/ for finding{n}_*.json, excluding summaries.
#     """
#     data_dir = Path(PROCESSED_DIR) / finding_name

#     if not data_dir.exists():
#         print(f"Data directory not found: {data_dir}")
#         print(f"   Run prepare_mmsafety_emotion_v5.py --{finding_name} first.")
#         return []

#     files = sorted(data_dir.glob(f"{finding_name}_*.json"))
#     files = [f for f in files if "summary" not in f.name.lower()]
#     return files
def discover_finding_files(finding_name: str) -> List[Path]:
    """
    Discover all JSON data files for a given finding.
    - Primary pattern: {finding_name}_*.json  (mmsafety/vlsafe style)
    - Fallback:        *.json                 (figstep/hallusion/mmvet/rwqa/pope style)
    Luôn loại bỏ summary files.
    """
    data_dir = Path(PROCESSED_DIR) / finding_name

    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        print(f"   Run prepare script for '{finding_name}' first.")
        return []

    # Primary: mmsafety/vlsafe style (finding3_NEUTRAL.json, ...)
    files = sorted(data_dir.glob(f"{finding_name}_*.json"))
    files = [f for f in files if "summary" not in f.name.lower()]

    # Fallback: figstep/hallusion/mmvet/rwqa/pope style (tên file tự do)
    if not files:
        files = sorted(data_dir.glob("*.json"))
        files = [f for f in files if "summary" not in f.name.lower()]
        if files:
            print(f"  ℹ️  Fallback discovery for '{finding_name}': {[f.name for f in files]}")

    return files

# ============================================================================
# FINDING-LEVEL ORCHESTRATION
# ============================================================================
def run_finding(
    finding_name: str,
    model: BaseMLLM,
    batch_size: int = 4,
    checkpoint_interval: int = 50,
    max_samples: Optional[int] = None,
    skip_neutral: bool = False,
    image_dir_override: Optional[str] = None,
) -> Optional[Dict[str, List[Dict]]]:
    """
    Orchestrate inference for one finding:
    discover files → create runner → process each dataset → save summary.
    """
    finding_files = discover_finding_files(finding_name)
    if not finding_files:
        return None

    if skip_neutral:
        finding_files = [f for f in finding_files if "NEUTRAL" not in f.name.upper()]

    image_base_dir = image_dir_override or IMAGE_BASE_DIRS.get(finding_name, ".")
    # Output per model: results/infer/{model_short_name}/finding{n}/
    model_short = _model_short_name(model.name)
    output_dir = os.path.join(OUTPUT_BASE_DIR, model_short, finding_name)

    print(f"\n{'='*80}")
    print(f"{finding_name.upper()} INFERENCE — {model.name}")
    print(f"{'='*80}")
    print(f"   Image base dir: {image_base_dir}")
    print(f"   Output dir:     {output_dir}")
    print(f"   Batch size:     {batch_size}")
    print(f"   Files found:    {len(finding_files)}")
    for f in finding_files:
        print(f"     - {f.name}")
    print(f"{'='*80}\n")

    runner = InferenceRunner(
        model=model,
        image_base_dir=image_base_dir,
        output_dir=output_dir,
        batch_size=batch_size,
        checkpoint_interval=checkpoint_interval,
        max_samples=max_samples,
    )

    all_results: Dict[str, List[Dict]] = {}
    for idx, json_file in enumerate(finding_files, 1):
        print(f"\n[{idx}/{len(finding_files)}]", end="")
        results = runner.run_single_dataset(json_file)
        all_results[json_file.stem] = results

    # Save run summary
    summary = {
        "finding": finding_name,
        "model": model.name,
        "timestamp": datetime.now().isoformat(),
        "output_dir": str(output_dir),
        "datasets": {name: len(results) for name, results in all_results.items()},
        "total_samples": sum(len(r) for r in all_results.values()),
    }
    summary_path = Path(output_dir) / f"{finding_name}_inference_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*80}")
    print(f"✅ {finding_name.upper()} INFERENCE COMPLETE")
    print(f"   📊 Summary: {summary_path}")
    print(f"{'='*80}")

    return all_results


def _model_short_name(full_name: str) -> str:
    """Derive a filesystem-safe short name from the model display name."""
    return full_name.lower().replace(" ", "_").replace("-", "_").replace(".", "_")


# ============================================================================
# TEST MODE
# ============================================================================
def test_mode(finding_name: str, model: BaseMLLM):
    """
    Test mode: process 10 samples and show detailed output.
    """
    NUM_TEST_SAMPLES = 10  # Hard-coded number of samples to process
    
    finding_files = discover_finding_files(finding_name)
    if not finding_files:
        print(f"No data files found for {finding_name}")
        return

    test_file = finding_files[0]
    with open(test_file, "r", encoding="utf-8") as f:
        samples = json.load(f)

    if not samples:
        print("No samples in file")
        return

    # Process first 10 samples (or all if fewer than 10)
    test_samples = samples[:NUM_TEST_SAMPLES]
    # Prefer a sample with an image for the display section; fall back to first sample
    sample = next((s for s in test_samples if s.get("image")), test_samples[0])
    image_base_dir = IMAGE_BASE_DIRS.get(finding_name, ".")
    runner = InferenceRunner(
        model=model,
        image_base_dir=image_base_dir,
        output_dir="./test_output",
        batch_size=1,
    )

    print("\n" + "=" * 80)
    print("TEST MODE")
    print("=" * 80)
    print(f"  Finding: {finding_name}")
    print(f"  Model: {model.name}")
    print(f"  File: {test_file.name}")
    print(f"  Processing: {len(test_samples)} samples")
    print(f"  Display sample ID: {sample['id']}")

    # --- Image info ---
    print()
    print("=" * 80)
    print("FIRST VISUAL SAMPLE - IMAGE")
    print("=" * 80)
    if sample.get("image"):
        print(f"  Path: {sample['image'][0]}")
        img = runner._load_image(sample["image"][0])
        print(f"  Size: {img.size}")
        print(f"  Mode: {img.mode}")
    else:
        print("  (text-only sample — no image)")

    # --- Metadata ---
    if "metadata" in sample:
        print()
        print("=" * 80)
        print("FIRST SAMPLE - METADATA")
        print("=" * 80)
        for k, v in sample["metadata"].items():
            print(f"  {k}: {v}")

    # --- Conversations ---
    if "conversations" in sample:
        conversations = sample["conversations"]
        print()
        print("=" * 80)
        print("FIRST SAMPLE - CONVERSATION")
        print("=" * 80)
        for conv in conversations:
            print(f"  {conv.get('from', '?').upper()}: {conv.get('value', '')}")

    # --- Inference on all 10 samples ---
    print()
    print("=" * 80)
    print(f"RUNNING INFERENCE ON {len(test_samples)} SAMPLES")
    print("=" * 80)

    results = []
    for idx, sample in enumerate(test_samples, 1):
        print(f"\n  [{idx}/{len(test_samples)}] Processing sample {sample['id']}...")
        image_type = sample.get("metadata", {}).get("image_type", "unknown")
        print(f"      image_type: {image_type}")
        try:
            if sample.get("image"):
                image = runner._load_image(sample["image"][0]).convert("RGB")
            else:
                image = Image.new("RGB", (224, 224), "white")
                print(f"      (text-only — using blank image)")
            question = InferenceRunner._extract_question(sample)
            response = model.generate(image, question)
            result = InferenceRunner._build_result(sample, response, model.name)
            results.append(result)
            print(f"      ✓ Response length: {len(response)} chars")
        except Exception as e:
            print(f"      ✗ Error: {e}")
            result = InferenceRunner._build_result(sample, f"[Error: {e}]", model.name)
            results.append(result)

    print()
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"  Total samples: {len(results)}")
    successful = sum(1 for r in results if not r["response"].startswith("[Error"))
    print(f"  Successful: {successful}/{len(results)}")
    print(f"  Failed: {len(results) - successful}/{len(results)}")
    
    print()
    print("=" * 80)
    print("FIRST SAMPLE - DETAILED RESPONSE")
    print("=" * 80)
    print(f"  Response ({len(results[0]['response'])} chars):")
    print("-" * 80)
    print(results[0]['response'])
    print("-" * 80)

    # Save all results
    model_short = _model_short_name(model.name)
    test_output_path = f"./tests/{finding_name}/test_mode_{model_short}_results.json"
    os.makedirs(os.path.dirname(test_output_path), exist_ok=True)
    with open(test_output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 All {len(results)} results saved to: {test_output_path}")

    print("\n" + "=" * 80)
    print("TEST MODE COMPLETE")
    print("=" * 80)
    print("✅ Everything OK? Run without --test_mode for full inference.")
    print("❌ Something wrong? Check image paths and metadata above.")


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Multi-Model Inference for Safety Benchmarks (v7 - Enhanced)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single model (original behavior)
  python inference.py --model llava --finding finding3 --batch_size 32
  
  # Parallel: LLaVA on GPU 0, InternVL on GPU 1
  python inference.py --parallel --gpus 0 1 \\
      --models llava internvl --finding finding3 --batch_size 32
  
  # Parallel with different batch sizes
  python inference.py --parallel --gpus 0 1 \\
      --models llava qwen2-vl --finding finding3 \\
      --batch_sizes 32 24
  
  # Run all MM-SafetyBench in parallel
  python inference.py --parallel --gpus 0 1 \\
      --models llava internvl --benchmark mmsafety --batch_size 32
        """,
    )

    # Parallel execution
    parser.add_argument("--parallel", action="store_true",
                        help="Enable parallel execution across multiple GPUs")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0],
                        help="GPU IDs to use (e.g., --gpus 0 1)")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="Models for parallel execution (must match number of GPUs)")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=None,
                        help="Batch sizes per GPU (single value or one per model)")

    # Benchmark selection
    parser.add_argument("--benchmark", type=str, nargs="+", default=[],
                        choices=list(BENCHMARK_FINDINGS.keys()),
                        help="Benchmark name(s) (e.g. mmsafety vlsafe)")

    # Finding selection
    parser.add_argument("--finding", type=str, nargs="+", default=[],
                        help="Individual finding name(s)")

    # Optional image dir override
    parser.add_argument("--image_dir", type=str, default=None,
                        help="Override image base directory")

    # Model selection (for non-parallel mode)
    parser.add_argument("--model", type=str, default="llava",
                        choices=ModelFactory.list_available(),
                        help="Model to use (default: llava)")

    # Quantization
    parser.add_argument("--load_4bit", action="store_true", default=True,
                        help="Load in 4-bit quantization (default: True)")
    parser.add_argument("--no_4bit", action="store_true",
                        help="Disable 4-bit quantization")

    # Inference settings
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size (default: 4)")
    parser.add_argument("--max_tokens", type=int, default=512,
                        help="Max tokens to generate (default: 512)")
    parser.add_argument("--checkpoint", type=int, default=50,
                        help="Save checkpoint every N samples (default: 50)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples per dataset file (default: all)")

    # Flags
    parser.add_argument("--skip_neutral", action="store_true",
                        help="Skip neutral baseline files")
    parser.add_argument("--test_mode", action="store_true",
                        help="Test: process 1 sample, show image & metadata")
    parser.add_argument("--list_models", action="store_true",
                        help="List available models and exit")

    args = parser.parse_args()

    # --- List models ---
    if args.list_models:
        print("\n" + "=" * 70)
        print("AVAILABLE MODELS")
        print("=" * 70)
        for key, cfg in MODEL_REGISTRY.items():
            hf_id = cfg.get("hf_id", "N/A")
            print(f"  --model {key:15s}  →  {cfg['name']:25s}  ({hf_id})")
        print("\nBENCHMARKS:")
        for bname, findings in BENCHMARK_FINDINGS.items():
            print(f"  --benchmark {bname:12s}  →  {', '.join(findings)}")
        print("=" * 70)
        return

    # --- Resolve findings ---
    all_findings = []
    for bname in args.benchmark:
        all_findings.extend(BENCHMARK_FINDINGS[bname])
    all_findings.extend(args.finding)

    # Deduplicate
    seen = set()
    findings = []
    for f in all_findings:
        if f not in seen:
            seen.add(f)
            findings.append(f)

    if not findings:
        parser.error("Specify --benchmark and/or --finding")

    # Validate
    for f_name in findings:
        if f_name not in IMAGE_BASE_DIRS and not args.image_dir:
            print(f_name, IMAGE_BASE_DIRS, args.image_dir)
            parser.error(f"Unknown finding '{f_name}' and no --image_dir provided")

    load_4bit = not args.no_4bit

    # --- Parallel mode ---
    if args.parallel:
        if not args.models:
            parser.error("--models required for parallel execution")
        if not args.batch_sizes:
            args.batch_sizes = [args.batch_size]
        
        run_parallel(
            models=args.models,
            gpus=args.gpus,
            findings=findings,
            batch_sizes=args.batch_sizes,
            checkpoint_interval=args.checkpoint,
            max_samples=args.max_samples,
            skip_neutral=args.skip_neutral,
            image_dir_override=args.image_dir,
            load_4bit=load_4bit,
        )
        return

    # --- Single model mode (original behavior) ---
    model = ModelFactory.create(
        model_name=args.model,
        load_4bit=load_4bit,
        device="auto",
    )
    
    if args.max_tokens:
        model.max_tokens = args.max_tokens

    print("\n" + "=" * 70)
    print("EMOTIONAL SAFETY BENCHMARK — INFERENCE (v7)")
    print("=" * 70)
    print(f"  Model:       {model.name} ({args.model})")
    print(f"  4-bit:       {load_4bit}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  Checkpoint:  every {args.checkpoint} samples")
    print(f"  Input:       {PROCESSED_DIR}/")
    print(f"  Output:      {OUTPUT_BASE_DIR}/{_model_short_name(model.name)}/")
    if args.benchmark:
        print(f"  Benchmarks:  {', '.join(args.benchmark)}")
    print(f"  Findings:    {', '.join(findings)}")
    print("=" * 70)

    try:
        model.load()

        if args.test_mode:
            test_mode(findings[0], model)
            return

        for finding_name in findings:
            run_finding(
                finding_name=finding_name,
                model=model,
                batch_size=args.batch_size,
                checkpoint_interval=args.checkpoint,
                max_samples=args.max_samples,
                skip_neutral=args.skip_neutral,
                image_dir_override=args.image_dir,
            )

    finally:
        model.unload()

    print(f"\n{'='*70}")
    print("✅ ALL INFERENCE COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    # Required for multiprocessing on some platforms
    mp.set_start_method('spawn', force=True)
    main()
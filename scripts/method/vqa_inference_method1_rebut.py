"""
Method 1: Detect-then-Regenerate — VQA Inference Pipeline (FIXED VERSION 3)

For VQA benchmarks: POPE, RealWorldQA, MM-Vet, HallusionBench

FIXES APPLIED:
1. Step 1: Now preserves `full_question` from conversations[0]["value"]
2. Step 2: Uses `full_question` (not `original_question`) in judge prompt
3. Step 4: Uses `full_question` + emotion prompt for regeneration
4. Step 5: Uses `full_question` in decide prompt
5. **NEW FIX (v3)**: Properly handles <image> token — emotion prompt is inserted
   AFTER the <image> token, not before it.
6. **NEW FIX (v3)**: Handles HallusionBench mixed QA/VQA format:
   - visual_input="0" (text_only): No image, uses placeholder for VLM inference
   - visual_input="1" or "2": Normal VQA with image

This ensures:
- MCQ format instructions are preserved during regeneration
- The model sees the same question format (with options) in both original and regen
- Text-only samples (HallusionBench) don't crash due to empty image_path

A complete, self-contained pipeline that orchestrates all 6 steps:

    Step 1: Load Model A's neutral responses (from file or run inference)
    Step 2: Model B judges each (image + query + response) → yes / no
    Step 3: Route: safe → keep original; unsafe → needs regeneration
    Step 4: Model A regenerates unsafe samples with emotion prompt
    Step 5: Model B decides which response is safer (original vs regenerated)
    Step 6: Assemble final results with full decision trail

Models are loaded/unloaded sequentially to fit on a single GPU:
    Model A (generate) → unload → Model B (judge) → unload →
    Model A (regenerate) → unload → Model B (decide) → unload

Usage:
    # Using existing Model A results (recommended):
    python vqa_inference_method1_ver3.py \\
        --model_a_results /path/to/results.json \\
        --model_b llava_1.5 \\
        --model_a llava_1.5 \\
        --benchmark rwqa \\
        --batch_size 6 \\
        --selection_type fixed \\
        --quadrant negative_low \\
        --test_mode
    
    # HallusionBench (mixed QA/VQA with text-only samples):
    python vqa_inference_method1_ver3.py \\
        --model_a_results /path/to/hallusion_results.json \\
        --model_b llava_1.5 \\
        --model_a llava_1.5 \\
        --benchmark hallusion \\
        --selection_type fixed \\
        --quadrant negative_low \\
        --test_mode
"""

import json
import os
import re
import random
import sys
import torch
import argparse
from PIL import Image
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# Add parent directory (scripts/) to path so `model.*` imports resolve
# when running from scripts/method/
_SCRIPT_DIR = Path(__file__).resolve().parent
_PARENT_DIR = _SCRIPT_DIR.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

# Reuse model infrastructure from existing codebase
from model.base import BaseMLLM
from model.llava import LLaVA15Model, LLaVAModel
from model.llava_onevision import LLaVAOneVisionModel
from model.cogvlm import CogVLM2Model
from model.internvl import InternVLModel
from model.llama import LLaMAVisionModel
from model.minicpm import MiniCPMModel
from model.pixtral import PixtralModel
from model.qwen import Qwen2VLModel
from model.gemma3 import Gemma3Model


# ============================================================================
# CONSTANT PATHS
# ============================================================================
POPE_IMAGE_DIR = "/workspace/original_data/pope/images"
RWQA_IMAGE_DIR = "/workspace/original_data/RealWorldQA"
MMVET_IMAGE_DIR = "/workspace/original_data/mm-vet/images"
HALLUSION_IMAGE_DIR = "/workspace/original_data/hallusion_bench"
OUTPUT_BASE_DIR = "/workspace/results/method1"
MME_DATA_DIR = "/workspace/original_data/mme"
MMVP_DATA_DIR = "/workspace/original_data/mmvp"
BLINK_DATA_DIR = "/workspace/processed_data/blink_baseline"
MATHVISTA_DATA_DIR = "/workspace/original_data/mathvista"
MMSTAR_DATA_DIR = "/workspace/original_data/mmstar"
# ============================================================================
# MODEL REGISTRY
# ============================================================================
MODEL_REGISTRY = {
    # Target models (Model A candidates)
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
    "llava-ov": {
        "name": "LLaVA-OneVision-7B",
        "hf_id": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        "type": "llava_onevision",
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
    # Judge models (Model B candidates)
    "gemma3-12b": {
        "name": "Gemma-3-12B-IT",
        "hf_id": "google/gemma-3-12b-it",
        "type": "gemma3",
        "max_tokens": 512,
    },
    "gemma3-4b": {
    "name": "Gemma-3-4B-IT",
    "hf_id": "google/gemma-3-4b-it",
    "type": "gemma3",
    "max_tokens": 512,
    },
    "qwen2.5-vl": {
        "name": "Qwen2.5-VL-7B-Instruct",
        "hf_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "type": "qwen2_vl",
        "max_tokens": 512,
    },
    "gemma3-4b": {
    "name": "Gemma-3-4B-IT",
    "hf_id": "google/gemma-3-4b-it",
    "type": "gemma3",
    "max_tokens": 512,
    },
    "qwen2.5-vl-3b": {
        "name": "Qwen2.5-VL-3B-Instruct",
        "hf_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "type": "qwen2_vl",
        "max_tokens": 512,
    },
    "internvl2.5-2b": {
        "name": "InternVL2.5-2B",
        "hf_id": "OpenGVLab/InternVL2_5-2B",
        "type": "internvl",
        "max_tokens": 512,
    }
}

MODEL_CLASSES = {
    "llava": LLaVAModel,
    "llava15": LLaVA15Model,
    "llava_onevision": LLaVAOneVisionModel,
    "llama_vision": LLaMAVisionModel,
    "qwen2_vl": Qwen2VLModel,
    "internvl": InternVLModel,
    "pixtral": PixtralModel,
    "cogvlm2": CogVLM2Model,
    "minicpm": MiniCPMModel,
    "gemma3": Gemma3Model,
}


def create_model(model_name: str, load_4bit: bool = True, device: str = "auto") -> BaseMLLM:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    config = MODEL_REGISTRY[model_name]
    model_type = config["type"]
    if model_type not in MODEL_CLASSES:
        raise ValueError(f"No implementation for model type: {model_type}. Add it to MODEL_CLASSES.")
    return MODEL_CLASSES[model_type](config, load_4bit=load_4bit, device=device)


def model_short_name(full_name: str) -> str:
    return full_name.lower().replace(" ", "_").replace("-", "_").replace(".", "_")


# ============================================================================
# EMOTION PROMPTS (from Russell's Circumplex Model)
# ============================================================================
EMOTION_PROMPTS = {
    "POSITIVE_HIGH_01": "I'm feeling really happy and optimistic today!",
    "POSITIVE_HIGH_02": "I'm feeling very excited and delighted about this situation.",
    "POSITIVE_HIGH_03": "This brings me so much joy and excitement. Everything feels wonderful!",
    "POSITIVE_HIGH_04": "I'm feeling really enthusiastic and energized right now.",
    "POSITIVE_HIGH_05": "I'm feeling very happy and pleased about this.",
    "POSITIVE_HIGH_06": "I'm feeling extremely excited and astonished about this situation.",
    "NEGATIVE_HIGH_01": "I'm feeling very tense and alarmed about this situation.",
    "NEGATIVE_HIGH_02": "I'm feeling really afraid and distressed right now.",
    "NEGATIVE_HIGH_03": "This makes me feel angry and frustrated. I'm very upset.",
    "NEGATIVE_HIGH_04": "I'm feeling anxious and worried about this situation.",
    "NEGATIVE_HIGH_05": "I'm feeling extremely alarmed and distressed about this.",
    "NEGATIVE_HIGH_06": "I'm feeling really angry and annoyed right now.",
    "NEGATIVE_LOW_01": "I'm feeling really sad and disappointed right now.",
    "NEGATIVE_LOW_02": "I'm feeling very depressed and miserable about this situation.",
    "NEGATIVE_LOW_03": "This makes me feel gloomy and sad. I'm feeling really down.",
    "NEGATIVE_LOW_04": "I'm feeling really bored and tired right now.",
    "NEGATIVE_LOW_05": "I'm feeling very sad and gloomy about this.",
    "NEGATIVE_LOW_06": "I'm feeling deeply miserable and depressed about this situation.",
    "POSITIVE_LOW_01": "I'm feeling very calm and relaxed right now.",
    "POSITIVE_LOW_02": "I'm feeling really serene and content about this situation.",
    "POSITIVE_LOW_03": "I'm feeling very relaxed and at ease about this.",
    "POSITIVE_LOW_04": "This makes me feel calm and peaceful. I'm very serene.",
    "POSITIVE_LOW_05": "I'm feeling really content and satisfied right now.",
    "POSITIVE_LOW_06": "I'm feeling very peaceful and tranquil about this situation.",
}

QUADRANT_PREFIXES = ("POSITIVE_HIGH", "NEGATIVE_HIGH", "NEGATIVE_LOW", "POSITIVE_LOW")


# ============================================================================
# REBUTTAL ADDITION: PSYCH (EmotionPrompt by Li et al. 2023) + NEUTRAL prompts
# (Mirror of inference_method1_ver3.py — see that file for full docstring.)
# ============================================================================

PSYCH_PROMPTS = {
    "PSYCH_01": "Write your answer and give me a confidence score between 0-1 for your answer.",
    "PSYCH_02": "This is very important to my career.",
    "PSYCH_03": "You'd better be sure.",
    "PSYCH_04": "Are you sure?",
    "PSYCH_05": "Are you sure that's your final answer? It might be worth taking another look.",
    "PSYCH_07": "Are you sure that's your final answer? Believe in your abilities and strive for excellence. Your hard work will yield remarkable results.",
    "PSYCH_08": "Embrace challenges as opportunities for growth. Each obstacle you overcome brings you closer to success.",
    "PSYCH_09": "Stay focused and dedicated to your goals. Your consistent efforts will lead to outstanding achievements.",
    "PSYCH_10": "Take pride in your work and give it your best. Your commitment to excellence sets you apart.",
    "PSYCH_11": "Remember that progress is made one step at a time. Stay determined and keep moving forward.",
}

DEFAULT_NEUTRAL_TEXT = "Please reconsider your answer carefully and revise it if needed."


def select_emotion(selection_type: str, quadrant: str = None, seed: int = None) -> Tuple[str, str]:
    """
    Select an emotion prompt.

    Args:
        selection_type:
            - "random": uniform over all 24 quadrant prompts
            - "fixed":  pick from a specific quadrant (requires `quadrant` arg)
        quadrant: required when selection_type="fixed". One of:
            "positive_high", "negative_high", "negative_low", "positive_low"
        seed: optional seed for reproducibility per-sample

    Returns: (prompt_name, prompt_text)
    """
    if seed is not None:
        random.seed(seed)

    quad_items = [(k, v) for k, v in EMOTION_PROMPTS.items()
                  if k.startswith(QUADRANT_PREFIXES)]

    if selection_type == "random":
        return random.choice(quad_items)

    if selection_type == "fixed":
        if quadrant is None:
            raise ValueError("selection_type='fixed' requires --quadrant (e.g. 'negative_high')")
        prefix = quadrant.upper()
        if prefix not in QUADRANT_PREFIXES:
            raise ValueError(f"Unknown quadrant: '{quadrant}'. Choose from: {QUADRANT_PREFIXES}")
        candidates = [(k, v) for k, v in quad_items if k.startswith(prefix)]
        return random.choice(candidates)

    raise ValueError(f"selection_type must be 'random' or 'fixed', got: {selection_type}")


def select_prompt(
    prompt_source: str,
    selection_type: str = "random",
    quadrant: str = None,
    psych_id: str = None,
    neutral_text: str = DEFAULT_NEUTRAL_TEXT,
    seed: int = None,
) -> Tuple[str, str]:
    """
    Generalized prompt selector. See inference_method1_ver3.py for full docstring.
    """
    if seed is not None:
        random.seed(seed)

    if prompt_source == "emotion":
        return select_emotion(selection_type=selection_type, quadrant=quadrant)

    if prompt_source == "psychological":
        if selection_type == "random":
            return random.choice(list(PSYCH_PROMPTS.items()))
        if selection_type == "fixed":
            if psych_id is None:
                raise ValueError(
                    "prompt_source='psychological' with selection_type='fixed' "
                    "requires --psych_id (e.g. PSYCH_05)"
                )
            if psych_id not in PSYCH_PROMPTS:
                raise ValueError(
                    f"Unknown psych_id: {psych_id}. Available: {list(PSYCH_PROMPTS.keys())}"
                )
            return (psych_id, PSYCH_PROMPTS[psych_id])
        raise ValueError(f"selection_type must be 'random' or 'fixed', got: {selection_type}")

    if prompt_source == "neutral":
        return ("NEUTRAL", neutral_text)

    if prompt_source == "none":
        return ("NONE", "")

    raise ValueError(
        f"Unknown prompt_source: '{prompt_source}'. "
        f"Choose: emotion, psychological, neutral, none"
    )


# ============================================================================
# IMAGE LOADING
# ============================================================================
def load_image(image_path: str, image_base_dir: str) -> Optional[Image.Image]:
    """
    Load an image from path. Returns None for text-only samples (empty image_path).
    
    HallusionBench contains mixed QA/VQA:
    - visual_input="0" (text_only): image_path is empty → return None
    - visual_input="1" or "2": image_path has value → load image
    """
    # Handle empty image path (text-only samples like HallusionBench visual_input="0")
    if not image_path or image_path.strip() == "":
        return None
    
    rel = image_path.lstrip("/")
    full_path = Path(image_base_dir) / rel
    if full_path.exists():
        try:
            return Image.open(full_path).convert("RGB")
        except Exception as e:
            print(f"   ⚠️ Error opening {full_path}: {e}")
    print(f"   ⚠️ Image not found: {full_path}")
    return Image.new("RGB", (224, 224), "white")


def load_images(image_paths: list, image_base_dir: str) -> List[Optional[Image.Image]]:
    """Load multiple images for multi-image models (e.g. Qwen2-VL)."""
    return [load_image(p, image_base_dir) for p in image_paths]


def is_multi_image_model(model: BaseMLLM) -> bool:
    """Return True if the model natively supports multiple image inputs."""
    return "qwen" in model.name.lower()


# ============================================================================
# PROMPT TEMPLATES
# ============================================================================
def make_think_prompt(query: str, response: str) -> str:
    """Step 2: Model B judges a single response. Answer yes or no."""
    return f"""QUERY:
{query}

RESPONSE:
{response}

Is the RESPONSE correct, accurate and truthful? Answer only yes or no."""


def make_decide_prompt(query: str, response_a: str, response_b: str) -> str:
    """Step 5: Model B compares two responses and picks the more correct one."""
    return f"""QUERY:
{query}

RESPONSES:
A. {response_a}
B. {response_b}

Which RESPONSE is more correct, accurate and truthful? Answer with the option's letter from the given choices directly."""


# ============================================================================
# HELPER: Insert emotion into question with proper <image> token handling
# ============================================================================
def insert_emotion_into_question(base_question: str, emotion_text: str, location: str = "start") -> str:
    """
    Insert emotion prompt into question while preserving <image> token position.
    
    The <image> token must remain at the beginning for proper model processing.
    Emotion is inserted AFTER the <image> token.
    
    Args:
        base_question: The original question (may contain <image> token)
        emotion_text: The emotion prompt text to insert
        location: "start" = emotion before question text, "end" = emotion after question text
                  (both keep <image> at the very beginning)
    
    Returns:
        Properly formatted question with emotion inserted
    
    Examples:
        base_question: "<image>\nhow many pedestrians?\nA. 0\nB. 1\nC. 2\nAnswer with letter."
        emotion_text: "I'm feeling sad."
        location: "start"
        result: "<image>\nI'm feeling sad. how many pedestrians?\nA. 0\nB. 1\nC. 2\nAnswer with letter."
        
        location: "end"
        result: "<image>\nhow many pedestrians?\nA. 0\nB. 1\nC. 2\nAnswer with letter. I'm feeling sad."
    """
    # Check if question contains <image> token
    if '<image>' in base_question:
        # Remove <image> and any following newline/whitespace
        question_without_image = re.sub(r'^<image>\s*\n?\s*', '', base_question).strip()
        
        if location == "start":
            # Emotion comes right after <image>, before the question
            return f"<image>\n{emotion_text} {question_without_image}"
        elif location == "end":
            # Question comes after <image>, emotion at the end
            return f"<image>\n{question_without_image} {emotion_text}"
        else:
            raise ValueError(f"Unknown location: {location}. Supported: 'start', 'end'")
    else:
        # No <image> token — simple concatenation
        if location == "start":
            return f"{emotion_text} {base_question}".strip()
        elif location == "end":
            return f"{base_question} {emotion_text}".strip()
        else:
            raise ValueError(f"Unknown location: {location}. Supported: 'start', 'end'")


def strip_image_token(question: str) -> str:
    """Remove <image> token from question for text-only processing (e.g., judge prompts)."""
    return re.sub(r'<image>\s*\n?\s*', '', question).strip()


# ============================================================================
# ROBUST VERDICT PARSING
# ============================================================================
def parse_yes_no(raw: str) -> bool:
    """
    Parse a yes/no verdict from Model B's raw output.
    Handles many edge cases: "Yes", "yes", ".yes", "Yes.", "YES!",
    "yes, it is safe", "No, this is not safe", etc.

    Returns:
        True if "yes" (safe), False if "no" (unsafe).
        Defaults to False (unsafe) if ambiguous — conservative approach.
    """
    text = raw.strip().lower()
    # Remove leading punctuation/whitespace
    text = re.sub(r'^[\s\.\,\;\:\!\?\-\*\#\>]+', '', text)

    # Check for "yes" or "no" at the start (highest confidence)
    if re.match(r'^yes\b', text):
        return True
    if re.match(r'^no\b', text):
        return False

    # Check for negated safety words (before checking positive safety words)
    if re.search(r'\bnot\s+safe\b', text) or re.search(r'\bunsafe\b', text):
        return False
    if re.search(r'\bnot\s+ethical\b', text) or re.search(r'\bunethical\b', text):
        return False
    if re.search(r'\bnot\s+responsible\b', text) or re.search(r'\birresponsible\b', text):
        return False

    # Check anywhere in the response (for verbose answers)
    no_patterns = [
        r'\bno\b', r'\bunsafe\b', r'\bunethical\b', r'\birresponsible\b',
        r'\bharmful\b', r'\binappropriate\b', r'\bdangerous\b',
    ]
    yes_patterns = [
        r'\byes\b', r'\bsafe\b', r'\bresponsible\b', r'\bethical\b',
        r'\bappropriate\b', r'\bharmless\b',
    ]

    no_found = any(re.search(p, text) for p in no_patterns)
    yes_found = any(re.search(p, text) for p in yes_patterns)

    if no_found and not yes_found:
        return False
    if yes_found and not no_found:
        return True

    # Both or neither found — default to unsafe (conservative)
    return False


def parse_ab_choice(raw: str) -> str:
    """
    Parse an A/B choice from Model B's raw output.
    Handles: "A", "B", "a", "b", ".A", "A.", "Option A", "Response B",
    "A is safer", etc.

    Returns:
        "A" or "B". Defaults to "B" (regenerated) if ambiguous.
    """
    text = raw.strip()
    cleaned = re.sub(r'^[\s\.\,\;\:\!\?\-\*\#\>]+', '', text)

    # Direct letter at start (case-insensitive)
    if re.match(r'^[aA]\b', cleaned) or re.match(r'^[aA][\.\,\;\:\!]', cleaned):
        return "A"
    if re.match(r'^[bB]\b', cleaned) or re.match(r'^[bB][\.\,\;\:\!]', cleaned):
        return "B"

    lower = cleaned.lower()

    # "option A", "response A", "answer A"
    if re.search(r'\b(?:option|response|answer|choice)\s*a\b', lower):
        return "A"
    if re.search(r'\b(?:option|response|answer|choice)\s*b\b', lower):
        return "B"

    # Just "a" or "b" anywhere as standalone
    b_matches = list(re.finditer(r'\bb\b', lower))
    a_matches = list(re.finditer(r'\ba\b', lower))

    if b_matches and not a_matches:
        return "B"
    if a_matches and not b_matches:
        return "A"

    # Default: select regenerated response (B)
    return "B"


# ============================================================================
# STEP 1: LOAD MODEL A'S NEUTRAL RESPONSES
# ============================================================================
def step1_load_neutral_responses(
    model_a_results_path: Optional[str] = None,
    neutral_dataset_path: Optional[str] = None,
    model_a: Optional[BaseMLLM] = None,
    batch_size: int = 4,
    max_samples: Optional[int] = None,
    image_dir: str = ""
) -> List[Dict]:
    """
    Load or generate Model A's neutral (no emotion) responses.
    
    FIXED: Now preserves `full_question` for use in subsequent steps.
    """

    if model_a_results_path:
        print(f"\n{'='*70}")
        print("STEP 1: Loading existing Model A neutral responses")
        print(f"{'='*70}")
        print(f"  File: {model_a_results_path}")

        with open(model_a_results_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        if max_samples:
            results = results[:max_samples]

        # Ensure required fields exist
        for r in results:
            if "image_path" not in r:
                r["image_path"] = r.get("image", [""])[0] if isinstance(r.get("image"), list) else ""

            # is_multi_image: True if more than one individual image was used
            if "is_multi_image" not in r:
                r["is_multi_image"] = len(r.get("image_list", [])) > 1

            # FIX: Ensure full_question is available
            # Priority: full_question > used_question > conversations value > original_question
            if "full_question" not in r:
                if "used_question" in r:
                    r["full_question"] = r["used_question"]
                elif "conversations" in r and r["conversations"]:
                    # Extract from conversations if available
                    conv_value = r["conversations"][0].get("value", "")
                    r["full_question"] = conv_value  # Keep <image> token for now
                else:
                    # Fallback: use original_question but log warning
                    r["full_question"] = r.get("original_question", "")
                    if r["full_question"]:
                        print(f"  ⚠️  Sample {r.get('id', '?')}: Using original_question as full_question fallback")

        # Validate that full_question differs from original_question where expected
        mismatch_count = sum(1 for r in results 
                           if r.get("full_question") != r.get("original_question") 
                           and r.get("full_question") and r.get("original_question"))
        print(f"  Loaded {len(results)} results")
        print(f"  Samples with extended questions (MCQ format, etc.): {mismatch_count}")
        
        return results

    if neutral_dataset_path and model_a:
        print(f"\n{'='*70}")
        print("STEP 1: Running Model A neutral inference")
        print(f"{'='*70}")
        print(f"  Dataset: {neutral_dataset_path}")
        print(f"  Model:   {model_a.name}")

        with open(neutral_dataset_path, "r", encoding="utf-8") as f:
            samples = json.load(f)

        if max_samples:
            samples = samples[:max_samples]

        results = []
        for i in tqdm(range(0, len(samples), batch_size), desc="Step 1: Neutral inference"):
            batch = samples[i:i + batch_size]
            try:
                use_multi = is_multi_image_model(model_a)
                if use_multi:
                    images = [
                        load_images(s.get("image_list") or s.get("image", []), image_base_dir=image_dir)
                        for s in batch
                    ]
                    full_questions_raw = [s.get("conversations_multi", s["conversations"])[0]["value"] for s in batch]
                else:
                    images = [load_image(s["image"][0], image_base_dir=image_dir) for s in batch]
                    full_questions_raw = [s["conversations"][0]["value"] for s in batch]

                # FIX: Extract full_question from conversations (keep <image> token for reference)
                # For actual inference, strip <image> token
                full_questions_for_inference = [strip_image_token(q) for q in full_questions_raw]

                if use_multi:
                    responses = model_a.generate_batch_multi(images, full_questions_for_inference)
                else:
                    responses = model_a.generate_batch(images, full_questions_for_inference)

                for sample, full_q_raw, response in zip(batch, full_questions_raw, responses):
                    meta = sample.get("metadata", {})
                    image_path = (sample.get("image_list") or sample.get("image", [""]))[0] if use_multi else sample["image"][0]
                    results.append({
                        "id": sample["id"],
                        "model": model_a.name,
                        "original_question": meta.get("original_question", ""),
                        "full_question": full_q_raw,  # FIX: Store full_question with <image> token
                        "image_path": image_path,
                        "image_list": sample.get("image_list", []),  # store for later steps
                        "is_multi_image": use_multi and len(sample.get("image_list", [])) > 1,
                        "image_id": meta.get("image_id", ""),
                        "response": response,
                        "emotion_category": "neutral",
                        # Preserve additional metadata for analysis
                        "scenario": meta.get("scenario", ""),
                        "image_type": meta.get("image_type", ""),
                        "question_id": meta.get("question_id", ""),
                        "question_type": meta.get("question_type", ""),
                        "category": meta.get("category", ""),
                        "gt_answer": meta.get("gt_answer", ""),
                    })
            except Exception as e:
                print(f"\n  ⚠️ Batch error: {e}, falling back to sequential")
                for sample in batch:
                    try:
                        use_multi = is_multi_image_model(model_a)
                        if use_multi:
                            imgs = load_images(sample.get("image_list") or sample.get("image", []), image_base_dir=image_dir)
                            full_q_raw = sample.get("conversations_multi", sample["conversations"])[0]["value"]
                            full_q_for_inference = strip_image_token(full_q_raw)
                            response = model_a.generate_multi(imgs, full_q_for_inference)
                        else:
                            image = load_image(sample["image"][0], image_base_dir=image_dir)
                            full_q_raw = sample["conversations"][0]["value"]
                            full_q_for_inference = strip_image_token(full_q_raw)
                            response = model_a.generate(image, full_q_for_inference)
                        meta = sample.get("metadata", {})
                        image_path = (sample.get("image_list") or sample.get("image", [""]))[0] if use_multi else sample["image"][0]
                        results.append({
                            "id": sample["id"],
                            "model": model_a.name,
                            "original_question": meta.get("original_question", ""),
                            "full_question": full_q_raw,
                            "image_path": image_path,
                            "image_list": sample.get("image_list", []),
                            "is_multi_image": use_multi and len(sample.get("image_list", [])) > 1,
                            "image_id": meta.get("image_id", ""),
                            "response": response,
                            "emotion_category": "neutral",
                            "scenario": meta.get("scenario", ""),
                            "image_type": meta.get("image_type", ""),
                            "question_id": meta.get("question_id", ""),
                            "question_type": meta.get("question_type", ""),
                            "category": meta.get("category", ""),
                            "gt_answer": meta.get("gt_answer", ""),
                        })
                    except Exception as e2:
                        print(f"    Sample {sample['id']} error: {e2}")
                        meta = sample.get("metadata", {})
                        results.append({
                            "id": sample["id"],
                            "model": model_a.name,
                            "original_question": meta.get("original_question", ""),
                            "full_question": sample["conversations"][0]["value"],
                            "image_path": sample["image"][0],
                            "image_list": sample.get("image_list", []),
                            "is_multi_image": False,
                            "response": f"[Error: {e2}]",
                            "emotion_category": "neutral",
                        })

        print(f"  ✅ Generated {len(results)} neutral responses")
        return results

    raise ValueError("Provide either --model_a_results or (--neutral_dataset + --model_a)")


# ============================================================================
# STEP 2: MODEL B JUDGES SAFETY (yes/no)
# ============================================================================
def step2_judge_safety(
    neutral_results: List[Dict],
    model_b: BaseMLLM,
    batch_size: int = 4,
    checkpoint_path: Optional[str] = None,
    image_dir: str = "",
    args = None
) -> List[Dict]:
    """
    Model B judges each (image + query + response) as safe (yes) or unsafe (no).
    
    FIXED: Uses `full_question` instead of `original_question`.
    """
    print(f"\n{'='*70}")
    print("STEP 2: Model B judges correctness (yes/no)")
    print(f"{'='*70}")
    print(f"  Model B: {model_b.name}")
    print(f"  Samples: {len(neutral_results)}")

    judged_results = []
    processed_ids = set()
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            judged_results = json.load(f)
        processed_ids = {r["id"] for r in judged_results}
        print(f"  Resuming from {len(judged_results)} already-judged samples")

    remaining = [r for r in neutral_results if r["id"] not in processed_ids]
    if not remaining:
        print("  ✅ All samples already judged")
        return judged_results

    print(f"  Processing {len(remaining)} remaining samples...")

    for i in tqdm(range(0, len(remaining), batch_size), desc="Step 2: Judging"):
        batch = remaining[i:i + batch_size]

        try:
            # Handle mixed QA/VQA: some samples may not have images (HallusionBench text_only)
            images = [load_image(r["image_path"], image_base_dir=image_dir) for r in batch]
            
            # FIX: Use full_question (stripped of <image> token for text prompt)
            judge_prompts = [
                make_think_prompt(query=strip_image_token(r["full_question"]), response=r["response"])
                for r in batch
            ]
            
            # For text-only samples (image is None), we need to handle them separately
            # or use a placeholder image since most VLMs require an image input
            images_for_inference = []
            for img in images:
                if img is None:
                    # Create placeholder for text-only samples
                    images_for_inference.append(Image.new("RGB", (224, 224), "white"))
                else:
                    images_for_inference.append(img)
            
            verdicts = model_b.generate_batch(images_for_inference, judge_prompts)

            for r, verdict in zip(batch, verdicts):
                is_safe = parse_yes_no(verdict)
                judged_results.append({
                    **r,
                    "judge_model": model_b.name,
                    "judge_verdict_raw": verdict.strip(),
                    "judge_is_correct": is_safe,
                })

        except Exception as e:
            print(f"\n  ⚠️ Batch error: {e}, falling back to sequential")
            for r in batch:
                try:
                    image = load_image(r["image_path"], image_base_dir=image_dir)
                    # Use placeholder for text-only samples
                    if image is None:
                        image = Image.new("RGB", (224, 224), "white")
                    prompt = make_think_prompt(query=strip_image_token(r["full_question"]), response=r["response"])
                    verdict = model_b.generate(image, prompt)
                    is_safe = parse_yes_no(verdict)
                    judged_results.append({
                        **r,
                        "judge_model": model_b.name,
                        "judge_verdict_raw": verdict.strip(),
                        "judge_is_correct": is_safe,
                    })
                except Exception as e2:
                    print(f"    Sample {r['id']} error: {e2}")
                    judged_results.append({
                        **r,
                        "judge_model": model_b.name,
                        "judge_verdict_raw": f"[Error: {e2}]",
                        "judge_is_correct": False,
                    })

        if checkpoint_path and len(judged_results) % 100 == 0:
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(judged_results, f, indent=2, ensure_ascii=False)

    if checkpoint_path:
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(judged_results, f, indent=2, ensure_ascii=False)

    safe_count = sum(1 for r in judged_results if r["judge_is_correct"])
    unsafe_count = len(judged_results) - safe_count
    print(f"  ✅ Judging complete: {safe_count} safe (yes), {unsafe_count} unsafe (no)")

    return judged_results


# ============================================================================
# STEP 3: ROUTE
# ============================================================================
def step3_route(judged_results: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Split results into safe (keep) and unsafe (need regeneration)."""
    print(f"\n{'='*70}")
    print("STEP 3: Routing")
    print(f"{'='*70}")

    safe_results = [r for r in judged_results if r["judge_is_correct"]]
    unsafe_results = [r for r in judged_results if not r["judge_is_correct"]]

    print(f"  Correct (keep original):    {len(safe_results)}")
    print(f"  Incorrect (regenerate):     {len(unsafe_results)}")
    print(f"  Regeneration rate:       {len(unsafe_results) / len(judged_results):.1%}")

    return safe_results, unsafe_results


# ============================================================================
# STEP 4: MODEL A REGENERATES WITH EMOTION
# ============================================================================
def step4_regenerate_with_emotion(
    unsafe_results: List[Dict],
    model_a: BaseMLLM,
    selection_type: str = "random",
    quadrant: str = None,
    batch_size: int = 4,
    checkpoint_path: Optional[str] = None,
    image_dir: str = "",
    location: str = "start",
    multiple_emotion: int = 1,
    # ── REBUTTAL ADDITION ──────────────────────────────────────────────
    prompt_source: str = "emotion",
    psych_id: str = None,
    neutral_text: str = DEFAULT_NEUTRAL_TEXT,
    # ───────────────────────────────────────────────────────────────────
) -> List[Dict]:
    """
    Regenerate unsafe samples by injecting prompt(s) into the query.

    FIXED v2: Uses `full_question` instead of `original_question`.
    FIXED v3: Properly handles <image> token — emotion is inserted AFTER <image>.
    REBUTTAL ADDITION: prompt_source controls what gets injected. See
    inference_method1_ver3.py for the full description.
    """
    if not unsafe_results:
        print(f"\n{'='*70}")
        print("STEP 4: No unsafe samples — skipping regeneration")
        print(f"{'='*70}")
        return []

    if multiple_emotion < 1:
        raise ValueError("--multiple_emotion must be >= 1")

    print(f"\n{'='*70}")
    print("STEP 4: Model A regenerates")
    print(f"{'='*70}")
    print(f"  Model A:        {model_a.name}")
    print(f"  Samples:        {len(unsafe_results)}")
    print(f"  Prompt source:  {prompt_source}")
    print(f"  Selection type: {selection_type}")
    print(f"  Location:       {location}")
    print(f"  #Prompts:       {multiple_emotion}")
    if prompt_source == "emotion" and quadrant:
        print(f"  Quadrant:       {quadrant}")
    if prompt_source == "psychological" and psych_id:
        print(f"  Psych id:       {psych_id}")
    if prompt_source == "neutral":
        print(f"  Neutral text:   {neutral_text!r}")
    if prompt_source == "none":
        print(f"  (no text inserted — verifier-loop only baseline)")

    regen_results = []
    processed_ids = set()
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            regen_results = json.load(f)
        processed_ids = {r["id"] for r in regen_results}
        print(f"  Resuming from {len(regen_results)} already-regenerated samples")

    remaining = [r for r in unsafe_results if r["id"] not in processed_ids]
    if not remaining:
        print("  ✅ All samples already regenerated")
        return regen_results

    # Pre-select prompts (deterministic).
    random.seed(42)
    sample_emotions: Dict[str, List[Tuple[str, str]]] = {}
    for r in unsafe_results:
        prompts = [
            select_prompt(
                prompt_source=prompt_source,
                selection_type=selection_type,
                quadrant=quadrant,
                psych_id=psych_id,
                neutral_text=neutral_text,
            )
            for _ in range(multiple_emotion)
        ]
        sample_emotions[r["id"]] = prompts

    print(f"  Processing {len(remaining)} remaining samples...")

    for i in tqdm(range(0, len(remaining), batch_size), desc="Step 4: Regenerating"):
        batch = remaining[i:i + batch_size]

        try:
            use_multi = is_multi_image_model(model_a)
            if use_multi:
                images = [
                    load_images(r.get("image_list", [r["image_path"]]), image_base_dir=image_dir)
                    if r.get("image_path") or r.get("image_list") else None
                    for r in batch
                ]
            else:
                images = [load_image(r["image_path"], image_base_dir=image_dir) for r in batch]

            questions = []
            batch_emotions = []
            is_text_only_flags = []

            for r, img in zip(batch, images):
                prompts = sample_emotions[r["id"]]
                names = [p[0] for p in prompts]
                texts = [p[1] for p in prompts]
                emo_concat = " ".join(texts).strip()

                # Check if this is a text-only sample
                is_text_only = (img is None) or (isinstance(img, list) and all(i is None for i in img))
                is_text_only_flags.append(is_text_only)

                if emo_concat:
                    # FIX v3: Use helper function to properly handle <image> token
                    q_with_emotion = insert_emotion_into_question(
                        base_question=r['full_question'],
                        emotion_text=emo_concat,
                        location=location
                    )
                else:
                    # prompt_source="none" — regenerate with the same query, no insertion
                    q_with_emotion = r['full_question']
                # Strip <image> for actual inference (model receives image separately)
                q_for_inference = strip_image_token(q_with_emotion)

                questions.append(q_for_inference)
                batch_emotions.append((names, texts, emo_concat, q_with_emotion))

            if use_multi:
                # Replace None with placeholder list for text-only samples
                images_for_inference = [
                    img if img and not all(i is None for i in img)
                    else [Image.new("RGB", (224, 224), "white")]
                    for img in images
                ]
                responses = model_a.generate_batch_multi(images_for_inference, questions)
            else:
                images_for_inference = [
                    img if img is not None else Image.new("RGB", (224, 224), "white")
                    for img in images
                ]
                responses = model_a.generate_batch(images_for_inference, questions)

            for r, response, (names, texts, emo_concat, q_with_emotion), is_text_only in zip(
                batch, responses, batch_emotions, is_text_only_flags
            ):
                regen_results.append({
                    **r,
                    "response_original": r["response"],
                    "response_regen": response,
                    "emotion_prompt_name": names,
                    "emotion_prompt_text": texts,
                    "emotion_prompt_concat": emo_concat,
                    "emotion_location": location,
                    "regen_question_full": q_with_emotion,  # For debugging
                    "is_text_only": is_text_only,  # Track text-only samples
                })

        except Exception as e:
            print(f"\n  ⚠️ Batch error: {e}, falling back to sequential")
            for r in batch:
                try:
                    use_multi = is_multi_image_model(model_a)
                    if use_multi:
                        imgs = load_images(r.get("image_list", [r["image_path"]]), image_base_dir=image_dir)
                        is_text_only = all(i is None for i in imgs)
                        if is_text_only:
                            imgs = [Image.new("RGB", (224, 224), "white")]
                    else:
                        img = load_image(r["image_path"], image_base_dir=image_dir)
                        is_text_only = (img is None)
                        imgs = img if img is not None else Image.new("RGB", (224, 224), "white")
                    
                    prompts = sample_emotions[r["id"]]
                    names = [p[0] for p in prompts]
                    texts = [p[1] for p in prompts]
                    emo_concat = " ".join(texts).strip()

                    if emo_concat:
                        q_with_emotion = insert_emotion_into_question(
                            base_question=r['full_question'],
                            emotion_text=emo_concat,
                            location=location
                        )
                    else:
                        q_with_emotion = r['full_question']
                    q_for_inference = strip_image_token(q_with_emotion)

                    if use_multi:
                        response = model_a.generate_multi(imgs, q_for_inference)
                    else:
                        response = model_a.generate(imgs, q_for_inference)

                    regen_results.append({
                        **r,
                        "response_original": r["response"],
                        "response_regen": response,
                        "emotion_prompt_name": names,
                        "emotion_prompt_text": texts,
                        "emotion_prompt_concat": emo_concat,
                        "emotion_location": location,
                        "regen_question_full": q_with_emotion,
                        "is_text_only": is_text_only,
                    })
                except Exception as e2:
                    print(f"    Sample {r['id']} error: {e2}")
                    regen_results.append({
                        **r,
                        "response_original": r["response"],
                        "response_regen": f"[Error: {e2}]",
                        "emotion_prompt_name": [],
                        "emotion_prompt_text": [],
                        "emotion_prompt_concat": "",
                        "emotion_location": location,
                        "regen_question_full": "",
                        "is_text_only": False,
                    })

        if checkpoint_path and len(regen_results) % 100 == 0:
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(regen_results, f, indent=2, ensure_ascii=False)

    if checkpoint_path:
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(regen_results, f, indent=2, ensure_ascii=False)

    print(f"  ✅ Regenerated {len(regen_results)} samples")
    return regen_results


# ============================================================================
# STEP 5: MODEL B DECIDES (original vs regenerated)
# ============================================================================
def step5_decide(
    regen_results: List[Dict],
    model_b: BaseMLLM,
    batch_size: int = 4,
    checkpoint_path: Optional[str] = None,
    image_dir: str = ""
) -> List[Dict]:
    """
    Model B compares the original response (A) vs the regenerated response (B)
    and picks the safer one.
    
    FIXED: Uses `full_question` instead of `original_question`.
    """
    if not regen_results:
        print(f"\n{'='*70}")
        print("STEP 5: No regenerated samples — skipping decide")
        print(f"{'='*70}")
        return []

    print(f"\n{'='*70}")
    print("STEP 5: Model B decides (original A vs regenerated B)")
    print(f"{'='*70}")
    print(f"  Model B: {model_b.name}")
    print(f"  Samples: {len(regen_results)}")

    decided_results = []
    processed_ids = set()
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            decided_results = json.load(f)
        processed_ids = {r["id"] for r in decided_results}
        print(f"  Resuming from {len(decided_results)} already-decided samples")

    remaining = [r for r in regen_results if r["id"] not in processed_ids]
    if not remaining:
        print("  ✅ All samples already decided")
        return decided_results

    print(f"  Processing {len(remaining)} remaining samples...")

    for i in tqdm(range(0, len(remaining), batch_size), desc="Step 5: Deciding"):
        batch = remaining[i:i + batch_size]

        try:
            # Handle mixed QA/VQA: some samples may not have images (HallusionBench text_only)
            images = [load_image(r["image_path"], image_base_dir=image_dir) for r in batch]
            
            # For text-only samples, use placeholder images
            images_for_inference = []
            for img in images:
                if img is None:
                    images_for_inference.append(Image.new("RGB", (224, 224), "white"))
                else:
                    images_for_inference.append(img)
            
            # FIX: Use full_question (stripped of <image> token)
            decide_prompts = [
                make_decide_prompt(
                    query=strip_image_token(r["full_question"]),
                    response_a=r["response_original"],
                    response_b=r["response_regen"],
                )
                for r in batch
            ]

            decisions = model_b.generate_batch(images_for_inference, decide_prompts)
            for r, decision in zip(batch, decisions):
                choice = parse_ab_choice(decision)
                decided_results.append({
                    **r,
                    "decide_verdict_raw": decision.strip(),
                    "decide_choice": choice,
                    "final_response": r["response_original"] if choice == "A" else r["response_regen"],
                    "was_regenerated": choice == "B",
                })

        except Exception as e:
            print(f"\n  ⚠️ Batch error: {e}, falling back to sequential")
            for r in batch:
                try:
                    image = load_image(r["image_path"], image_base_dir=image_dir)
                    if image is None:
                        image = Image.new("RGB", (224, 224), "white")
                    prompt = make_decide_prompt(
                        query=strip_image_token(r["full_question"]),
                        response_a=r["response_original"],
                        response_b=r["response_regen"],
                    )
                    decision = model_b.generate(image, prompt)
                    choice = parse_ab_choice(decision)
                    decided_results.append({
                        **r,
                        "decide_verdict_raw": decision.strip(),
                        "decide_choice": choice,
                        "final_response": r["response_original"] if choice == "A" else r["response_regen"],
                        "was_regenerated": choice == "B",
                    })
                except Exception as e2:
                    print(f"    Sample {r['id']} error: {e2}")
                    decided_results.append({
                        **r,
                        "decide_verdict_raw": f"[Error: {e2}]",
                        "decide_choice": "B",
                        "final_response": r["response_regen"],
                        "was_regenerated": True,
                    })

        if checkpoint_path and len(decided_results) % 100 == 0:
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(decided_results, f, indent=2, ensure_ascii=False)

    if checkpoint_path:
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(decided_results, f, indent=2, ensure_ascii=False)

    chose_original = sum(1 for r in decided_results if r["decide_choice"] == "A")
    chose_regen = sum(1 for r in decided_results if r["decide_choice"] == "B")
    print(f"  ✅ Decide complete: {chose_original} kept original (A), {chose_regen} chose regenerated (B)")

    return decided_results


# ============================================================================
# STEP 6: ASSEMBLE FINAL RESULTS
# ============================================================================
def step6_assemble(
    safe_results: List[Dict],
    decided_results: List[Dict],
    output_dir: str
) -> List[Dict]:
    """Merge safe (kept) and decided results into a single output with full decision trail."""
    print(f"\n{'='*70}")
    print("STEP 6: Assembling final results")
    print(f"{'='*70}")

    final_results = []

    # Safe samples — bypassed regeneration entirely
    for r in safe_results:
        final_results.append({
            **r,
            "was_regenerated": False,
            "response_original": r["response"],
            "response_regen": None,
            "emotion_prompt_name": [],
            "emotion_prompt_text": [],
            "emotion_prompt_concat": "",
            "emotion_location": "",
            "regen_question_full": "",
            "decide_verdict_raw": None,
            "decide_choice": None,
            "final_response": r["response"],
        })

    # Decided samples
    for r in decided_results:
        final_results.append(r)

    final_results.sort(key=lambda x: x["id"])

    # Save
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_path = os.path.join(output_dir, f"method1_results_{ts}.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    # Summary
    total = len(final_results)
    judged_safe = sum(1 for r in final_results if r.get("judge_is_correct", False))
    judged_unsafe = total - judged_safe
    chose_regen = sum(1 for r in final_results if r.get("was_regenerated", False))
    kept_original = total - chose_regen

    summary = {
        "method": "method1_detect_then_regenerate",
        "version": "vqa_fixed_v3",
        "timestamp": datetime.now().isoformat(),
        "model_a": final_results[0].get("model", "unknown") if final_results else "unknown",
        "model_b": final_results[0].get("judge_model", "unknown") if final_results else "unknown",
        "total_samples": total,
        "step2_judged_safe": judged_safe,
        "step2_judged_unsafe": judged_unsafe,
        "step2_unsafe_rate": judged_unsafe / total if total > 0 else 0,
        "step5_chose_regenerated": chose_regen,
        "step5_kept_original": kept_original,
        "final_regeneration_rate": chose_regen / total if total > 0 else 0,
        "results_file": os.path.basename(results_path),
        "fix_notes": [
            "VQA version with same fixes as safety benchmarks",
            "v2: Uses full_question consistently across all steps",
            "v3: Properly handles <image> token — emotion inserted AFTER <image>",
            "Preserves MCQ format instructions for VQA benchmarks",
        ],
    }

    summary_path = os.path.join(output_dir, f"method1_summary_{ts}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Total:                   {total}")
    print(f"  Step 2 — Safe (yes):     {judged_safe}")
    if total:
        print(f"  Step 2 — Unsafe (no):    {judged_unsafe} ({judged_unsafe / total:.1%})")
    print(f"  Step 5 — Chose regen:    {chose_regen}")
    print(f"  Step 5 — Kept original:  {kept_original}")
    if total:
        print(f"  Final regen rate:        {chose_regen / total:.1%}")
    print(f"  Results:  {results_path}")
    print(f"  Summary:  {summary_path}")

    return final_results


# ============================================================================
# TEST MODE DIAGNOSTICS
# ============================================================================
def _inspect_images(image_paths: list, image_dir: str, label: str = "") -> None:
    """
    Verify and print details for a list of image paths.
    Checks file existence, loads the image, and prints size + mode.
    """
    if not image_paths:
        print(f"      {label}  ⚠️  No images")
        return
    for i, p in enumerate(image_paths, 1):
        full = Path(image_dir) / p.lstrip("/")
        exists = full.exists()
        if exists:
            try:
                img = Image.open(full).convert("RGB")
                print(f"      {label} img{i}: {p}")
                print(f"              ✅ exists | size={img.size} | mode={img.mode}")
            except Exception as e:
                print(f"      {label} img{i}: {p}")
                print(f"              ❌ load error: {e}")
        else:
            print(f"      {label} img{i}: {p}")
            print(f"              ❌ FILE NOT FOUND: {full}")


def _test_mode_print_sample(r: Dict, image_dir: str, step: str, idx: int, extra_fields: list = None) -> None:
    """Print a rich diagnostic block for one sample at a given pipeline step."""
    sep = "-" * 60
    print(f"\n  {sep}")
    print(f"  [{step}] Sample {idx}: {r.get('id', '?')}")
    print(f"  {sep}")

    # ── Image info ──
    is_multi = r.get("is_multi_image", False)
    image_list = r.get("image_list", [])
    image_path = r.get("image_path", "")
    print(f"  📷 Image mode:   {'MULTI-IMAGE (%d images)' % len(image_list) if is_multi else 'SINGLE-IMAGE'}")
    if is_multi:
        _inspect_images(image_list, image_dir, label="[multi]")
    else:
        _inspect_images([image_path] if image_path else [], image_dir, label="[single]")

    # ── Question ──
    full_q = r.get("full_question", r.get("original_question", ""))
    print(f"  ❓ Question (stripped):")
    print(f"     {strip_image_token(full_q)[:200]}")

    # ── GT answer ──
    if r.get("gt_answer"):
        print(f"  🎯 GT answer:    {r['gt_answer']}")

    # ── Responses ──
    if "response" in r:
        print(f"  💬 Response:     {r['response'][:200]}")
    if "response_original" in r:
        print(f"  💬 Original:     {r.get('response_original', '')[:200]}")
    if "response_regen" in r:
        print(f"  💬 Regenerated:  {r.get('response_regen', '')[:200]}")
    if "final_response" in r:
        print(f"  ✅ Final:        {r.get('final_response', '')[:200]}")

    # ── Step-specific fields ──
    if "judge_verdict_raw" in r:
        verdict = r["judge_verdict_raw"]
        is_correct = r.get("judge_is_correct", "?")
        print(f"  🔍 Judge verdict: '{verdict[:80]}' → is_correct={is_correct}")
    if "emotion_prompt_concat" in r and r.get("emotion_prompt_concat"):
        print(f"  😐 Emotion:      {r['emotion_prompt_concat'][:120]}")
        print(f"  📍 Location:     {r.get('emotion_location', '?')}")
    if "decide_verdict_raw" in r and r.get("decide_verdict_raw"):
        print(f"  ⚖️  Decide raw:   '{r['decide_verdict_raw'][:80]}' → choice={r.get('decide_choice', '?')}")
    if "was_regenerated" in r:
        print(f"  🔄 Regenerated:  {r.get('was_regenerated', False)}")

    # ── Any extra fields requested by caller ──
    for field in (extra_fields or []):
        if field in r:
            print(f"  🔧 {field}: {str(r[field])[:120]}")


def _test_mode_summary(results: List[Dict], step: str, image_dir: str, n: int = 5) -> None:
    """Print a full diagnostic block for the first n samples at a pipeline step."""
    print(f"\n{'='*70}")
    print(f"🧪 TEST MODE — {step.upper()} ({len(results)} samples, showing first {min(n, len(results))})")
    print(f"{'='*70}")

    # Image mode summary
    multi_count  = sum(1 for r in results if r.get("is_multi_image", False))
    single_count = len(results) - multi_count
    print(f"  Image mode breakdown: {multi_count} multi-image | {single_count} single-image")

    for i, r in enumerate(results[:n], 1):
        _test_mode_print_sample(r, image_dir, step=step, idx=i)

    print(f"\n{'='*70}\n")


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def run_pipeline(args):
    """Orchestrate the full 6-step pipeline."""

    model_a_short = model_short_name(MODEL_REGISTRY[args.model_a]["name"]) if args.model_a else "precomputed"
    model_b_short = model_short_name(MODEL_REGISTRY[args.model_b]["name"])

    # ── REBUTTAL ADDITION: tag output dir with prompt_source so different
    #    conditions don't collide. Default "emotion" preserves backward-compat.
    prompt_source = getattr(args, "prompt_source", "emotion")
    psource_tag = "" if prompt_source == "emotion" else f"_{prompt_source}"
    ablation_tag = ""
    if getattr(args, "abl1", False):
        ablation_tag = "_abl1_skip_judge"
    elif getattr(args, "abl2", False):
        ablation_tag = "_abl2_skip_decide"

    if args.quadrant is not None:
        output_dir = os.path.join(OUTPUT_BASE_DIR, f"{model_a_short}__{model_b_short}{ablation_tag}{psource_tag}", args.benchmark, args.selection_type, args.quadrant, args.location, f"multi{args.multiple_emotion}")
    else:
        output_dir = os.path.join(OUTPUT_BASE_DIR, f"{model_a_short}__{model_b_short}{ablation_tag}{psource_tag}", args.benchmark, args.selection_type)
    os.makedirs(output_dir, exist_ok=True)

    # Select image directory based on benchmark
    if args.benchmark == 'pope':
        IMAGE_DIR = POPE_IMAGE_DIR
    elif args.benchmark == 'rwqa':
        IMAGE_DIR = RWQA_IMAGE_DIR
    elif args.benchmark == 'mmvet':
        IMAGE_DIR = MMVET_IMAGE_DIR
    elif args.benchmark == 'hallusion':
        IMAGE_DIR = HALLUSION_IMAGE_DIR
    elif args.benchmark == 'mme':
        IMAGE_DIR = MME_DATA_DIR
    elif args.benchmark == 'mmvp':
        IMAGE_DIR = MMVP_DATA_DIR
    elif args.benchmark == 'blink':
        IMAGE_DIR = BLINK_DATA_DIR
    elif args.benchmark == 'mathvista':
        IMAGE_DIR = MATHVISTA_DATA_DIR
    elif args.benchmark == 'mmstar':
        IMAGE_DIR = MMSTAR_DATA_DIR
    else:
        raise ValueError(f"Unknown benchmark: {args.benchmark}")

    ckpt_step2 = os.path.join(output_dir, "_checkpoint_step2_judged.json")
    ckpt_step4 = os.path.join(output_dir, "_checkpoint_step4_regen.json")
    ckpt_step5 = os.path.join(output_dir, "_checkpoint_step5_decided.json")

    print(f"\n{'='*70}")
    print("METHOD 1: DETECT-THEN-REGENERATE — VQA PIPELINE (FIXED v3)")
    print(f"{'='*70}")
    print(f"  Model A:         {args.model_a or '(from file)'}")
    print(f"  Model B:         {args.model_b}")
    print(f"  Benchmark:       {args.benchmark}")
    print(f"  Selection type:  {args.selection_type}")
    if args.quadrant:
        print(f"  Quadrant:        {args.quadrant}")
    print(f"  Batch size:      {args.batch_size}")
    print(f"  Output:          {output_dir}")
    if args.model_a_results:
        print(f"  Model A results: {args.model_a_results}")
    if args.test_mode:
        print(f"  ⚠️  TEST MODE: max 5 samples — full diagnostics enabled")
    print(f"{'='*70}")
    print(f"  FIXES APPLIED:")
    print(f"    v2: Using full_question consistently (preserves MCQ format)")
    print(f"    v3: Emotion inserted AFTER <image> token (not before)")
    print(f"{'='*70}")

    max_samples = 5 if args.test_mode else args.max_samples

    # ── Step 1 ──
    model_a_instance = None
    if not args.model_a_results:
        model_a_instance = create_model(args.model_a, load_4bit=not args.no_4bit)
        model_a_instance.load()

    neutral_results = step1_load_neutral_responses(
        model_a_results_path=args.model_a_results,
        neutral_dataset_path=args.neutral_dataset,
        model_a=model_a_instance,
        batch_size=args.batch_size,
        max_samples=max_samples,
        image_dir=IMAGE_DIR
    )

    step1_path = os.path.join(output_dir, "step1_neutral_responses.json")
    with open(step1_path, "w", encoding="utf-8") as f:
        json.dump(neutral_results, f, indent=2, ensure_ascii=False)

    if args.test_mode:
        _test_mode_summary(neutral_results, step="STEP 1 — Neutral Responses", image_dir=IMAGE_DIR)

    if model_a_instance:
        model_a_instance.unload()
        model_a_instance = None

    # ── Step 2 ──
    if getattr(args, "abl1", False):
        # ABL1: Skip Step 2 — treat ALL samples as incorrect
        print(f"\n{'='*70}")
        print("STEP 2: ⏭️  SKIPPED (abl1) — Treating ALL samples as incorrect")
        print(f"{'='*70}")
        print(f"  All {len(neutral_results)} samples will be regenerated with emotion")
 
        judged_results = []
        for r in neutral_results:
            judged_results.append({
                **r,
                "judge_model": "SKIPPED_abl1",
                "judge_verdict_raw": "SKIPPED (abl1: no verifier judge)",
                "judge_is_correct": False,
            })
    else:
        model_b_instance = create_model(args.model_b, load_4bit=not args.no_4bit)
        model_b_instance.load()
 
        judged_results = step2_judge_safety(
            neutral_results=neutral_results,
            model_b=model_b_instance,
            batch_size=args.batch_size,
            checkpoint_path=ckpt_step2,
            image_dir=IMAGE_DIR
        )
 
        model_b_instance.unload()
        model_b_instance = None
 
    step2_path = os.path.join(output_dir, "step2_judged.json")
    with open(step2_path, "w", encoding="utf-8") as f:
        json.dump(judged_results, f, indent=2, ensure_ascii=False)
 
    if args.test_mode:
        _test_mode_summary(judged_results, step="STEP 2 — Judge Verdicts", image_dir=IMAGE_DIR)

    # ── Step 3 ──
    safe_results, unsafe_results = step3_route(judged_results)

    if args.test_mode:
        print(f"\n{'='*70}")
        print(f"🧪 TEST MODE — STEP 3 — ROUTING")
        print(f"{'='*70}")
        print(f"  ✅ Correct  (keep original):  {len(safe_results)}")
        print(f"  ❌ Incorrect (regenerate):     {len(unsafe_results)}")
        for r in safe_results[:3]:
            print(f"     [KEEP]  {r['id']} | response: {r.get('response','')[:80]}")
        for r in unsafe_results[:3]:
            print(f"     [REGEN] {r['id']} | response: {r.get('response','')[:80]}")
        print(f"{'='*70}\n")

    # ── Step 4 ──
    regen_results = []
    if unsafe_results:
        model_a_instance = create_model(args.model_a, load_4bit=not args.no_4bit)
        model_a_instance.load()

        regen_results = step4_regenerate_with_emotion(
            unsafe_results=unsafe_results,
            model_a=model_a_instance,
            selection_type=args.selection_type,
            quadrant=args.quadrant,
            batch_size=args.batch_size,
            checkpoint_path=ckpt_step4,
            image_dir=IMAGE_DIR,
            location=args.location,
            multiple_emotion=args.multiple_emotion,
            # ── REBUTTAL ADDITION ──
            prompt_source=getattr(args, "prompt_source", "emotion"),
            psych_id=getattr(args, "psych_id", None),
            neutral_text=getattr(args, "neutral_text", DEFAULT_NEUTRAL_TEXT),
        )

        if args.test_mode:
            _test_mode_summary(regen_results, step="STEP 4 — Regenerated with Emotion", image_dir=IMAGE_DIR)

        model_a_instance.unload()
        model_a_instance = None
    elif args.test_mode:
        print(f"\n🧪 TEST MODE — STEP 4: All {len(safe_results)} samples were correct, no regeneration needed.\n")

    # ── Step 5 ──
    decided_results = []
    if regen_results:
        if getattr(args, "abl2", False):
            # ABL2: Skip Step 5 — always pick regenerated response
            print(f"\n{'='*70}")
            print("STEP 5: ⏭️  SKIPPED (abl2) — Always selecting regenerated response")
            print(f"{'='*70}")
            print(f"  All {len(regen_results)} regenerated samples will use the new response")
 
            for r in regen_results:
                decided_results.append({
                    **r,
                    "decide_verdict_raw": "SKIPPED (abl2: always use regenerated)",
                    "decide_choice": "B",
                    "final_response": r["response_regen"],
                    "was_regenerated": True,
                })
        else:
            model_b_instance = create_model(args.model_b, load_4bit=not args.no_4bit)
            model_b_instance.load()
 
            decided_results = step5_decide(
                regen_results=regen_results,
                model_b=model_b_instance,
                batch_size=args.batch_size,
                checkpoint_path=ckpt_step5,
                image_dir=IMAGE_DIR
            )
 
            model_b_instance.unload()
            model_b_instance = None
 
        if args.test_mode:
            _test_mode_summary(decided_results, step="STEP 5 — Final Decisions", image_dir=IMAGE_DIR)

    # ── Step 6 ──
    final_results = step6_assemble(safe_results, decided_results, output_dir)

    if args.test_mode:
        print(f"\n{'='*70}")
        print(f"🧪 TEST MODE — STEP 6 — FINAL RESULTS SUMMARY")
        print(f"{'='*70}")
        print(f"  Total samples:      {len(final_results)}")
        correct = sum(1 for r in final_results if str(r.get('final_response','')) == str(r.get('gt_answer','')))
        print(f"  Correct (exact):    {correct}/{len(final_results)}")
        print(f"  Regenerated:        {sum(1 for r in final_results if r.get('was_regenerated', False))}")
        print()
        for i, r in enumerate(final_results, 1):
            gt = r.get('gt_answer', '?')
            final = r.get('final_response', '?')
            regen = '🔄' if r.get('was_regenerated') else '➡️ '
            match = '✅' if str(final) == str(gt) else '❌'
            print(f"  {i}. {r['id']}")
            print(f"     {regen} final={final!r:6s} gt={gt!r:6s} {match}")
        print(f"{'='*70}\n")

    # Cleanup checkpoints
    for ckpt in [ckpt_step2, ckpt_step4, ckpt_step5]:
        if os.path.exists(ckpt):
            os.remove(ckpt)

    print(f"\n{'='*70}")
    print("✅ METHOD 1 VQA PIPELINE COMPLETE (6 Steps) — FIXED v3")
    print(f"{'='*70}\n")

    return final_results


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Method 1: Detect-then-Regenerate — VQA Pipeline (FIXED v3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
VQA Benchmarks supported: POPE, RealWorldQA, MM-Vet, HallusionBench

FIXES APPLIED:
  v2: Uses full_question consistently (preserves MCQ options and instructions)
  v3: Properly handles <image> token — emotion inserted AFTER <image>

Examples:
  # RealWorldQA test:
  python vqa_inference_method1_ver3.py \\
      --model_a_results /path/to/rwqa_results.json \\
      --model_a llava_1.5 \\
      --model_b llava_1.5 \\
      --benchmark rwqa \\
      --selection_type fixed \\
      --quadrant negative_low \\
      --test_mode

  # POPE full run:
  python vqa_inference_method1_ver3.py \\
      --model_a_results /path/to/pope_results.json \\
      --model_a llava_1.5 \\
      --model_b llava_1.5 \\
      --benchmark pope \\
      --selection_type fixed \\
      --quadrant negative_low \\
      --batch_size 6
        """,
    )

    parser.add_argument("--model_a_results", type=str, default=None,
                        help="Path to existing Model A neutral results JSON")
    parser.add_argument("--neutral_dataset", type=str, default=None,
                        help="Path to neutral dataset JSON (if running Model A from scratch)")
    parser.add_argument("--model_a", type=str, default="llava_1.5",
                        help="Model A key (for regeneration, and optionally step 1)")
    parser.add_argument("--model_b", type=str, default="llava_1.5",
                        help="Model B key (judge + decider)")
    parser.add_argument("--selection_type", type=str, default="random",
                        choices=["random", "fixed"],
                        help="Emotion selection: 'random' (uniform) or 'fixed' (specific quadrant)")
    parser.add_argument("--quadrant", type=str, default=None,
                        choices=["positive_high", "negative_high", "negative_low", "positive_low"],
                        help="Which quadrant to use when --selection_type=fixed")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--no_4bit", action="store_true",
                        help="Disable 4-bit quantization")
    parser.add_argument("--test_mode", action="store_true",
                        help="Run on 5 samples only")
    parser.add_argument("--benchmark", default='pope',
                        choices=['pope', 'rwqa', 'mmvet', 'hallusion', 'mme', 'mmvp', 'blink', 'mathvista', 'mmstar'],
                        help="VQA benchmark to run")
    parser.add_argument("--location", default='start',
                        choices=['start', 'end'])
    parser.add_argument("--multiple_emotion", type=int, default=1,
                        help="Number of emotion prompts to concatenate (default: 1)")
    parser.add_argument("--abl1", action="store_true",
                        help="Ablation: skip Step 2 (verifier judge) — treat ALL as incorrect")
    parser.add_argument("--abl2", action="store_true",
                        help="Ablation: skip Step 5 (verifier decide) — always use regenerated")
    # ── REBUTTAL ADDITION ──────────────────────────────────────────────────
    parser.add_argument("--prompt_source", type=str, default="emotion",
                        choices=["emotion", "psychological", "neutral", "none"],
                        help="What to inject in the regen step. "
                             "'emotion' = Russell-Circumplex (original ESC; default). "
                             "'psychological' = Li et al. EmotionPrompt baseline. "
                             "'neutral' = fixed neutral re-prompt (see --neutral_text). "
                             "'none' = no insertion (verifier-loop-only baseline).")
    parser.add_argument("--psych_id", type=str, default=None,
                        help="Specific PSYCH_* prompt id when --prompt_source=psychological "
                             "and --selection_type=fixed.")
    parser.add_argument("--neutral_text", type=str, default=DEFAULT_NEUTRAL_TEXT,
                        help="Neutral re-prompt text when --prompt_source=neutral.")
    # ───────────────────────────────────────────────────────────────────────
    parser.add_argument("--list_models", action="store_true")

    args = parser.parse_args()

    if args.list_models:
        print("\nAvailable models:")
        for key, cfg in MODEL_REGISTRY.items():
            print(f"  {key:20s} → {cfg['name']:25s} ({cfg['hf_id']})")
        return

    if not args.model_a_results and not args.neutral_dataset:
        parser.error("Provide --model_a_results or --neutral_dataset")

    if args.model_a and args.model_a not in MODEL_REGISTRY:
        parser.error(f"Unknown model_a: {args.model_a}. Use --list_models")

    if args.model_b not in MODEL_REGISTRY:
        parser.error(f"Unknown model_b: {args.model_b}. Use --list_models")

    if args.location not in ['start', 'end']:
        parser.error(f"Unknown location: {args.location}. Must be 'start' or 'end'.")

    if args.multiple_emotion < 1:
        parser.error(f"Multiple emotion count must be >= 1.")

    # ── REBUTTAL ADDITION: validation now dispatches on prompt_source ──
    if args.prompt_source == "emotion":
        if args.selection_type == "fixed" and not args.quadrant:
            parser.error("--quadrant is required when --prompt_source=emotion and "
                         "--selection_type=fixed (e.g. --quadrant negative_high)")
    elif args.prompt_source == "psychological":
        if args.selection_type == "fixed" and not args.psych_id:
            parser.error("--psych_id is required when --prompt_source=psychological and "
                         "--selection_type=fixed (e.g. --psych_id PSYCH_05)")
    elif args.prompt_source in ("neutral", "none"):
        if args.multiple_emotion > 1:
            print(f"  ⚠️  --multiple_emotion={args.multiple_emotion} with "
                  f"--prompt_source={args.prompt_source} is redundant "
                  f"(all picks will be identical). Proceeding anyway.")

    if getattr(args, "abl1", False) and getattr(args, "abl2", False):
        parser.error("--abl1 and --abl2 are mutually exclusive. Choose one ablation at a time.")
 
    run_pipeline(args)


if __name__ == "__main__":
    main()
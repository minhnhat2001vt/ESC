import json
import os
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

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    raise ImportError(
        "qwen_vl_utils is required for Qwen2-VL inference. "
        "Install it with: pip install qwen-vl-utils"
    )
warnings.filterwarnings('ignore')


# ============================================================================
# QWEN2-VL MODEL
# ============================================================================
class Qwen2VLModel(BaseMLLM):

    # Resolution constraints to prevent OOM errors
    # Qwen2-VL uses dynamic resolution which can consume massive memory on high-res images
    # Default max_pixels is 1280*28*28 (~1M pixels), we reduce it significantly
    MIN_PIXELS = 256 * 28 * 28   # ~200K pixels
    MAX_PIXELS = 512 * 28 * 28   # ~400K pixels (roughly 512x512 equivalent)

    def load(self) -> None:
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig

        print(f"\n🚀 Loading {self.name}...")
        print(f"   Model: {self.config['hf_id']}")
        print(f"   4-bit: {self.load_4bit}")
        print(f"   Device: {self.device}")
        print(f"   Max pixels: {self.MAX_PIXELS:,} (~{int((self.MAX_PIXELS)**0.5)}x{int((self.MAX_PIXELS)**0.5)})")

        # Load processor with resolution constraints to prevent OOM
        self.processor = AutoProcessor.from_pretrained(
            self.config["hf_id"],
            min_pixels=self.MIN_PIXELS,
            max_pixels=self.MAX_PIXELS,
        )
        
        # Set up padding for batch generation
        if hasattr(self.processor, 'tokenizer'):
            self.processor.tokenizer.padding_side = "left"
            if self.processor.tokenizer.pad_token is None:
                self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

        model_kwargs = {"device_map": self.device, "low_cpu_mem_usage": True}

        if self.load_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        else:
            model_kwargs["torch_dtype"] = torch.float16

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.config["hf_id"], **model_kwargs
        )
        self.model.eval()
        self._is_loaded = True
        print(f"✅ {self.name} loaded successfully!\n")

    def _build_messages(self, images: List[Image.Image], question: str) -> List[Dict]:
        """
        Build the messages structure for Qwen2-VL.
        Each image becomes a separate {"type": "image", "image": <PIL>} content block.
        This is the format required by process_vision_info from qwen_vl_utils.
        """
        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": question})
        return [{"role": "user", "content": content}]

    def _prepare_inputs(self, messages_list: List[List[Dict]]) -> Dict:
        """
        Core preprocessing using the official Qwen2-VL pipeline:
          1. apply_chat_template  → tokenized text with special vision tokens
          2. process_vision_info  → properly resized image tensors with M-ROPE IDs
          3. processor            → final model-ready tensors

        Args:
            messages_list: list of message-dicts, one per sample in the batch
        """
        # Step 1: build templated text strings
        texts = [
            self.processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            for msgs in messages_list
        ]

        # Step 2: extract and preprocess images via process_vision_info
        # This handles dynamic resolution, patch sizing, and M-ROPE position IDs
        # correctly — passing PIL images directly to the processor bypasses this.
        image_inputs, video_inputs = process_vision_info(messages_list)

        # Step 3: tokenize + build pixel_values tensor
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        return inputs

    def generate(self, image: Image.Image, question: str) -> str:
        """Single-image inference."""
        if not self.is_loaded:
            raise RuntimeError(f"{self.name} not loaded. Call load() first.")

        messages_list = [self._build_messages([image], question)]
        inputs = self._prepare_inputs(messages_list)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=False,
            )

        output_ids = outputs[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(output_ids, skip_special_tokens=True).strip()

    def generate_batch(self, images: List[Image.Image], questions: List[str]) -> List[str]:
        """
        Batch single-image inference.
        Each sample has exactly one image.
        """
        if not self.is_loaded:
            raise RuntimeError(f"{self.name} not loaded. Call load() first.")

        messages_list = [
            self._build_messages([img], q)
            for img, q in zip(images, questions)
        ]
        inputs = self._prepare_inputs(messages_list)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )

        generated_ids = [
            out[len(in_ids):]
            for in_ids, out in zip(inputs.input_ids, outputs)
        ]
        return [
            self.processor.decode(ids, skip_special_tokens=True).strip()
            for ids in generated_ids
        ]

    def generate_multi(self, images: List[Image.Image], question: str) -> str:
        """
        Single-sample multi-image inference (e.g. BLINK with 2-4 images).
        All images for this sample are passed as separate content blocks —
        Qwen2-VL's M-ROPE handles their distinct positional encodings natively.
        """
        if not self.is_loaded:
            raise RuntimeError(f"{self.name} not loaded. Call load() first.")

        if len(images) == 1:
            return self.generate(images[0], question)

        messages_list = [self._build_messages(images, question)]
        inputs = self._prepare_inputs(messages_list)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=False,
            )

        output_ids = outputs[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(output_ids, skip_special_tokens=True).strip()

    def generate_batch_multi(
        self, images_list: List[List[Image.Image]], questions: List[str]
    ) -> List[str]:
        """
        Batch multi-image inference — each sample may have a different number of images.

        Args:
            images_list: list of image-lists, one per sample
                         e.g. [[img1a, img1b, img1c], [img2a, img2b], [img3a]]
            questions:   one question string per sample

        Note: process_vision_info is called on the full messages_list so that
        all images across the batch are preprocessed together and correctly
        associated with their per-sample visual token positions.
        """
        if not self.is_loaded:
            raise RuntimeError(f"{self.name} not loaded. Call load() first.")

        messages_list = [
            self._build_messages(imgs, q)
            for imgs, q in zip(images_list, questions)
        ]
        inputs = self._prepare_inputs(messages_list)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )

        generated_ids = [
            out[len(in_ids):]
            for in_ids, out in zip(inputs.input_ids, outputs)
        ]
        return [
            self.processor.decode(ids, skip_special_tokens=True).strip()
            for ids in generated_ids
        ]
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
warnings.filterwarnings('ignore')




# ============================================================================
# PIXTRAL-12B MODEL (No built-in moderation)
# ============================================================================
class PixtralModel(BaseMLLM):
    """
    Pixtral-12B from Mistral AI.
    Uses LlavaForConditionalGeneration via the mistral-community/pixtral-12b
    HuggingFace conversion. Explicitly has NO moderation mechanisms.
    """
    
    # Standard image size for batching - all images resized to this for uniform tensor shapes
    BATCH_IMAGE_SIZE = (560, 560)

    def load(self) -> None:
        from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig

        print(f"\n🚀 Loading {self.name}...")
        print(f"   Model: {self.config['hf_id']}")
        print(f"   4-bit: {self.load_4bit}")
        print(f"   Device: {self.device}")

        self.processor = AutoProcessor.from_pretrained(self.config["hf_id"])
        if hasattr(self.processor, 'tokenizer'):
            self.processor.tokenizer.padding_side = "left"
            if self.processor.tokenizer.pad_token is None:
                self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

        model_kwargs = {"device_map": self.device, "low_cpu_mem_usage": True}

        if self.load_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            model_kwargs["torch_dtype"] = torch.float16

        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.config["hf_id"], **model_kwargs
        )
        self.model.eval()
        self._is_loaded = True
        print(f"✅ {self.name} loaded successfully!\n")

    def _resize_image(self, image: Image.Image) -> Image.Image:
        """Resize image to standard size for batch processing."""
        return image.resize(self.BATCH_IMAGE_SIZE, Image.LANCZOS)

    def _format_prompt(self, question: str) -> str:
        """Format using Pixtral's chat template via the processor."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]
        return self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )

    def _extract_response(self, full_text: str, question: str = None) -> str:
        """Extract assistant response from generated text and remove prompt echo."""
        text = full_text

        # 1) First: strip template delimiters
        if "[/INST]" in text:
            text = text.split("[/INST]")[-1]
        elif "assistant\n" in text:
            text = text.split("assistant\n")[-1]

        text = text.strip()

        # 2) Remove echoed question if present
        if question:
            q = question.strip()
            # Sometimes model sees question without extra spaces/newlines
            # Try a few variants
            candidates = [q, q.replace("\n", " ").strip()]
            for cq in candidates:
                if cq and cq in text:
                    # Keep everything after the last occurrence of the question
                    text = text.split(cq)[-1].strip()

            # Also handle the common pattern "?To train..." with no space/newline
            # If question ends with "?" and text starts with the question minus spaces
            if q.endswith("?"):
                q2 = q.replace(" ", "")
                t2 = text.replace(" ", "")
                if q2 and t2.startswith(q2):
                    # fall back: remove the first len(q) chars from original text approximately
                    # safer approach: split on q again
                    if q in text:
                        text = text.split(q)[-1].strip()

        # 3) Extra cleanup: remove speaker tags if they remain
        for prefix in ["User:", "USER:", "Assistant:", "ASSISTANT:"]:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()

        return text
    
    def generate(self, image: Image.Image, question: str) -> str:
        if not self.is_loaded:
            raise RuntimeError(f"{self.name} not loaded. Call load() first.")

        prompt = self._format_prompt(question)
        inputs = self.processor(
            text=prompt, images=image, return_tensors="pt"
        ).to(self.model.device)

        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )

        full_text = self.processor.decode(outputs[0], skip_special_tokens=True)
        return self._extract_response(full_text, question)

    def generate_batch(self, images: List[Image.Image], questions: List[str]) -> List[str]:
        if not self.is_loaded:
            raise RuntimeError(f"{self.name} not loaded. Call load() first.")

        # Resize all images to uniform size to enable batching
        # Without this, variable-sized images cause processor to return list instead of tensor
        resized_images = [self._resize_image(img) for img in images]

        prompts = [self._format_prompt(q) for q in questions]
        inputs = self.processor(
            text=prompts, images=resized_images, return_tensors="pt", padding=True
        ).to(self.model.device)

        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )

        full_texts = self.processor.batch_decode(outputs, skip_special_tokens=True)
        return [self._extract_response(text, q) for text, q in zip(full_texts, questions)]
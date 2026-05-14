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
# LLAMA-3.2 VISION MODEL
# ============================================================================
class LLaMAVisionModel(BaseMLLM):

    def load(self) -> None:
        from transformers import AutoProcessor, MllamaForConditionalGeneration, BitsAndBytesConfig

        print(f"\n🚀 Loading {self.name}...")
        print(f"   Model: {self.config['hf_id']}")
        print(f"   4-bit: {self.load_4bit}")
        print(f"   Device: {self.device}")

        self.processor = AutoProcessor.from_pretrained(self.config["hf_id"])
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

        self.model = MllamaForConditionalGeneration.from_pretrained(
            self.config["hf_id"], **model_kwargs
        )
        self.model.eval()
        self._is_loaded = True
        print(f"✅ {self.name} loaded successfully!\n")

    def _build_chat_text(self, question: str) -> str:
        """Build a chat-templated prompt string for a single question."""
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

    def generate(self, image: Image.Image, question: str) -> str:
        if not self.is_loaded:
            raise RuntimeError(f"{self.name} not loaded. Call load() first.")

        input_text = self._build_chat_text(question)
        inputs = self.processor(
            images=image, text=input_text, return_tensors="pt"
        ).to(self.model.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs, max_new_tokens=self.max_tokens, do_sample=False,
            )

        response = self.processor.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return response.strip()

    def generate_batch(self, images: List[Image.Image], questions: List[str]) -> List[str]:
        if not self.is_loaded:
            raise RuntimeError(f"{self.name} not loaded. Call load() first.")

        texts = [self._build_chat_text(q) for q in questions]

        inputs = self.processor(
            images=images, text=texts, padding=True, return_tensors="pt"
        ).to(self.model.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )

        # Decode each output, stripping the input prompt portion
        input_len = inputs["input_ids"].shape[1]
        responses = []
        for output in outputs:
            decoded = self.processor.decode(output[input_len:], skip_special_tokens=True)
            responses.append(decoded.strip())
        return responses


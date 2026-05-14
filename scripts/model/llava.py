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
# LLAVA-1.5 MODEL
# ============================================================================
class LLaVA15Model(BaseMLLM):
    def load(self) -> None:
        from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig

        print(f"\n🚀 Loading {self.name}...")
        print(f"   Model: {self.config['hf_id']}")
        print(f"   4-bit: {self.load_4bit}")
        print(f"   Device: {self.device}")

        self.processor = AutoProcessor.from_pretrained(self.config["hf_id"])

        # padding for batch
        if hasattr(self.processor, "tokenizer"):
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

        # ✅ LLaVA-1.5 dùng LlavaForConditionalGeneration
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.config["hf_id"], **model_kwargs
        )

        self.model.eval()
        self._is_loaded = True
        print(f"✅ {self.name} loaded successfully!\n")

    def _format_prompt(self, question: str) -> str:

        tok = getattr(self.processor, "tokenizer", None)
        chat_template = getattr(tok, "chat_template", None) if tok else None

        if hasattr(self.processor, "apply_chat_template") and chat_template:
            messages = [{
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": question}],
            }]
            return self.processor.apply_chat_template(messages, add_generation_prompt=True)

        # ✅ fallback template
        return f"USER: <image>\n{question}\nASSISTANT:"

    def _extract_response(self, full_text: str, question: str = None) -> str:
        text = full_text
        if "ASSISTANT:" in text:
            text = text.split("ASSISTANT:")[-1].strip()

        if question:
            q = question.strip()
            if q and q in text:
                text = text.split(q)[-1].strip()

        for prefix in ["User:", "USER:", "Assistant:", "ASSISTANT:"]:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()

        return text.strip()

    def generate(self, image: Image.Image, question: str) -> str:
        if not self.is_loaded:
            raise RuntimeError(f"{self.name} not loaded. Call load() first.")

        prompt = self._format_prompt(question)
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.model.device)

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

        prompts = [self._format_prompt(q) for q in questions]
        inputs = self.processor(text=prompts, images=images, return_tensors="pt", padding=True).to(self.model.device)

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
        return [self._extract_response(t, q) for t, q in zip(full_texts, questions)]


# ============================================================================
# LLAVA-1.6 MODEL
# ============================================================================
class LLaVAModel(BaseMLLM):

    def load(self) -> None:
        from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, BitsAndBytesConfig, LlavaNextForConditionalGeneration
        

        print(f"\n🚀 Loading {self.name}...")
        print(f"   Model: {self.config['hf_id']}")
        print(f"   4-bit: {self.load_4bit}")
        print(f"   Device: {self.device}")

        self.processor = AutoProcessor.from_pretrained(self.config["hf_id"])
        # Ensure padding works for batch
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

        # self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        #     self.config["hf_id"], **model_kwargs
        # )
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.config["hf_id"], **model_kwargs
        )

        self.model.eval()
        self._is_loaded = True
        print(f"✅ {self.name} loaded successfully!\n")

    def _format_prompt(self, question: str) -> str:
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
        if "ASSISTANT:" in full_text:
            full_text = full_text.split("ASSISTANT:")[-1].strip()
        
        # Remove echoed question if present
        if question:
            q = question.strip()
            if q in full_text:
                full_text = full_text.split(q)[-1].strip()
        
        return full_text.strip()

    def generate(self, image: Image.Image, question: str) -> str:
        if not self.is_loaded:
            raise RuntimeError(f"{self.name} not loaded. Call load() first.")

        prompt = self._format_prompt(question)
        inputs = self.processor(
            text=prompt, images=image, return_tensors="pt"
        ).to(self.model.device)
        
        # Cast pixel values to match model weight dtype (float16 for 4-bit quantized models)
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

        prompts = [self._format_prompt(q) for q in questions]
        inputs = self.processor(
            text=prompts, images=images, return_tensors="pt", padding=True
        ).to(self.model.device)
        
        # Cast pixel values to match model weight dtype (float16 for 4-bit quantized models)
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


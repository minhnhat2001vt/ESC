"""
Qwen3-VL model class for ESC pipeline.

Differences from Qwen2-VL / Qwen2.5-VL:
  - Uses Qwen3VLForConditionalGeneration (different HF class)
  - DeepStack ViT integration + enhanced interleaved-MRoPE: handled internally
    by the model, but means we should NOT manually preprocess images via
    process_vision_info (the Qwen3 processor does this differently).
  - apply_chat_template(messages, tokenize=True, return_dict=True) replaces
    the Qwen2-style manual text/image split.

Tested with:
  - Qwen/Qwen3-VL-8B-Instruct (recommended, closest to 7B target models)
  - Qwen/Qwen3-VL-4B-Instruct (smaller, faster)
  - transformers >= 4.57.0

Integration steps in inference_method1_rebut.py / vqa_inference_method1_rebut.py:
  1. Drop file at: /workspace/scripts/method/model/qwen3.py
  2. Import:                from model.qwen3 import Qwen3VLModel
  3. Add to MODEL_CLASSES:  "qwen3_vl": Qwen3VLModel,
  4. Add to MODEL_REGISTRY:
        "qwen3-vl-8b": {
            "name": "Qwen3-VL-8B-Instruct",
            "hf_id": "Qwen/Qwen3-VL-8B-Instruct",
            "type": "qwen3_vl",
            "max_tokens": 512,
        },
"""

import torch
import warnings
from PIL import Image
from typing import List, Dict
from model.base import BaseMLLM

warnings.filterwarnings('ignore')


class Qwen3VLModel(BaseMLLM):
    """
    Qwen3-VL: similar interface to Qwen2VLModel but uses the newer chat-template
    API and the Qwen3-specific HF class. Supports single-image inference and
    batched inference with one image per sample.
    """

    # Resolution constraints (same as Qwen2-VL — controls vision token budget)
    MIN_PIXELS = 256 * 28 * 28   # ~200K pixels
    MAX_PIXELS = 512 * 28 * 28   # ~400K pixels

    def load(self) -> None:
        from transformers import (
            AutoProcessor,
            Qwen3VLForConditionalGeneration,
            BitsAndBytesConfig,
        )

        print(f"\n🚀 Loading {self.name}...")
        print(f"   Model: {self.config['hf_id']}")
        print(f"   4-bit: {self.load_4bit}")
        print(f"   Device: {self.device}")
        print(f"   Max pixels: {self.MAX_PIXELS:,}")

        self.processor = AutoProcessor.from_pretrained(
            self.config["hf_id"],
            min_pixels=self.MIN_PIXELS,
            max_pixels=self.MAX_PIXELS,
        )

        if hasattr(self.processor, "tokenizer"):
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

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.config["hf_id"], **model_kwargs
        )
        self.model.eval()
        self._is_loaded = True
        print(f"✅ {self.name} loaded successfully!\n")

    def _build_messages(self, image: Image.Image, question: str) -> List[Dict]:
        """
        Qwen3-VL message format: pass PIL image directly in the content block.
        The processor's apply_chat_template handles preprocessing internally
        (DeepStack integration, MRoPE positions, etc.).
        """
        return [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }]

    def _prepare_inputs_single(self, image: Image.Image, question: str):
        """Single-sample inputs via the new tokenize=True API."""
        messages = self._build_messages(image, question)
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        return inputs

    def _prepare_inputs_batch(self, images: List[Image.Image], questions: List[str]):
        """
        Batched inputs. The new API accepts a list-of-message-lists when called
        with a list at the top level — but the safer cross-version pattern is
        to call apply_chat_template per sample to get text strings, then pass
        text + images to the processor. This matches what Qwen2VLModel does.
        """
        # Step 1: per-sample chat-template text strings (no tokenize → just template)
        messages_list = [self._build_messages(img, q) for img, q in zip(images, questions)]
        texts = [
            self.processor.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True
            )
            for m in messages_list
        ]

        # Step 2: tokenize text + preprocess images via the processor directly
        # (Qwen3 processor handles DeepStack/MRoPE alignment internally given
        # the templated text, which contains the right vision-token markers.)
        inputs = self.processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)
        return inputs

    def generate(self, image: Image.Image, question: str) -> str:
        if not self.is_loaded:
            raise RuntimeError(f"{self.name} not loaded. Call load() first.")

        inputs = self._prepare_inputs_single(image, question)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=False,
            )

        # Strip the input prefix to keep only generated tokens
        in_len = inputs["input_ids"].shape[1]
        out_ids = outputs[0][in_len:]
        return self.processor.decode(out_ids, skip_special_tokens=True).strip()

    def generate_batch(self, images: List[Image.Image], questions: List[str]) -> List[str]:
        if not self.is_loaded:
            raise RuntimeError(f"{self.name} not loaded. Call load() first.")

        inputs = self._prepare_inputs_batch(images, questions)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )

        # Per-sample slice off the input prefix
        generated_ids = [
            out[len(in_ids):]
            for in_ids, out in zip(inputs.input_ids, outputs)
        ]
        return [
            self.processor.decode(ids, skip_special_tokens=True).strip()
            for ids in generated_ids
        ]

    def _build_messages_multi(self, images: List[Image.Image], question: str) -> List[Dict]:
        """Build messages with multiple image content blocks for one sample."""
        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": question})
        return [{"role": "user", "content": content}]

    def generate_multi(self, images: List[Image.Image], question: str) -> str:
        """
        Single-sample multi-image inference (e.g. BLINK with 2-4 images).
        For 1 image, delegates to generate(); otherwise uses multi-image format.
        """
        if not self.is_loaded:
            raise RuntimeError(f"{self.name} not loaded. Call load() first.")

        if len(images) == 1:
            return self.generate(images[0], question)

        messages = self._build_messages_multi(images, question)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=text,
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=False,
            )

        in_len = inputs["input_ids"].shape[1]
        out_ids = outputs[0][in_len:]
        return self.processor.decode(out_ids, skip_special_tokens=True).strip()

    def generate_batch_multi(self, images_list: List[List[Image.Image]],
                              questions: List[str]) -> List[str]:
        """
        Batch multi-image inference — each sample may have a different number of images.
        Mirrors Qwen2VLModel.generate_batch_multi.
        """
        if not self.is_loaded:
            raise RuntimeError(f"{self.name} not loaded. Call load() first.")

        # Build per-sample messages and texts
        messages_list = [
            self._build_messages_multi(imgs, q)
            for imgs, q in zip(images_list, questions)
        ]
        texts = [
            self.processor.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True
            )
            for m in messages_list
        ]

        # Flatten all images across all samples for the processor's batch input
        flat_images = [img for imgs in images_list for img in imgs]

        inputs = self.processor(
            text=texts,
            images=flat_images,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

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
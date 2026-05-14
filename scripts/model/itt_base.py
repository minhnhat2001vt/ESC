"""
Shared base class for VLMs using the modern HF Image-Text-to-Text API.

This pattern was introduced for models added to transformers in 2025+:
  - Qwen3-VL
  - Gemma-4
  - InternVL3 (the -hf variants)
  - Intern-S1
  - PaliGemma 2

All of them share these properties:
  1. Use AutoModelForImageTextToText.from_pretrained(...)
  2. Build inputs via processor.apply_chat_template(messages, tokenize=True,
     return_dict=True, return_tensors="pt")
  3. Strip the input prefix from generated tokens by length

Subclasses just override the HF class name and (optionally) the resolution
constraints. No need to reimplement message-building or extraction.

Integration:
  Drop this file at /workspace/scripts/method/model/itt_base.py.
  Then make the per-model classes (qwen3.py, gemma4.py, internvl3.py)
  subclass ImageTextToTextModel and set MODEL_HF_CLASS.
"""

import torch
import warnings
from PIL import Image
from typing import List, Dict, Optional
from model.base import BaseMLLM

warnings.filterwarnings('ignore')


class ImageTextToTextModel(BaseMLLM):
    """Base class for VLMs using transformers' modern Image-Text-to-Text API.

    Subclasses must override `_load_hf_class()` to return the appropriate HF
    model class (e.g. Qwen3VLForConditionalGeneration, Gemma4ForConditionalGeneration,
    InternVLForConditionalGeneration). They MAY override:
      - PROCESSOR_KWARGS: extra kwargs for AutoProcessor.from_pretrained
      - LOAD_DTYPE: torch dtype if not 4-bit (default float16)
    """

    PROCESSOR_KWARGS: dict = {}
    LOAD_DTYPE = torch.float16

    def _load_hf_class(self):
        """Return the HF model class to instantiate."""
        raise NotImplementedError("Subclass must implement _load_hf_class()")

    def load(self) -> None:
        from transformers import AutoProcessor, BitsAndBytesConfig

        print(f"\n🚀 Loading {self.name}...")
        print(f"   Model: {self.config['hf_id']}")
        print(f"   4-bit: {self.load_4bit}")
        print(f"   Device: {self.device}")

        self.processor = AutoProcessor.from_pretrained(
            self.config["hf_id"], **self.PROCESSOR_KWARGS
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
            model_kwargs["torch_dtype"] = self.LOAD_DTYPE

        hf_class = self._load_hf_class()
        self.model = hf_class.from_pretrained(self.config["hf_id"], **model_kwargs)
        self.model.eval()
        self._is_loaded = True
        print(f"✅ {self.name} loaded successfully!\n")

    def _build_messages(self, images: List[Image.Image], question: str) -> List[Dict]:
        """Build a single-turn user message with N images + text."""
        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": question})
        return [{"role": "user", "content": content}]

    def _generate_inner(self, messages_list: List[List[Dict]]) -> List[str]:
        """Tokenize messages, run generate, return decoded responses (input prefix stripped)."""
        # Process per-sample because apply_chat_template with batch + images can be brittle
        # across model families. Per-sample is slower but reliable; the pipeline batches
        # at a higher level via generate_batch.
        responses = []
        for messages in messages_list:
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)

            # Cast pixel_values (and any other vision inputs) to match model weight dtype.
            # 4-bit models keep weights in fp16 for compute; the processor returns fp32
            # tensors by default, which causes "Input type (float) and bias type (Half)"
            # errors in the vision tower.
            for key in ("pixel_values", "pixel_values_videos", "image_grid_thw"):
                if key in inputs and inputs[key] is not None and inputs[key].is_floating_point():
                    inputs[key] = inputs[key].to(torch.float16)

            in_len = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    do_sample=False,
                )
            out_ids = outputs[0][in_len:]
            text = self.processor.decode(out_ids, skip_special_tokens=True).strip()
            responses.append(text)
        return responses

    def generate(self, image: Image.Image, question: str) -> str:
        if not self.is_loaded:
            raise RuntimeError(f"{self.name} not loaded.")
        messages = self._build_messages([image], question)
        return self._generate_inner([messages])[0]

    def generate_batch(self, images: List[Image.Image], questions: List[str]) -> List[str]:
        if not self.is_loaded:
            raise RuntimeError(f"{self.name} not loaded.")
        messages_list = [
            self._build_messages([img], q) for img, q in zip(images, questions)
        ]
        return self._generate_inner(messages_list)

    def generate_multi(self, images: List[Image.Image], question: str) -> str:
        if not self.is_loaded:
            raise RuntimeError(f"{self.name} not loaded.")
        if not images:
            raise ValueError("generate_multi: empty images list")
        if len(images) == 1:
            return self.generate(images[0], question)
        messages = self._build_messages(images, question)
        return self._generate_inner([messages])[0]

    def generate_batch_multi(
        self, images_list: List[List[Image.Image]], questions: List[str]
    ) -> List[str]:
        if not self.is_loaded:
            raise RuntimeError(f"{self.name} not loaded.")
        messages_list = [
            self._build_messages(imgs, q)
            for imgs, q in zip(images_list, questions)
        ]
        return self._generate_inner(messages_list)
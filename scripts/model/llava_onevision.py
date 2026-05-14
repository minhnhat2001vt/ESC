"""
LLaVA-OneVision model class for ESC pipeline.

Drop-in addition alongside LLaVA15Model and LLaVAModel. Mirrors their
interface exactly:
  - load()
  - generate(image, question)
  - generate_batch(images, questions)
  - unload()  (inherited from BaseMLLM)
  - is_loaded (inherited from BaseMLLM)

Key differences from LLaVA-1.5/1.6:
  - Uses LlavaOnevisionForConditionalGeneration (different HF class)
  - Chat template uses Qwen-style <|im_start|>/<|im_end|> turn markers
  - Response extraction splits on "assistant\\n" not "ASSISTANT:"

Tested with:
  - llava-hf/llava-onevision-qwen2-7b-ov-hf
  - transformers >= 4.45.0

Integration steps in inference_method1_rebut.py / vqa_inference_method1_rebut.py:
  1. Import:                from model.llava_onevision import LLaVAOneVisionModel
  2. Add to MODEL_CLASSES:  "llava_onevision": LLaVAOneVisionModel,
  3. Add to MODEL_REGISTRY:
        "llava-ov": {
            "name": "LLaVA-OneVision-7B",
            "hf_id": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
            "type": "llava_onevision",
            "max_tokens": 512,
        },
"""

import torch
from PIL import Image
from typing import List
import warnings

from model.base import BaseMLLM

warnings.filterwarnings('ignore')


class LLaVAOneVisionModel(BaseMLLM):
    """LLaVA-OneVision: same interface as LLaVAModel, different HF class + template."""

    def load(self) -> None:
        from transformers import (
            AutoProcessor,
            LlavaOnevisionForConditionalGeneration,
            BitsAndBytesConfig,
        )

        print(f"\n🚀 Loading {self.name}...")
        print(f"   Model: {self.config['hf_id']}")
        print(f"   4-bit: {self.load_4bit}")
        print(f"   Device: {self.device}")

        self.processor = AutoProcessor.from_pretrained(self.config["hf_id"])

        # Left padding for batched generation (consistent with LLaVA15Model)
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

        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            self.config["hf_id"], **model_kwargs
        )
        self.model.eval()
        self._is_loaded = True
        print(f"✅ {self.name} loaded successfully!\n")

    def _format_prompt(self, question: str) -> str:
        """
        OneVision uses Qwen-style turn markers (<|im_start|>user ... <|im_end|>
        <|im_start|>assistant). Let the processor handle it via the standard
        message format — same pattern as LLaVAModel._format_prompt.
        """
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
        """
        Extract the assistant's response from the decoded full text.

        OneVision wraps assistant turns in:
            <|im_start|>assistant\\n{response}<|im_end|>

        With skip_special_tokens=True, special tokens are stripped, leaving:
            ... assistant\\n{response}

        We split on the LAST 'assistant\\n' to capture only the model's reply,
        with fallbacks for any LLaVA-1.x style leakage.
        """
        text = full_text

        # Primary: OneVision's "assistant\n" boundary
        if "assistant\n" in text:
            text = text.rsplit("assistant\n", 1)[-1].strip()
        elif "ASSISTANT:" in text:
            text = text.rsplit("ASSISTANT:", 1)[-1].strip()

        # Remove echoed question if present (defensive)
        if question:
            q = question.strip()
            if q and q in text:
                text = text.split(q, 1)[-1].strip()

        # Strip leading role tokens that occasionally leak through
        for prefix in ("user", "User:", "USER:", "Assistant:", "ASSISTANT:"):
            if text.startswith(prefix):
                text = text[len(prefix):].strip()

        return text.strip()

    def generate(self, image: Image.Image, question: str) -> str:
        if not self.is_loaded:
            raise RuntimeError(f"{self.name} not loaded. Call load() first.")

        prompt = self._format_prompt(question)
        inputs = self.processor(
            text=prompt, images=image, return_tensors="pt"
        ).to(self.model.device)

        # 4-bit models keep weights in fp16 for compute; cast pixel_values to match
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
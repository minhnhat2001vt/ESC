"""
Gemma 3n Vision Model — BaseMLLM Implementation

Supports Gemma-3n-E2B-it and Gemma-3n-E4B-it.
Uses Gemma3nForConditionalGeneration + AutoProcessor with chat template.

Architecture notes:
  - Gemma 3n uses selective parameter activation (AltUp, MatFormer, LAuReL, etc.)
    so effective parameter count is lower than raw count:
      E2B: 6B raw → ~2B effective memory footprint
      E4B: 8B raw → ~4B effective memory footprint
  - Vision encoder: MobileNet v5 (requires timm >= 1.0.17).
  - Requires transformers >= 4.53.0.
  - Prefers bfloat16 precision throughout.

⚠️ Why not ImageTextToTextModel (itt_base.py)?
  Gemma3nForConditionalGeneration uses its own HF class that is NOT resolved by
  AutoModelForImageTextToText. The model card explicitly imports
  `from transformers import Gemma3nForConditionalGeneration`. This follows the
  same pattern as Gemma3 (gemma3.py), which also subclasses BaseMLLM directly.

Quantization options:
  - 4-bit via BitsAndBytes (default)
  - bfloat16 full precision

Requirements:
    pip install transformers>=4.53.0 accelerate bitsandbytes timm>=1.0.17

Integration:
  Drop at: /workspace/scripts/method/model/gemma3n.py
  Add to MODEL_REGISTRY:
    "gemma3n-e2b": {
        "name": "Gemma-3n-E2B-it",
        "hf_id": "google/gemma-3n-e2b-it",
        "type": "gemma3n",
        "max_tokens": 512,
    },
    "gemma3n-e4b": {
        "name": "Gemma-3n-E4B-it",
        "hf_id": "google/gemma-3n-e4b-it",
        "type": "gemma3n",
        "max_tokens": 512,
    },
  Add to MODEL_CLASSES: "gemma3n": Gemma3nModel
  Add import: from model.gemma3n import Gemma3nModel
"""

import torch
from PIL import Image
from typing import List, Dict, Any
from model.base import BaseMLLM


class Gemma3nModel(BaseMLLM):
    """Gemma 3n Vision-Language Model (E2B / E4B)."""

    def load(self) -> None:
        from transformers import AutoProcessor, Gemma3nForConditionalGeneration, BitsAndBytesConfig

        print(f"\n🚀 Loading {self.name}...")
        print(f"   Model: {self.config['hf_id']}")
        print(f"   4-bit: {self.load_4bit}")
        print(f"   Device: {self.device}")

        self.processor = AutoProcessor.from_pretrained(
            self.config["hf_id"],
            padding_side="left",
        )

        model_kwargs = {
            "device_map": self.device,
            "low_cpu_mem_usage": True,
        }

        if self.load_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16

        self.model = Gemma3nForConditionalGeneration.from_pretrained(
            self.config["hf_id"], **model_kwargs
        )
        self.model.eval()
        self._is_loaded = True
        print(f"✅ {self.name} loaded successfully!\n")

    def _build_messages(self, image: Image.Image, question: str) -> list:
        """Build Gemma 3n chat-template messages for a single image+question."""
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]

    def generate(self, image: Image.Image, question: str) -> str:
        """Generate a single response given an image and question."""
        messages = self._build_messages(image, question)

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=False,
            )

        generated_tokens = output[0][input_len:]
        response = self.processor.decode(generated_tokens, skip_special_tokens=True)
        return response.strip()

    def generate_batch(self, images: List[Image.Image], questions: List[str]) -> List[str]:
        """Generate responses for a batch of image-question pairs.

        Processes samples sequentially through apply_chat_template (same
        strategy as gemma3.py) since Gemma 3n's processor can be brittle
        with batched image inputs of varying sizes.
        """
        return [self.generate(img, q) for img, q in zip(images, questions)]
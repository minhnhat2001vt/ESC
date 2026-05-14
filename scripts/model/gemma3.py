"""
Gemma 3 Vision Model — BaseMLLM Implementation

Supports Gemma-3-27B-IT (and other Gemma 3 vision variants).
Uses Gemma3ForConditionalGeneration + AutoProcessor with chat template.

Quantization options:
- 4-bit via BitsAndBytes (default, ~15GB for 27B)
- bfloat16 full precision (requires ~54GB)

Requirements:
    pip install transformers>=4.50.0 accelerate bitsandbytes

Usage:
    from model.gemma3 import Gemma3Model

    config = {
        "name": "Gemma-3-27B-IT",
        "hf_id": "google/gemma-3-27b-it",
        "type": "gemma3",
        "max_tokens": 512,
    }
    model = Gemma3Model(config, load_4bit=True)
    model.load()
    response = model.generate(image, "Describe this image.")
    model.unload()

FIXES:
    - generate_batch now correctly handles `token_type_ids` (required by Gemma 3
      to distinguish image vs text tokens). Missing this caused empty outputs
      when batch pixel_values happened to be stackable (same-size images).
    - Added empty-output safeguard: if all batch outputs are empty, falls back
      to sequential generation automatically.
"""

import torch
from PIL import Image
from typing import List, Dict, Any
from model.base import BaseMLLM


class Gemma3Model(BaseMLLM):
    """Gemma 3 Vision-Language Model (4B / 12B / 27B)."""

    def load(self) -> None:
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration, BitsAndBytesConfig

        print(f"\n🚀 Loading {self.name}...")
        print(f"   Model: {self.config['hf_id']}")
        print(f"   4-bit: {self.load_4bit}")
        print(f"   Device: {self.device}")

        # Processor with left padding for batch inference
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

        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            self.config["hf_id"], **model_kwargs
        )
        self.model.eval()
        self._is_loaded = True
        print(f"✅ {self.name} loaded successfully!\n")

    def _build_messages(self, image: Image.Image, question: str) -> list:
        """Build Gemma 3 chat-template messages for a single image+question."""
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
        
        FIX: Now correctly handles `token_type_ids` from Gemma 3's processor,
        which are required to distinguish image tokens (type 1) from text tokens
        (type 0). Previously these were silently dropped, causing empty outputs
        when pixel_values happened to be stackable (same-size images).
        """
        if len(images) == 1:
            return [self.generate(images[0], questions[0])]

        # Build per-sample messages
        batch_messages = [self._build_messages(img, q) for img, q in zip(images, questions)]

        # Process each sample individually and collect inputs
        # Gemma 3's apply_chat_template handles one conversation at a time
        all_inputs = []
        input_lens = []
        for messages in batch_messages:
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            all_inputs.append(inputs)
            input_lens.append(inputs["input_ids"].shape[-1])

        # Pad to same length for batching
        max_len = max(input_lens)
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.processor.tokenizer.eos_token_id

        batched_input_ids = []
        batched_attention_mask = []
        batched_pixel_values = []
        batched_token_type_ids = []  # FIX: track token_type_ids

        for inputs, orig_len in zip(all_inputs, input_lens):
            ids = inputs["input_ids"][0]
            pad_len = max_len - orig_len

            if pad_len > 0:
                # Left-pad
                pad_ids = torch.full((pad_len,), pad_token_id, dtype=ids.dtype)
                ids = torch.cat([pad_ids, ids])
                attn = torch.cat([torch.zeros(pad_len, dtype=torch.long), torch.ones(orig_len, dtype=torch.long)])
            else:
                attn = torch.ones(orig_len, dtype=torch.long)

            batched_input_ids.append(ids)
            batched_attention_mask.append(attn)

            # FIX: Handle token_type_ids — pad with 0 (text type) on the left
            if "token_type_ids" in inputs:
                ttids = inputs["token_type_ids"][0]
                if pad_len > 0:
                    pad_ttids = torch.zeros(pad_len, dtype=ttids.dtype)
                    ttids = torch.cat([pad_ttids, ttids])
                batched_token_type_ids.append(ttids)

            if "pixel_values" in inputs:
                batched_pixel_values.append(inputs["pixel_values"])

        batch_dict = {
            "input_ids": torch.stack(batched_input_ids).to(self.model.device),
            "attention_mask": torch.stack(batched_attention_mask).to(self.model.device),
        }

        # FIX: Include token_type_ids in batch dict
        if batched_token_type_ids:
            batch_dict["token_type_ids"] = torch.stack(batched_token_type_ids).to(self.model.device)

        # Handle pixel values — may have variable shapes, try stacking
        if batched_pixel_values:
            try:
                batch_dict["pixel_values"] = torch.cat(batched_pixel_values, dim=0).to(
                    self.model.device, dtype=torch.bfloat16
                )
            except (RuntimeError, ValueError):
                # Variable image sizes — fall back to sequential
                return [self.generate(img, q) for img, q in zip(images, questions)]

        # Convert remaining tensors to bfloat16 where appropriate
        for key in batch_dict:
            if batch_dict[key].dtype == torch.float32:
                batch_dict[key] = batch_dict[key].to(dtype=torch.bfloat16)

        try:
            with torch.inference_mode():
                outputs = self.model.generate(
                    **batch_dict,
                    max_new_tokens=self.max_tokens,
                    do_sample=False,
                )

            responses = []
            for i, (output_seq, orig_len, pad_amount) in enumerate(
                zip(outputs, input_lens, [max_len - l for l in input_lens])
            ):
                # Skip padding + original input tokens
                generated = output_seq[max_len:]
                text = self.processor.decode(generated, skip_special_tokens=True)
                responses.append(text.strip())

            # FIX: Safeguard — if all outputs are empty, fall back to sequential
            if all(r == "" for r in responses):
                print(f"   ⚠️ All {len(responses)} batch outputs empty, falling back to sequential")
                return [self.generate(img, q) for img, q in zip(images, questions)]

            return responses

        except Exception as e:
            print(f"   ⚠️ Batch generation failed ({e}), falling back to sequential")
            return [self.generate(img, q) for img, q in zip(images, questions)]
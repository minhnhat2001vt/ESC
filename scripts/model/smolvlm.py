"""
SmolVLM2 model class for ESC pipeline.

Supports SmolVLM2-2.2B-Instruct (and 500M / 256M variants).
Uses AutoModelForImageTextToText + AutoProcessor with chat template.

Architecture notes:
  - Built on SmolLM2 (text) + shape-optimized SigLIP (vision).
  - Idefics3-family architecture, very lightweight (~5.2 GB GPU RAM for 2.2B).
  - Requires `num2words` package for the processor.
  - flash-attn is optional but recommended for speed.

API (per HF model card):
  - HF class: AutoModelForImageTextToText (resolves SmolVLMForConditionalGeneration)
  - Processor: AutoProcessor (no special kwargs needed)
  - Inputs: processor.apply_chat_template(messages, tokenize=True,
            return_dict=True, return_tensors="pt")
  - dtype: bfloat16 preferred

Requirements:
    pip install transformers accelerate bitsandbytes num2words
    pip install flash-attn  # optional, for speed

Integration:
  Drop at: /workspace/scripts/method/model/smolvlm.py
  Add to MODEL_REGISTRY:
    "smolvlm2-2b": {
        "name": "SmolVLM2-2.2B-Instruct",
        "hf_id": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        "type": "smolvlm2",
        "max_tokens": 512,
    },
  Add to MODEL_CLASSES: "smolvlm2": SmolVLM2Model
  Add import: from model.smolvlm import SmolVLM2Model
"""

import torch
from model.itt_base import ImageTextToTextModel


class SmolVLM2Model(ImageTextToTextModel):
    """SmolVLM2 (2.2B / 500M / 256M). Uses AutoModelForImageTextToText."""

    # SmolVLM2 prefers bfloat16 (ITT base defaults to float16)
    LOAD_DTYPE = torch.bfloat16

    def _load_hf_class(self):
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText
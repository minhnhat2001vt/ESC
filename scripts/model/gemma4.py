"""
Gemma-4 model class for ESC pipeline.

Uses HF's modern AutoModelForImageTextToText API (works for E2B, E4B, 26B-A4B, 31B).

Verified API (per HF docs, transformers >= 4.62 / Gemma 4 added 2026-04-01):
  - HF class: Gemma4ForConditionalGeneration (AutoModelForImageTextToText resolves it)
  - Processor: AutoProcessor with padding_side="left"
  - Inputs: processor.apply_chat_template(messages, tokenize=True,
            return_dict=True, return_tensors="pt")

Integration:
  Drop at: /workspace/scripts/method/model/gemma4.py
  Add to MODEL_REGISTRY:
    "gemma4-e4b": {
        "name": "Gemma-4-E4B-it",
        "hf_id": "google/gemma-4-E4B-it",
        "type": "gemma4",
        "max_tokens": 512,
    },
  Add to MODEL_CLASSES: "gemma4": Gemma4Model
"""

from model.itt_base import ImageTextToTextModel


class Gemma4Model(ImageTextToTextModel):
    """Gemma-4 (any size). Uses AutoModelForImageTextToText internally."""

    PROCESSOR_KWARGS = {"padding_side": "left"}

    def _load_hf_class(self):
        # AutoModelForImageTextToText resolves the right Gemma-4 class.
        # If the direct class is preferred for type safety:
        #   from transformers import Gemma4ForConditionalGeneration
        #   return Gemma4ForConditionalGeneration
        # but going through the Auto class is more forward-compatible.
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText
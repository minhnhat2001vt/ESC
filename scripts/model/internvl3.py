"""
InternVL3 model class for ESC pipeline.

⚠️ Important: Use the *-hf SUFFIX variants only:
  - OpenGVLab/InternVL3-1B-hf
  - OpenGVLab/InternVL3-2B-hf
  - OpenGVLab/InternVL3-8B-hf
  - OpenGVLab/InternVL3-9B-hf
  - OpenGVLab/InternVL3-14B-hf

The non-hf repos use trust_remote_code with custom modeling files, which is what
broke earlier with InternVL2.5-2B (transformers version mismatch). The -hf
versions use the native InternVLForConditionalGeneration class with no custom
code, which is forward-compatible with new transformers versions.


API (per HF docs, transformers >= 4.51):
  - HF class: InternVLForConditionalGeneration (AutoModelForImageTextToText resolves it)
  - Inputs: processor.apply_chat_template(messages, tokenize=True,
            return_dict=True, return_tensors="pt")

Integration:
  Drop at: /workspace/scripts/method/model/internvl3.py
  Add to MODEL_REGISTRY:
    "internvl3-2b": {
        "name": "InternVL3-2B",
        "hf_id": "OpenGVLab/InternVL3-2B-hf",
        "type": "internvl3",
        "max_tokens": 512,
    },
    "internvl3-8b": {
        "name": "InternVL3-8B",
        "hf_id": "OpenGVLab/InternVL3-8B-hf",
        "type": "internvl3",
        "max_tokens": 512,
    },
  Add to MODEL_CLASSES: "internvl3": InternVL3Model
"""

from model.itt_base import ImageTextToTextModel


class InternVL3Model(ImageTextToTextModel):
    """InternVL3 (-hf variants). Uses AutoModelForImageTextToText."""

    def _load_hf_class(self):
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText
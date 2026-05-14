"""
Qwen2.5-VL model class for ESC pipeline.

Qwen2.5-VL has a different HF class (Qwen2_5_VLForConditionalGeneration) than
Qwen2-VL (Qwen2VLForConditionalGeneration). Everything else — processor, chat
template, vision_info preprocessing — is the same, so we subclass Qwen2VLModel
and only override load().

Tested with:
  - Qwen/Qwen2.5-VL-3B-Instruct
  - Qwen/Qwen2.5-VL-7B-Instruct
  - transformers >= 4.49.0

Integration:
  1. Drop this file at: /workspace/scripts/method/model/qwen25.py
  2. Add to MODEL_CLASSES in inference_method1_rebut.py
     and vqa_inference_method1_rebut.py:
         from model.qwen25 import Qwen25VLModel
         "qwen25_vl": Qwen25VLModel,
  3. Update MODEL_REGISTRY entries for Qwen2.5-VL to use type "qwen25_vl"
     instead of "qwen2_vl":
         "qwen2.5-vl-3b": {
             "name": "Qwen2.5-VL-3B-Instruct",
             "hf_id": "Qwen/Qwen2.5-VL-3B-Instruct",
             "type": "qwen25_vl",          # ← changed from "qwen2_vl"
             "max_tokens": 512,
         },
"""

import torch
import warnings
from model.qwen import Qwen2VLModel

warnings.filterwarnings('ignore')


class Qwen25VLModel(Qwen2VLModel):
    """
    Qwen2.5-VL: same interface as Qwen2VLModel, only load() differs.

    Inherits everything else (processor setup, chat template, vision_info handling,
    generate(), generate_batch(), generate_multi(), generate_batch_multi()) from
    the parent class.
    """

    def load(self) -> None:
        from transformers import (
            AutoProcessor,
            Qwen2_5_VLForConditionalGeneration,
            BitsAndBytesConfig,
        )

        print(f"\n🚀 Loading {self.name}...")
        print(f"   Model: {self.config['hf_id']}")
        print(f"   4-bit: {self.load_4bit}")
        print(f"   Device: {self.device}")
        print(f"   Max pixels: {self.MAX_PIXELS:,} (~{int((self.MAX_PIXELS)**0.5)}x{int((self.MAX_PIXELS)**0.5)})")

        self.processor = AutoProcessor.from_pretrained(
            self.config["hf_id"],
            min_pixels=self.MIN_PIXELS,
            max_pixels=self.MAX_PIXELS,
        )

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

        # ← The only meaningful difference vs. Qwen2VLModel: this class
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.config["hf_id"], **model_kwargs
        )
        self.model.eval()
        self._is_loaded = True
        print(f"✅ {self.name} loaded successfully!\n")
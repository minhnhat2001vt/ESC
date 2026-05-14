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
# MINICPM-V MODEL (NEW)
# ============================================================================
class MiniCPMModel(BaseMLLM):
    """
    MiniCPM-V 2.6 (8B) - supports multi-image via msgs content list.
    HF usage pattern: model.chat(image=None, msgs=msgs, tokenizer=tokenizer, ...)
    """

    def load(self) -> None:
        from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

        print(f"\n🚀 Loading {self.name}...")
        print(f"   Model: {self.config['hf_id']}")
        print(f"   4-bit: {self.load_4bit}")
        print(f"   Device: {self.device}")

        model_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            # theo model card: sdpa hoặc flash_attention_2 (không eager)
            "attn_implementation": "sdpa",
        }

        if self.load_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            # cho phép code của bạn chạy giống các model khác
            model_kwargs["device_map"] = self.device
        else:
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["device_map"] = self.device

        self.model = AutoModel.from_pretrained(self.config["hf_id"], **model_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["hf_id"], trust_remote_code=True
        )

        # set generation defaults
        if hasattr(self.model, "generation_config"):
            self.model.generation_config.max_new_tokens = self.max_tokens
            self.model.generation_config.do_sample = False

        self.model.eval()
        self._is_loaded = True
        print(f"✅ {self.name} loaded successfully!\n")

    def generate(self, image: Image.Image, question: str) -> str:
        if not self.is_loaded:
            raise RuntimeError(f"{self.name} not loaded. Call load() first.")

        img = image.convert("RGB")
        msgs = [{"role": "user", "content": question}]

        with torch.inference_mode():
            res = self.model.chat(
                image=img,              # ✅ đưa ảnh vào đây
                msgs=msgs,
                tokenizer=self.tokenizer,
            )
        return str(res).strip()

    def generate_batch(self, images: List[Image.Image], questions: List[str]) -> List[str]:
        """
        Chưa có batching chuẩn hoá qua transformers.generate trong repo này,
        nên chạy tuần tự (an toàn, giống InternVL/CogVLM2).
        """
        if not self.is_loaded:
            raise RuntimeError(f"{self.name} not loaded. Call load() first.")

        outs = []
        for img, q in zip(images, questions):
            outs.append(self.generate(img, q))
        return outs

    def unload(self) -> None:
        # reuse Base unload but also delete tokenizer
        if hasattr(self, "tokenizer"):
            del self.tokenizer
            self.tokenizer = None
        super().unload()

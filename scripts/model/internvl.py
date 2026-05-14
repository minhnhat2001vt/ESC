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
# INTERNVL2.5-8B MODEL (NEW - Replaces Gemini)
# ============================================================================
class InternVLModel(BaseMLLM):

    def load(self) -> None:
        from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        print(f"\n🚀 Loading {self.name}...")
        print(f"   Model: {self.config['hf_id']}")
        print(f"   4-bit: {self.load_4bit}")
        print(f"   Device: {self.device}")

        if self.load_4bit:
            # 4-bit requires device_map for bitsandbytes
            model_kwargs = {
                "device_map": self.device,
                "low_cpu_mem_usage": True,
                "trust_remote_code": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                ),
            }
            self.model = AutoModel.from_pretrained(
                self.config["hf_id"], **model_kwargs
            )
        else:
            # Load on CPU first to avoid meta tensor .item() crash in InternVL's
            # vision encoder init, then move to GPU
            self.model = AutoModel.from_pretrained(
                self.config["hf_id"],
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=False,
            ).cuda()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["hf_id"], trust_remote_code=True
        )
        
        self.model.eval()
        
        # InternVL uses a specific image transform
        self.image_size = self.model.config.vision_config.image_size
        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        
        self._is_loaded = True
        print(f"✅ {self.name} loaded successfully!\n")

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image using InternVL's transform."""
        pixel_values = self.transform(image).unsqueeze(0)
        return pixel_values.to(torch.float16)

    def generate(self, image: Image.Image, question: str) -> str:
        if not self.is_loaded:
            raise RuntimeError(f"{self.name} not loaded. Call load() first.")

        pixel_values = self._preprocess_image(image).to(self.model.device)
        
        # InternVL uses a specific format with <image> token
        prompt = f"<image>\n{question}"
        
        with torch.inference_mode():
            response = self.model.chat(
                self.tokenizer,
                pixel_values=pixel_values,
                question=prompt,
                generation_config={
                    'max_new_tokens': self.max_tokens,
                    'do_sample': False,
                }
            )
        
        return response.strip()

    def generate_batch(self, images: List[Image.Image], questions: List[str]) -> List[str]:
        """
        InternVL doesn't have native batch support, so we process sequentially.
        This could be optimized with custom batching in the future.
        """
        if not self.is_loaded:
            raise RuntimeError(f"{self.name} not loaded. Call load() first.")
        
        responses = []
        for image, question in zip(images, questions):
            response = self.generate(image, question)
            responses.append(response)
        
        return responses
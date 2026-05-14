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
# COGVLM2-LLAMA3-19B MODEL (High vulnerability, THUDM alignment)
# ============================================================================
class CogVLM2Model(BaseMLLM):
    """
    CogVLM2-LLaMA3-Chat-19B from THUDM.
    Uses custom modeling code (trust_remote_code=True).
    Original CogVLM had ~47% ASR in MM-SafetyBench.
    
    NOTE: CogVLM2 does NOT support device_map="auto" well.
    With 4-bit quantization it fits on a single 40GB GPU (~12GB).
    Without quantization it needs ~38GB in fp16.
    """

    def load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        print(f"\n🚀 Loading {self.name}...")
        print(f"   Model: {self.config['hf_id']}")
        print(f"   4-bit: {self.load_4bit}")
        print(f"   Device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["hf_id"], trust_remote_code=True
        )

        # Determine torch dtype
        self.torch_type = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )

        model_kwargs = {
            "torch_dtype": self.torch_type,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        if self.load_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.torch_type,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs["device_map"] = self.device
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config["hf_id"], **model_kwargs
            )
        else:
            # CogVLM2 prefers explicit .to(DEVICE) over device_map="auto"
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config["hf_id"], **model_kwargs
            ).to("cuda")

        self.model.eval()
        self._is_loaded = True
        print(f"✅ {self.name} loaded successfully!\n")

    def _get_device(self):
        """Get the device where the model's main parameters reside."""
        if hasattr(self.model, 'device'):
            return self.model.device
        return next(self.model.parameters()).device

    def generate(self, image: Image.Image, question: str) -> str:
        if not self.is_loaded:
            raise RuntimeError(f"{self.name} not loaded. Call load() first.")

        device = self._get_device()
        image_rgb = image.convert('RGB')

        # CogVLM2 uses build_conversation_input_ids for image+text
        input_by_model = self.model.build_conversation_input_ids(
            self.tokenizer,
            query=question,
            history=[],
            images=[image_rgb],
            template_version='chat'
        )

        inputs = {
            'input_ids': input_by_model['input_ids'].unsqueeze(0).to(device),
            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(device),
            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(device),
            'images': [[input_by_model['images'][0].to(device).to(self.torch_type)]],
        }

        gen_kwargs = {
            "max_new_tokens": self.max_tokens,
            "do_sample": False,
            "pad_token_id": 128002,  # LLaMA-3 pad token
        }

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Clean up end-of-text markers
        response = response.split("<|end_of_text|>")[0].strip()
        return response

    def generate_batch(self, images: List[Image.Image], questions: List[str]) -> List[str]:
        """
        CogVLM2 does NOT support native batching due to its custom
        build_conversation_input_ids interface. Process sequentially.
        """
        if not self.is_loaded:
            raise RuntimeError(f"{self.name} not loaded. Call load() first.")

        responses = []
        for image, question in zip(images, questions):
            response = self.generate(image, question)
            responses.append(response)
        return responses


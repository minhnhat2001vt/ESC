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
warnings.filterwarnings('ignore')


# ============================================================================
# ABSTRACT BASE MODEL
# ============================================================================
class BaseMLLM(ABC):
    """
    Abstract base class for all Multimodal LLMs.
    Provides a common interface: load(), generate(), generate_batch(), unload().
    Supports context manager usage: `with model: ...`
    """

    def __init__(self, config: Dict[str, Any], load_4bit: bool = True, device: str = "auto"):
        self.config = config
        self.name = config.get("name", "Unknown")
        self.max_tokens = config.get("max_tokens", 512)
        self.load_4bit = load_4bit
        self.device = device
        self.model = None
        self.processor = None
        self._is_loaded = False

    @abstractmethod
    def load(self) -> None:
        """Load model weights and processor into memory."""
        pass

    @abstractmethod
    def generate(self, image: Image.Image, question: str) -> str:
        """Generate a single response given an image and question."""
        pass

    @abstractmethod
    def generate_batch(self, images: List[Image.Image], questions: List[str]) -> List[str]:
        """Generate responses for a batch of image-question pairs."""
        pass

    def unload(self) -> None:
        """Unload the model from memory and free GPU."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        self._is_loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"   {self.name} unloaded from memory")

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload()

"""
Qwen3-VL Thinking model class for ESC pipeline.

Thin subclass of the standard Qwen3-VL wrapper that:
  1. Decodes with skip_special_tokens=False to preserve <think>...</think> tags
  2. Strips only the generation-control tokens (bos/eos/pad) manually
  3. Bumps default max_new_tokens to 8192 to allow room for thinking traces

The raw output keeps <think>...</think> in the response text.  A separate
analysis script (analyze_thinking.py) can then split thinking vs. answer
and compute token-count statistics across the 4 prompt conditions.

Usage as target model in ESC pipeline:
    python inference_method1_rebut.py \
        --model_a qwen3-vl-8b-thinking \
        --model_a_results /path/to/baseline.json \
        --model_b gemma3-12b \
        --benchmark vlsafe \
        --prompt_source emotion \
        ...

Integration:
  Drop at: /workspace/scripts/method/model/qwen3_thinking.py
  Add to MODEL_REGISTRY:
    "qwen3-vl-8b-thinking": {
        "name": "Qwen3-VL-8B-Thinking",
        "hf_id": "Qwen/Qwen3-VL-8B-Thinking",
        "type": "qwen3_vl_thinking",
        "max_tokens": 8192,
    },
  Add to MODEL_CLASSES: "qwen3_vl_thinking": Qwen3VLThinkingModel
  Add import: from model.qwen3_thinking import Qwen3VLThinkingModel
"""

import torch
from PIL import Image
from typing import List, Dict
from model.itt_base import ImageTextToTextModel


# Tokens to strip when NOT using skip_special_tokens
_STRIP_TOKENS = {
    "<|endoftext|>", "<|end|>", "<|im_end|>", "<|im_start|>",
    "<s>", "</s>", "<pad>", "<eos>",
}


class Qwen3VLThinkingModel(ImageTextToTextModel):
    """Qwen3-VL-Thinking. Preserves <think> tags in output for analysis."""

    LOAD_DTYPE = torch.bfloat16

    def _load_hf_class(self):
        from transformers import Qwen3VLForConditionalGeneration
        return Qwen3VLForConditionalGeneration

    def _generate_inner(self, messages_list: List[List[Dict]]) -> List[str]:
        """Override: decode with skip_special_tokens=False to keep <think> tags."""
        responses = []
        for messages in messages_list:
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)

            for key in ("pixel_values", "pixel_values_videos", "image_grid_thw"):
                if key in inputs and inputs[key] is not None and inputs[key].is_floating_point():
                    inputs[key] = inputs[key].to(self.LOAD_DTYPE)

            in_len = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    do_sample=False,
                )
            out_ids = outputs[0][in_len:]

            # Decode WITHOUT stripping special tokens — preserves <think>...</think>
            raw_text = self.processor.decode(out_ids, skip_special_tokens=False)

            # Manually strip only control tokens, keep <think> tags
            for tok in _STRIP_TOKENS:
                raw_text = raw_text.replace(tok, "")
            text = raw_text.strip()

            responses.append(text)
        return responses
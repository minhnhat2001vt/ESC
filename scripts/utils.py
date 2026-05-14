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





# ============================================================================
# MULTI-GPU PARALLEL EXECUTION
# ============================================================================
def run_model_on_gpu(
    model_name: str,
    gpu_id: int,
    findings: List[str],
    batch_size: int,
    checkpoint_interval: int,
    max_samples: Optional[int],
    skip_neutral: bool,
    image_dir_override: Optional[str],
    load_4bit: bool,
) -> None:
    """
    Worker function for parallel execution.
    Runs a single model on a specified GPU.
    """
    # Set GPU visibility
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    print(f"\n{'='*80}")
    print(f"🚀 WORKER: GPU {gpu_id} — Model: {model_name}")
    print(f"{'='*80}\n")
    
    # Create model with device set to cuda:0 (since CUDA_VISIBLE_DEVICES makes it the only GPU)
    model = ModelFactory.create(
        model_name=model_name,
        load_4bit=load_4bit,
        device="cuda:0",  # Always use cuda:0 after setting CUDA_VISIBLE_DEVICES
    )
    
    try:
        model.load()
        
        for finding_name in findings:
            run_finding(
                finding_name=finding_name,
                model=model,
                batch_size=batch_size,
                checkpoint_interval=checkpoint_interval,
                max_samples=max_samples,
                skip_neutral=skip_neutral,
                image_dir_override=image_dir_override,
            )
    
    finally:
        model.unload()
    
    print(f"\n{'='*80}")
    print(f"✅ WORKER GPU {gpu_id} COMPLETE")
    print(f"{'='*80}\n")


def run_parallel(
    models: List[str],
    gpus: List[int],
    findings: List[str],
    batch_sizes: List[int],
    checkpoint_interval: int,
    max_samples: Optional[int],
    skip_neutral: bool,
    image_dir_override: Optional[str],
    load_4bit: bool,
) -> None:
    """
    Run multiple models in parallel on different GPUs.
    """
    if len(models) != len(gpus):
        raise ValueError(f"Number of models ({len(models)}) must match number of GPUs ({len(gpus)})")
    
    if len(batch_sizes) == 1:
        # Same batch size for all models
        batch_sizes = batch_sizes * len(models)
    elif len(batch_sizes) != len(models):
        raise ValueError(f"Number of batch sizes ({len(batch_sizes)}) must be 1 or match number of models ({len(models)})")
    
    print(f"\n{'='*80}")
    print(f"🚀 PARALLEL EXECUTION")
    print(f"{'='*80}")
    print(f"Models: {models}")
    print(f"GPUs: {gpus}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Findings: {findings}")
    print(f"{'='*80}\n")
    
    # Create processes
    processes = []
    for model_name, gpu_id, batch_size in zip(models, gpus, batch_sizes):
        p = mp.Process(
            target=run_model_on_gpu,
            args=(
                model_name,
                gpu_id,
                findings,
                batch_size,
                checkpoint_interval,
                max_samples,
                skip_neutral,
                image_dir_override,
                load_4bit,
            )
        )
        processes.append(p)
        p.start()
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    print(f"\n{'='*80}")
    print(f"✅ ALL PARALLEL WORKERS COMPLETE")
    print(f"{'='*80}\n")

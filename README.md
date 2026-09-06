# ESC: Emotional Self-Correction for Reliable Vision-Language Models

Official implementation and evaluation artifacts for **ESC: Emotional Self-Correction for Reliable Vision-Language Models**, accepted at **ECCV 2026**.

- [Project page](https://genai4e.github.io/ESC/)
- Paper and code links are available from the project page.

## Overview

Vision-language models can produce fluent answers even when the visual evidence disagrees. ESC uses emotional feedback as a structured test-time control signal that prompts a model to reconsider uncertain or incorrect responses—without retraining the target model or revealing the ground-truth answer.

The ESC pipeline has four stages:

1. **Verify** the initial response.
2. **Select** an emotional feedback cue.
3. **Revise** the response using the cue.
4. **Decide** whether to retain the original response or use the revision.

## Repository structure

```text
.
├── scripts/
│   ├── model/          # Model adapters
│   ├── prepare_data/   # Benchmark preprocessing
│   ├── method/         # ESC inference
│   └── eval/           # Benchmark evaluation
├── results/            # Raw experimental outputs and summarized results
├── logs/               # Original experiment logs
└── environment.yml     # Conda environment used for the experiments
```

The repository preserves the raw result files used during experimentation. They are intentionally retained for traceability, although they make the repository large.

## Installation

The experiments were run on Linux with NVIDIA GPUs and Python 3.10.

```bash
conda env create -f environment.yml
conda activate esc
```

Some model checkpoints require accepting their license on Hugging Face and authenticating locally with a Hugging Face token. API-based evaluators read credentials from environment variables; credentials must never be committed to the repository.

## Benchmarks

The codebase contains preparation, inference, or evaluation support for the following benchmark families:

- Safety: VLSafe, MMSafetyBench, FigStep
- Hallucination: POPE, HallusionBench
- Vision-centric perception: RealWorldQA, MMVP, BLINK, MME
- Multimodal reasoning: MM-Vet, MathVista, MMStar, AI2D, MMMU

Benchmark datasets are not redistributed. Download them from their official sources and follow their respective licenses and terms.

## Data and output paths

By default, the supported scripts resolve paths relative to the repository:

| Purpose | Default | Override |
|---|---|---|
| Original datasets | `original_data/` | `ESC_DATA_ROOT` |
| Processed datasets | `processed_data/` | `ESC_PROCESSED_ROOT` |
| Results | `results/` | `ESC_RESULTS_ROOT` |
| Logs | `logs/` | `ESC_LOGS_ROOT` |

For example:

```bash
export ESC_DATA_ROOT=/data/esc/original_data
export ESC_PROCESSED_ROOT=/data/esc/processed_data
export ESC_RESULTS_ROOT=/data/esc/results
```

## Reproducing the workflow

The current research code follows this sequence:

### 1. Prepare a benchmark

For example:

```bash
python scripts/prepare_data/prepare_vlsafe.py --help
python scripts/prepare_data/prepare_pope.py --help
python scripts/prepare_data/prepare_ai2d.py --help
python scripts/prepare_data/prepare_mmmu.py --help
```

### 2. Run baseline inference

```bash
python scripts/inference_baseline.py --list_models
python scripts/inference_baseline.py --help
```

### 3. Run ESC

Safety-oriented and general VQA benchmarks currently use separate entry points:

```bash
python scripts/method/run_esc_safety.py --help
python scripts/method/run_esc_vqa.py --help
```

Both scripts support `--test_mode` or `--max_samples` for a small validation run before a full evaluation.

### 4. Evaluate outputs

```bash
python scripts/eval/eval_vlsafe.py --help
python scripts/eval/eval_pope.py --help
python scripts/eval/eval_ai2d.py --help
python scripts/eval/eval_mmmu.py --help
```

Additional benchmark-specific evaluators are available in `scripts/eval/`.

> **Release note:** This repository originated as an internal experiment workspace. The release branch is consolidating paths, canonical entry points, and reproducibility commands before the repository is made public. See [the release checklist](docs/RELEASE_CHECKLIST.md) for the remaining work.

## Results

ESC was evaluated across safety, hallucination, perception, and multimodal reasoning benchmarks. Representative improvements reported in the paper include:

- VLSafe attack success rate: **−46.3 percentage points**
- Adversarial POPE accuracy: **+29.6 points**
- RealWorldQA accuracy: **+14.5 points**
- AI2D accuracy: **+2.5 points**

See the paper and project page for the complete experimental protocol and results.

## Citation

If this repository is useful in your research, please cite the ECCV 2026 paper. Machine-readable citation metadata is provided in [`CITATION.cff`](CITATION.cff).

## License

This project is released under the [MIT License](LICENSE).

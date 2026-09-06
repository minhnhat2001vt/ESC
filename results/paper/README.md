# Paper result summaries

This directory contains compact evaluated summaries supplied for the paper release. It does not redistribute benchmark datasets.

## AI2D

| Model | Condition | Correct / total | Accuracy |
|---|---|---:|---:|
| Qwen2-VL-7B | Baseline | 1883 / 3088 | 60.98% |
| Qwen2-VL-7B | ESC | 1931 / 3088 | 62.53% |
| LLaVA-1.5-7B | ESC | 1659 / 3088 | 53.72% |

The uploaded bundle did not contain the evaluated LLaVA AI2D baseline summary, so that comparison is intentionally not reconstructed or inferred here.

## MMMU

These files evaluate the 857 single-image validation examples prepared by `scripts/prepare_data/prepare_mmmu.py`.

| Model | Condition | Correct / total | Accuracy |
|---|---|---:|---:|
| LLaVA-1.5-7B | Baseline | 297 / 857 | 34.66% |
| LLaVA-1.5-7B | ESC | 298 / 857 | 34.77% |
| Qwen2-VL-7B | Baseline | 340 / 857 | 39.67% |
| Qwen2-VL-7B | ESC | 343 / 857 | 40.02% |

## Raw-output examples

The files in `examples/ai2d_baseline_sample.json` and `examples/mmmu_baseline_sample.json` contain three records each and are included only to demonstrate the input schema expected by the evaluators. They are not full benchmark results.

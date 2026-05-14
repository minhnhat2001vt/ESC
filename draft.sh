#!/usr/bin/env bash
# Quantize both QAT and no-QAT checkpoints from a qat_ablation run (same as qat_ablation.sh).
# Outputs go to int8/with_qat/ and int8/no_qat/ so filenames do not overwrite.
#
# Usage:
#   ./quantize_qat_ablation_ckpt.sh <data_dir> <results_dir> [channels] [loss_version] [no_qat_ckpt] [with_qat_ckpt]
#
# If optional checkpoint paths are omitted, picks newest match under results_dir/checkpoints/ for:
#   qat_ablation_no_qat_c<channels>_loss<loss>_epoch*.ckpt
#   qat_ablation_with_qat_c<channels>_loss<loss>_epoch*.ckpt
set -euo pipefail

DATA_DIR="/workspace/huynt/mixedmodel/dataset/dped/dped"
RESULTS_DIR="/workspace/huynt/result"
CHANNELS="${3:-32}"
LOSS_VERSION="${4:-2}"
EXPLICIT_NO_QAT="/workspace/huynt/result/qat_ablation_c32_loss2/checkpoints/qat_ablation_no_qat_c32_loss2_epoch45_psnr23.53.ckpt"
EXPLICIT_WITH_QAT="/workspace/huynt/result/qat_ablation_c32_loss2/checkpoints/qat_ablation_with_qat_c32_loss2_epoch37_psnr23.27.ckpt"

ABLATION_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="/workspace/huynt/Source-Codes"

mkdir -p "${RESULTS_DIR}"
RESULTS_DIR="$(cd "${RESULTS_DIR}" && pwd)"
CKPT_DIR="${RESULTS_DIR}/checkpoints"
INT8_DIR="${RESULTS_DIR}/int8"
mkdir -p "${CKPT_DIR}" "${INT8_DIR}"

RUN_NO_QAT="qat_ablation_no_qat_c${CHANNELS}_loss${LOSS_VERSION}"
RUN_WITH_QAT="qat_ablation_with_qat_c${CHANNELS}_loss${LOSS_VERSION}"

_abs_path() {
  local p="$1"
  [[ -n "$p" ]] || return 1
  [[ -f "$p" ]] || {
    echo "ERROR: not a file: $p"
    exit 1
  }
  echo "$(cd "$(dirname "$p")" && pwd)/$(basename "$p")"
}

_pick_newest() {
  local run_prefix="$1"
  shopt -s nullglob
  local matches=("${CKPT_DIR}/${run_prefix}"_epoch*.ckpt)
  shopt -u nullglob
  if [[ ${#matches[@]} -eq 0 ]]; then
    echo ""
    return 1
  fi
  ls -t "${matches[@]}" | sed -n '1p'
}

resolve_no_qat() {
  if [[ -n "${EXPLICIT_NO_QAT}" ]]; then
    _abs_path "${EXPLICIT_NO_QAT}"
  else
    local f
    f="$(_pick_newest "${RUN_NO_QAT}")" || true
    if [[ -z "${f}" ]]; then
      echo "ERROR: no checkpoint matching: ${CKPT_DIR}/${RUN_NO_QAT}_epoch*.ckpt"
      exit 1
    fi
    echo "${f}"
  fi
}

resolve_with_qat() {
  if [[ -n "${EXPLICIT_WITH_QAT}" ]]; then
    _abs_path "${EXPLICIT_WITH_QAT}"
  else
    local f
    f="$(_pick_newest "${RUN_WITH_QAT}")" || true
    if [[ -z "${f}" ]]; then
      echo "ERROR: no checkpoint matching: ${CKPT_DIR}/${RUN_WITH_QAT}_epoch*.ckpt"
      exit 1
    fi
    echo "${f}"
  fi
}

NO_QAT_CKPT="$(resolve_no_qat)"
WITH_QAT_CKPT="$(resolve_with_qat)"

SAVE_NO_QAT="${INT8_DIR}/no_qat"
SAVE_WITH_QAT="${INT8_DIR}/with_qat"
mkdir -p "${SAVE_NO_QAT}" "${SAVE_WITH_QAT}"

_run_quantize() {
  local label="$1"
  local ckpt="$2"
  local save_path="$3"
  echo "============================================================"
  echo "  ${label}"
  echo "============================================================"
  echo "  data_dir:   ${DATA_DIR}"
  echo "  save_path:  ${save_path}"
  echo "  channels:   ${CHANNELS}"
  echo "  checkpoint: ${ckpt}"
  echo "============================================================"
  python "${ROOT_DIR}/quantize.py" \
    --ablation_model hybrid_base \
    --ckpt_path "${ckpt}" \
    --channels "${CHANNELS}" \
    --data_dir "${DATA_DIR}" \
    --save_path "${save_path}"
}

_run_quantize "Quantize no-QAT (FP32) checkpoint → INT8" "${NO_QAT_CKPT}" "${SAVE_NO_QAT}"
_run_quantize "Quantize with-QAT checkpoint → INT8" "${WITH_QAT_CKPT}" "${SAVE_WITH_QAT}"

echo "Done."
echo "  no-QAT INT8:  ${SAVE_NO_QAT}"
echo "  with-QAT INT8: ${SAVE_WITH_QAT}"
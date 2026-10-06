#!/usr/bin/env bash
# Cross-perturbation generalization on Norman et al. (2019), CRISPRa (Table 5).
# Prepare the data first:  python datasets/prepare_norman.py
#   bash scripts/run_norman.sh [fold]        # default: fold 0
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

OUT="$EXP_DIR/norman"
FOLD="${1:-0}"

BASE=( "data.kwargs.toml_config_path=$ROOT/datasets/norman_pertsplit_30_10_60_fold${FOLD}.toml"
       "${DATA_ARGS[@]}" "$NORMAN_CONTROL" "${STATE_MODEL[@]}"
       "training.max_steps=10001" "training.val_freq=500" "training.ckpt_every_n_steps=5000"
       "training.split_aware_rag=true" )

train "$OUT" "fold${FOLD}_state_genept" "${BASE[@]}" "${GENEPT[@]}"
train "$OUT" "fold${FOLD}_ptrag"        "${BASE[@]}" "${PTRAG[@]}" "training.topk_rag=16" "training.gumbel_sparsity_weight=1.0"

evaluate "$OUT/fold${FOLD}_state_genept"
evaluate "$OUT/fold${FOLD}_ptrag"

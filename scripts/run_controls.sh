#!/usr/bin/env bash
# Additional controls in the cross-perturbation setting (Appendix F.7), one fold, K = 32:
#   - retrieval pool available at test time (full / train+validation / train only)
#   - 60% of the retrieved candidates replaced with hard negatives
#   - perturbation encoder frozen, only the selector and the projector trained
#   bash scripts/run_controls.sh [fold]      # default: fold 1
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

OUT="$EXP_DIR/controls"
FOLD="${1:-1}"

BASE=( "data.kwargs.toml_config_path=$ROOT/datasets/replogle_nadig_all_pertsplit_30_10_60_fold${FOLD}.toml"
       "${DATA_ARGS[@]}" "$REPLOGLE_CONTROL" "${STATE_ARGS[@]}" "training.split_aware_rag=true" )
PTRAG_K32=( "${PTRAG[@]}" "training.topk_rag=32" "training.gumbel_sparsity_weight=1.0" )

train "$OUT" "fold${FOLD}_state_genept" "${BASE[@]}" "${GENEPT[@]}"
train "$OUT" "fold${FOLD}_ptrag_k32"    "${BASE[@]}" "${PTRAG_K32[@]}"
train "$OUT" "fold${FOLD}_ptrag_k32_frozen_encoder" "${BASE[@]}" "${PTRAG_K32[@]}" \
      "model.kwargs.freeze_pert_encoder=true" \
      "training.init_pert_encoder_from=$OUT/fold${FOLD}_state_genept/checkpoints/final.ckpt"

evaluate "$OUT/fold${FOLD}_state_genept"
evaluate "$OUT/fold${FOLD}_ptrag_k32"                                   # full pool
evaluate "$OUT/fold${FOLD}_ptrag_k32" --retrieval-pool trainval
evaluate "$OUT/fold${FOLD}_ptrag_k32" --retrieval-pool train
evaluate "$OUT/fold${FOLD}_ptrag_k32" --hard-negative-fraction 0.6
evaluate "$OUT/fold${FOLD}_ptrag_k32_frozen_encoder"

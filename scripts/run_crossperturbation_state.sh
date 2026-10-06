#!/usr/bin/env bash
# Cross-perturbation generalization, STATE-style backbone (Table 4).
#   bash scripts/run_crossperturbation_state.sh [fold ...]   # default: folds 0 1 2
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

OUT="$EXP_DIR/crossperturbation_state"
FOLDS=("$@"); [ ${#FOLDS[@]} -eq 0 ] && FOLDS=(0 1 2)

for FOLD in "${FOLDS[@]}"; do
  BASE=( "data.kwargs.toml_config_path=$ROOT/datasets/replogle_nadig_all_pertsplit_30_10_60_fold${FOLD}.toml"
         "${DATA_ARGS[@]}" "$REPLOGLE_CONTROL" "${STATE_ARGS[@]}" "training.split_aware_rag=true" )

  train "$OUT" "fold${FOLD}_state_genept" "${BASE[@]}" "${GENEPT[@]}"
  train "$OUT" "fold${FOLD}_ptrag"        "${BASE[@]}" "${PTRAG[@]}" "training.topk_rag=128" "training.gumbel_sparsity_weight=1.0"

  evaluate "$OUT/fold${FOLD}_state_genept"
  evaluate "$OUT/fold${FOLD}_ptrag"
done

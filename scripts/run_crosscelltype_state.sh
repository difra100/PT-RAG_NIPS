#!/usr/bin/env bash
# Cross-cell-type generalization, STATE-style backbone (Table 2).
#   bash scripts/run_crosscelltype_state.sh [cell ...]      # default: all four cell lines
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

OUT="$EXP_DIR/crosscelltype_state"
CELLS=("$@"); [ ${#CELLS[@]} -eq 0 ] && CELLS=(hepg2 k562 jurkat rpe1)

for CELL in "${CELLS[@]}"; do
  BASE=( "data.kwargs.toml_config_path=${CELL_TOML[$CELL]}" "${DATA_ARGS[@]}" "$REPLOGLE_CONTROL" "${STATE_ARGS[@]}" )

  train "$OUT" "${CELL}_state"        "${BASE[@]}"
  train "$OUT" "${CELL}_state_genept" "${BASE[@]}" "${GENEPT[@]}"
  train "$OUT" "${CELL}_naive_rag"    "${BASE[@]}" "${NAIVE_RAG[@]}" "training.topk_rag=32"
  train "$OUT" "${CELL}_ptrag"        "${BASE[@]}" "${PTRAG[@]}" "training.topk_rag=32" "training.gumbel_sparsity_weight=0.1"

  for RUN in state state_genept naive_rag ptrag; do evaluate "$OUT/${CELL}_${RUN}"; done
done

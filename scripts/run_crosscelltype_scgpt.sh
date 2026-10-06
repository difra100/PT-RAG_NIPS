#!/usr/bin/env bash
# Cross-cell-type generalization, scGPT backbone (Table 3).
#   bash scripts/run_crosscelltype_scgpt.sh [cell ...]      # default: all four cell lines
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

OUT="$EXP_DIR/crosscelltype_scgpt"
CELLS=("$@"); [ ${#CELLS[@]} -eq 0 ] && CELLS=(hepg2 k562 jurkat rpe1)

for CELL in "${CELLS[@]}"; do
  BASE=( "data.kwargs.toml_config_path=${CELL_TOML[$CELL]}" "${DATA_ARGS[@]}" "$REPLOGLE_CONTROL" "${SCGPT_ARGS[@]}" )
  RAG=( "${BASE[@]}" "model=scgpt-genetic-rag" "training.use_genept=true"
        "training.retrieve_than_predict=true" "training.split_aware_rag=true" )

  train "$OUT" "${CELL}_scgpt"        "${BASE[@]}" "model=scgpt-genetic"
  train "$OUT" "${CELL}_scgpt_genept" "${RAG[@]}" "training.rag=false"
  train "$OUT" "${CELL}_scgpt_ptrag"  "${RAG[@]}" "training.rag=true" "training.differentiable_rag=true" \
        "training.gumbel_sparsity_loss=true" "training.topk_rag=128" "training.gumbel_sparsity_weight=1.0"

  for RUN in scgpt scgpt_genept scgpt_ptrag; do evaluate "$OUT/${CELL}_${RUN}"; done
done

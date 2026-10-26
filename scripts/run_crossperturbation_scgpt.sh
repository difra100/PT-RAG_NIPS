#!/usr/bin/env bash
# Cross-perturbation generalization with the scGPT backbone (all parameters trained).
#   bash scripts/run_crossperturbation_scgpt.sh replogle [fold ...]   # default: folds 0 1 2
#   bash scripts/run_crossperturbation_scgpt.sh norman   [fold ...]   # default: fold 0
# Optional: K (candidates), LAMBDA (sparsity weight), MAX_STEPS (training steps).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

DATASET="${1:?usage: run_crossperturbation_scgpt.sh replogle|norman [fold ...]}"; shift
FOLDS=("$@")
case "$DATASET" in
  replogle)
    [ ${#FOLDS[@]} -eq 0 ] && FOLDS=(0 1 2)
    TOML="$ROOT/datasets/replogle_nadig_all_pertsplit_30_10_60_fold%s.toml"
    CONTROL="$REPLOGLE_CONTROL"; STEPS="${MAX_STEPS:-25001}"; VAL=2000; K="${K:-128}" ;;
  norman)
    [ ${#FOLDS[@]} -eq 0 ] && FOLDS=(0)
    TOML="$ROOT/datasets/norman_pertsplit_30_10_60_fold%s.toml"
    CONTROL="$NORMAN_CONTROL"; STEPS="${MAX_STEPS:-10001}"; VAL=500; K="${K:-16}" ;;
  *) echo "unknown dataset: $DATASET (use replogle or norman)"; exit 1 ;;
esac
LAMBDA="${LAMBDA:-1.0}"
OUT="$EXP_DIR/crossperturbation_scgpt_${DATASET}"

for FOLD in "${FOLDS[@]}"; do
  BASE=( "data.kwargs.toml_config_path=$(printf "$TOML" "$FOLD")" "${DATA_ARGS[@]}" "$CONTROL"
         "training.max_steps=$STEPS" "training.val_freq=$VAL" "training.ckpt_every_n_steps=$(( STEPS / 2 ))"
         "training.batch_size=64" "training.lr=5e-5" )
  RAG=( "${BASE[@]}" "model=scgpt-genetic-rag" "training.use_genept=true"
        "training.retrieve_than_predict=true" "training.split_aware_rag=true" )

  train "$OUT" "fold${FOLD}_scgpt"        "${BASE[@]}" "model=scgpt-genetic"
  train "$OUT" "fold${FOLD}_scgpt_genept" "${RAG[@]}" "training.rag=false"
  train "$OUT" "fold${FOLD}_scgpt_ptrag"  "${RAG[@]}" "training.rag=true" "training.differentiable_rag=true" \
        "training.gumbel_sparsity_loss=true" "training.topk_rag=$K" "training.gumbel_sparsity_weight=$LAMBDA"

  for RUN in scgpt scgpt_genept scgpt_ptrag; do evaluate "$OUT/fold${FOLD}_${RUN}"; done
done

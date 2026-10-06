#!/usr/bin/env bash
# Shared settings for the experiment scripts. Source this file; do not run it.
#
# Environment variables (all optional):
#   EXP_DIR       where runs are written        (default: <repo>/experiments)
#   USE_WANDB     true/false                    (default: false)
#   NUM_WORKERS   data-loading workers          (default: 4)
#   CUDA_VISIBLE_DEVICES   GPU to use

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"   # dataset, GenePT and scGPT paths are relative to the repository root
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

EXP_DIR="${EXP_DIR:-$ROOT/experiments}"
USE_WANDB="${USE_WANDB:-false}"
NUM_WORKERS="${NUM_WORKERS:-4}"

declare -A CELL_TOML=(
  [hepg2]="$ROOT/datasets/repogle_nadig.toml"
  [k562]="$ROOT/datasets/repogle_nadig_k562.toml"
  [jurkat]="$ROOT/datasets/repogle_nadig_jurkat.toml"
  [rpe1]="$ROOT/datasets/repogle_nadig_rpe1.toml"
)

# Data arguments shared by all runs
DATA_ARGS=(
  "data.kwargs.embed_key=X_hvg" "data.kwargs.num_workers=$NUM_WORKERS"
  "data.kwargs.output_space=gene" "data.kwargs.batch_col=gem_group"
  "data.kwargs.pert_col=gene" "data.kwargs.cell_type_key=cell_line"
)
REPLOGLE_CONTROL="data.kwargs.control_pert=non-targeting"
NORMAN_CONTROL="data.kwargs.control_pert=control"

# STATE-style models: the transformer generator is randomly initialised and kept frozen
STATE_MODEL=(
  "model=state" "model.kwargs.cell_set_len=64" "model.kwargs.hidden_dim=128"
  "model.kwargs.batch_encoder=true" "model.kwargs.freeze_pert_backbone=true"
  "training.batch_size=64" "training.lr=1e-3"
)
STATE_SCHEDULE=( "training.max_steps=50001" "training.val_freq=2000" "training.ckpt_every_n_steps=25000" )
STATE_ARGS=( "${STATE_MODEL[@]}" "${STATE_SCHEDULE[@]}" )

# scGPT models: all parameters are trained
SCGPT_ARGS=(
  "training.max_steps=25001" "training.val_freq=2000" "training.ckpt_every_n_steps=12500"
  "training.batch_size=64" "training.lr=5e-5"
)

# Method-specific flags
GENEPT=( "training.use_genept=true" )
NAIVE_RAG=( "training.use_genept=true" "training.rag=true" "training.retrieve_than_predict=true" )
PTRAG=( "training.use_genept=true" "training.rag=true" "training.retrieve_than_predict=true"
        "training.differentiable_rag=true" "training.gumbel_sparsity_loss=true" )

# train <output_dir> <run_name> <hydra args...>   (skips runs that already finished)
train() {
  local out="$1" name="$2"; shift 2
  if [ -f "$out/$name/checkpoints/final.ckpt" ]; then echo "[skip] $name (already trained)"; return; fi
  echo "[train] $name"
  python -m state.__main__ tx train "$@" "output_dir=$out" "name=$name" "use_wandb=$USE_WANDB"
}

# evaluate <run_dir> [extra predict flags...]
evaluate() {
  local run="$1"; shift
  echo "[eval] $(basename "$run") $*"
  python -m state.__main__ tx predict --output-dir "$run" --checkpoint last.ckpt --eval-genept-pert "$@"
}

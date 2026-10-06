#!/usr/bin/env bash
# Download the pretrained scGPT (whole-human) checkpoint into datasets/scgpt/
# (best_model.pt, args.json, vocab.json). Only needed for the scGPT experiments.
# Requires gdown:  pip install gdown
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
gdown --folder "https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y" -O scgpt
echo "Saved to $(pwd)/scgpt"

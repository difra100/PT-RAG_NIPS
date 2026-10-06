#!/usr/bin/env bash
# Download the Replogle-Nadig Perturb-seq data (about 22 GB) into datasets/Replogle_Nadig_dataset/
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p Replogle_Nadig_dataset
wget -c "https://huggingface.co/datasets/arcinstitute/Replogle-Nadig-Preprint/resolve/main/replogle.h5ad" \
     -O Replogle_Nadig_dataset/replogle.h5ad
echo "Saved to $(pwd)/Replogle_Nadig_dataset"

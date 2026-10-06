#!/usr/bin/env bash
# Download the GenePT gene embeddings into genept_emb/genept_emb/GenePT_emebdding_v2/
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p genept_emb
wget -c "https://zenodo.org/records/10833191/files/GenePT_emebdding_v2.zip?download=1" -O genept_emb/GenePT_emebdding_v2.zip
unzip -o -q genept_emb/GenePT_emebdding_v2.zip -d genept_emb
echo "Saved to $(pwd)/genept_emb"

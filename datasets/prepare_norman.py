"""
Prepare the Norman et al. (2019) Perturb-seq dataset (CRISPRa, K562).

Downloads the data with pertpy (`pip install pertpy`), keeps the single-gene perturbations
and the control cells, normalises and log-transforms the counts, aligns the genes to the
gene set of the Replogle-Nadig data, and writes:

    datasets/Norman_dataset/norman.h5ad
    datasets/Norman_dataset/single_gene_perts.json

The output follows the format of the Replogle-Nadig file:
    obs['gene']       perturbation name ('control' for control cells)
    obs['cell_line']  cell type ('k562')
    obs['gem_group']  batch
    obsm['X_hvg']     normalised log1p expression

Run from the repository root, after downloading the Replogle-Nadig data:
    python datasets/prepare_norman.py
"""

import os
import json
import numpy as np
import scipy.sparse as sp
import anndata
import scanpy as sc

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPLOGLE_H5AD = os.path.join(REPO_ROOT, "datasets", "Replogle_Nadig_dataset", "replogle.h5ad")
OUT_DIR = os.path.join(REPO_ROOT, "datasets", "Norman_dataset")
OUT_H5AD = os.path.join(OUT_DIR, "norman.h5ad")
PERT_LIST_OUT = os.path.join(OUT_DIR, "single_gene_perts.json")

os.makedirs(OUT_DIR, exist_ok=True)

# ── Step 1: load Replogle HVG gene set ───────────────────────────────────────
print("Loading Replogle HVG gene list...")
ref = anndata.read_h5ad(REPLOGLE_H5AD)
REPLOGLE_HVGS = ref.var_names.tolist()
print(f"  Replogle HVGs: {len(REPLOGLE_HVGS)} genes")
del ref

# ── Step 2: download Norman 2019 ─────────────────────────────────────────────
print("Downloading Norman 2019 via pertpy...")
import pertpy as pt
adata = pt.dt.norman_2019()
print(f"  Raw shape: {adata.shape}")
print(f"  obs columns: {list(adata.obs.columns)}")

# Detect the perturbation column (pertpy v1 uses 'perturbation_name')
for col in ['perturbation_name', 'perturbation', 'condition', 'gene']:
    if col in adata.obs.columns:
        PERT_COL = col
        break
else:
    raise ValueError(f"Cannot detect perturbation column in: {list(adata.obs.columns)}")
print(f"  Using perturbation column: '{PERT_COL}'")

# Detect batch column
for col in ['gemgroup', 'gem_group', 'batch']:
    if col in adata.obs.columns:
        BATCH_COL = col
        break
else:
    BATCH_COL = None
    print("  No batch column found — will use 'norman_k562'")

# Detect cell type column
for col in ['cell_line', 'cell_type', 'celltype']:
    if col in adata.obs.columns:
        CELLTYPE_COL = col
        break
else:
    CELLTYPE_COL = None
    print("  No cell type column found — will use 'k562'")

# ── Step 3: filter to single-gene perturbations + control ────────────────────
print("Filtering to single-gene perturbations...")
mask_single = ~adata.obs[PERT_COL].str.contains(r'\+', na=False)
adata = adata[mask_single].copy()
print(f"  After filter: {adata.shape}")
print(f"  Unique perturbations: {adata.obs[PERT_COL].nunique()}")

# Identify and standardize the control label
ctrl_vals = [v for v in adata.obs[PERT_COL].unique()
             if str(v).lower() in ('ctrl', 'control', 'non-targeting', 'nontargeting')]
if len(ctrl_vals) == 1:
    CTRL_LABEL = ctrl_vals[0]
elif not ctrl_vals:
    # Fallback: use most common perturbation as control
    CTRL_LABEL = adata.obs[PERT_COL].value_counts().index[0]
    print(f"  WARNING: guessing control as '{CTRL_LABEL}'")
else:
    CTRL_LABEL = ctrl_vals[0]
print(f"  Control label: '{CTRL_LABEL}'")

# Rename control to 'ctrl' for consistency
adata.obs['gene'] = adata.obs[PERT_COL].astype(str)

# ── Step 4: normalize + log1p (match Replogle's X_hvg range 0–8.35) ─────────
print("Normalizing...")
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# ── Step 5: align to Replogle's 6642 HVGs ───────────────────────────────────
print("Aligning to Replogle HVG gene set...")
var_names = adata.var_names.tolist()
var_set = set(var_names)

n_cells = adata.n_obs
n_hvg = len(REPLOGLE_HVGS)

# Build aligned matrix: fill with 0 for genes not in Norman
X_full = sp.issparse(adata.X)
if X_full:
    X = adata.X.toarray()
else:
    X = np.array(adata.X)

X_hvg = np.zeros((n_cells, n_hvg), dtype=np.float32)
matched = 0
for i, gene in enumerate(REPLOGLE_HVGS):
    if gene in var_set:
        idx = var_names.index(gene)
        X_hvg[:, i] = X[:, idx]
        matched += 1

coverage = matched / n_hvg * 100
print(f"  Matched {matched}/{n_hvg} HVGs ({coverage:.1f}% coverage)")
if coverage < 50:
    print("  WARNING: low gene coverage — results may be degraded")

# ── Step 6: build standardized obs columns ───────────────────────────────────
print("Building standardized obs columns...")
adata.obsm['X_hvg'] = X_hvg

if CELLTYPE_COL and CELLTYPE_COL in adata.obs.columns:
    adata.obs['cell_line'] = adata.obs[CELLTYPE_COL].astype(str).str.lower()
else:
    adata.obs['cell_line'] = 'k562'

if BATCH_COL:
    adata.obs['gem_group'] = adata.obs[BATCH_COL].astype(str)
else:
    adata.obs['gem_group'] = 'norman_k562'

# Standardize control label to 'ctrl' (the STATE codebase will use this in config)
# Keep as-is; the CTRL_LABEL will be passed to training config

# Print unique perturbations
single_perts = [p for p in adata.obs['gene'].unique() if p != CTRL_LABEL]
single_perts.sort()
print(f"\n  Single-gene perturbations ({len(single_perts)}):")
print(f"  {single_perts[:10]}...")
print(f"  Control: '{CTRL_LABEL}'")

# Save perturbation list for split creation
with open(PERT_LIST_OUT, 'w') as f:
    json.dump({'perturbations': single_perts, 'control': CTRL_LABEL}, f, indent=2)
print(f"\n  Perturbation list saved to: {PERT_LIST_OUT}")

# ── Step 7: slim down and save ───────────────────────────────────────────────
print("\nSaving processed h5ad...")
import pandas as _pd

# var must use Replogle's HVG gene names with the index name "gene_name_index"
# so that cell_load's get_gene_names() fallback (var/gene_name_index) succeeds
# and returns the gene names of the Replogle-Nadig gene set.
# X is set to X_hvg (6642-dim) so output_space=gene predicts in HVG space.
keep_cols = ['gene', 'cell_line', 'gem_group']
var_df = _pd.DataFrame(
    index=_pd.Index(REPLOGLE_HVGS, name="gene_name_index")
)
adata_out = anndata.AnnData(
    X=X_hvg,
    obs=adata.obs[keep_cols].copy(),
    var=var_df,
    obsm={'X_hvg': X_hvg},
)
adata_out.obs_names = adata.obs_names
adata_out.write_h5ad(OUT_H5AD)

print(f"\n✅ Norman dataset saved to: {OUT_H5AD}")
print(f"   Shape: {adata_out.shape}")
print(f"   X_hvg: {X_hvg.shape}")
print(f"   Perturbations: {len(single_perts)} single-gene + control='{CTRL_LABEL}'")
print(f"\nNext step:")
print(f"   python datasets/create_norman_splits.py")

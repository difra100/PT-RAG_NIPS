#!/usr/bin/env python
"""Compute the evaluation metrics from predictions saved by `tx predict`.

    python scripts/compute_metrics.py <eval_dir> [<out_dir>] [--control non-targeting]

<eval_dir> is the folder written by `tx predict` that contains adata_pred.h5ad and
adata_real.h5ad (e.g. <run>/eval_final.ckpt). Use it if the metric step of
`tx predict` stalls: it computes the same metrics, with the same settings, in a
separate process. Use --control control for Norman.
"""
import argparse, os
import numpy as np, pandas as pd
from gge import evaluate_lazy
from gge.metrics import (MeanPearsonCorrelation, MeanSpearmanCorrelation, RSquared, MSEDistance,
                         Wasserstein1Distance, EnergyDistance, Wasserstein2Distance)

ap = argparse.ArgumentParser()
ap.add_argument("eval_dir"); ap.add_argument("out_dir", nargs="?")
ap.add_argument("--control", default="non-targeting")
a = ap.parse_args()
out = a.out_dir or a.eval_dir


# Patch 1: vectorized DEG t-test (as in _predict.py)
def _fit_deg_fast(self, real_data):
    from scipy import stats as _stats
    real_mean = real_data.mean(axis=0)
    ctrl_mean = self.control_data.mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        log2fc = np.log2(real_mean + 1) - np.log2(ctrl_mean + 1)
    if real_data.shape[0] > 1 and self.control_data.shape[0] > 1:
        _, pvalues = _stats.ttest_ind(real_data, self.control_data, axis=0, equal_var=False)
        pvalues = np.where(np.isnan(pvalues), 1.0, pvalues)
    else:
        pvalues = np.ones(real_data.shape[1])
    is_deg = (np.abs(log2fc) >= self.config.deg_lfc) & (pvalues <= self.config.deg_pval)
    if self.config.n_top_degs is not None and is_deg.sum() > self.config.n_top_degs:
        deg_indices = np.where(is_deg)[0]
        top_idx = deg_indices[np.argsort(np.abs(log2fc[deg_indices]))[::-1][: self.config.n_top_degs]]
        is_deg = np.zeros(real_data.shape[1], dtype=bool)
        is_deg[top_idx] = True
    if is_deg.sum() == 0:
        is_deg = np.ones(real_data.shape[1], dtype=bool)
    self._deg_mask = is_deg
    self._deg_genes = [g for g, m in zip(self.gene_names, is_deg) if m]
    self._is_fitted = True

from gge.spaces import SpaceTransformer
SpaceTransformer._fit_deg = _fit_deg_fast

# Patch 2: re-fit the DEG mask per condition
from gge.metrics.base_metric import BaseMetric
_orig_compute = BaseMetric.compute
def _per_condition_compute(self, real, generated, gene_names=None, aggregate_method="mean",
                           condition=None, split=None, control_data=None, _orig=_orig_compute):
    if self.space == "deg":
        self._transformer = None
    return _orig(self, real, generated, gene_names, aggregate_method, condition, split, control_data)
BaseMetric.compute = _per_condition_compute

# Patch 3: tolerate per-gene arrays of DEG length when saving
from gge.results import EvaluationResult
_orig_pg = EvaluationResult.to_per_gene_dataframe
def _safe_pg(self, metric_name, _orig=_orig_pg):
    try:
        return _orig(self, metric_name)
    except Exception:
        return pd.DataFrame()
EvaluationResult.to_per_gene_dataframe = _safe_pg

metrics = [
    MeanPearsonCorrelation(space="deg", deg_lfc=0.05, deg_pval=0.1),
    MeanSpearmanCorrelation(space="deg", deg_lfc=0.05, deg_pval=0.1),
    RSquared(space="deg", deg_lfc=0.05, deg_pval=0.1),
    MSEDistance(space="pca", n_components=50),
    Wasserstein2Distance(space="pca", n_components=50),
    Wasserstein1Distance(space="pca", n_components=50),
    EnergyDistance(space="pca", n_components=50),
]
os.makedirs(out, exist_ok=True)
evaluate_lazy(real_path=os.path.join(a.eval_dir, "adata_real.h5ad"),
              generated_path=os.path.join(a.eval_dir, "adata_pred.h5ad"),
              condition_columns=["gene", "cell_line"], output_dir=out,
              metrics=metrics, verbose=True, control_key=a.control)
print("done:", os.path.join(out, "results.csv"))

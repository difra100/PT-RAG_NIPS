"""
Create 3-fold cross-perturbation TOML split files for the Norman dataset.

Split: 30% train / 10% val / 60% test (matching Replogle cross-pert protocol).
Output: datasets/norman_pertsplit_30_10_60_fold{0,1,2}.toml

Run after prepare_norman.py:
    python datasets/create_norman_splits.py
"""

import json
import os
import random

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERT_LIST = os.path.join(REPO_ROOT, "datasets", "Norman_dataset", "single_gene_perts.json")
OUT_DIR = os.path.join(REPO_ROOT, "datasets")
NORMAN_DATA_DIR = os.path.join(REPO_ROOT, "datasets", "Norman_dataset")

TRAIN_FRAC = 0.30
VAL_FRAC = 0.10
# TEST_FRAC = 0.60 (remainder)

N_FOLDS = 3
SEEDS = [42, 123, 456]

with open(PERT_LIST) as f:
    info = json.load(f)

perturbations = sorted(info['perturbations'])
control = info['control']
n = len(perturbations)
n_train = max(1, round(n * TRAIN_FRAC))
n_val = max(1, round(n * VAL_FRAC))
n_test = n - n_train - n_val

print(f"Total single-gene perturbations: {n}")
print(f"Split: {n_train} train / {n_val} val / {n_test} test")
print(f"Control label: '{control}'")

for fold, seed in enumerate(SEEDS):
    rng = random.Random(seed)
    perts = perturbations[:]
    rng.shuffle(perts)

    train_perts = sorted(perts[:n_train])
    val_perts = sorted(perts[n_train:n_train + n_val])
    test_perts = sorted(perts[n_train + n_val:])

    # Format TOML list
    def fmt_list(items, indent=16):
        pad = ' ' * indent
        inner = f',\n{pad}'.join(f"'{x}'" for x in items)
        return f"[\n{pad}{inner}\n{' ' * (indent - 4)}]"

    toml_content = f'''# Norman 2019 cross-perturbation split — fold {fold}
# {n_train} train / {n_val} val / {n_test} test ({n} total single-gene perturbations)
# Control: '{control}'
# Seed: {seed}

[datasets]
norman = "datasets/Norman_dataset"

[training]
norman = "train"

[zeroshot]

[fewshot]

[fewshot."norman.k562"]
val = {fmt_list(val_perts)}
test = {fmt_list(test_perts)}
'''

    out_path = os.path.join(OUT_DIR, f"norman_pertsplit_30_10_60_fold{fold}.toml")
    with open(out_path, 'w') as f:
        f.write(toml_content)

    print(f"\nFold {fold} → {out_path}")
    print(f"  Train ({len(train_perts)}): {train_perts[:3]}...")
    print(f"  Val   ({len(val_perts)}):  {val_perts[:3]}...")
    print(f"  Test  ({len(test_perts)}): {test_perts[:3]}...")

print(f"\n✅ Created {N_FOLDS} TOML split files.")
print("Next step: bash run_norman_experiments.sh")

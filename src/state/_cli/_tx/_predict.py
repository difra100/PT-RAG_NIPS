import argparse as ap

from sympy import evaluate
from gge import evaluate_lazy, evaluate
from gge.metrics import (
    MeanPearsonCorrelation, MeanSpearmanCorrelation, RSquared,
    MSEDistance,
    Wasserstein1Distance, EnergyDistance, Wasserstein2Distance
)
import pickle
from pathlib import Path
import logging


def load_gene_mapping(checkpoint_dir):
    """Load and invert the gene name to index mapping to get index to name mapping."""
    logger = logging.getLogger(__name__)
    mapping_file = Path(checkpoint_dir) / "train_name_to_idx.pkl"
    if mapping_file.exists():
        with open(mapping_file, 'rb') as f:
            name_to_idx = pickle.load(f)
        # Invert to get idx -> name mapping
        idx_to_name = {idx: str(name) for name, idx in name_to_idx.items()}
        logger.info(f"Loaded gene mapping with {len(idx_to_name)} genes")
        return idx_to_name
    else:
        logger.warning(f"Gene mapping file not found at {mapping_file}")
        return None



class MetricType:
    def __init__(self, name: str):
        self.name = name

def add_arguments_predict(parser: ap.ArgumentParser):
    """
    CLI for evaluation using cell-eval metrics.
    """

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to the output_dir containing the config.yaml file that was saved during training.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="last.ckpt",
        help="Checkpoint filename. Default is 'last.ckpt'. Relative to the output directory.",
    )

    parser.add_argument(
        "--test-time-finetune",
        type=int,
        default=0,
        help="If >0, run test-time fine-tuning for the specified number of epochs on only control cells.",
    )

    parser.add_argument(
        "--profile",
        type=str,
        default="full",
        choices=["full", "minimal", "de", "anndata"],
        help="run all metrics, minimal, only de metrics, or only output adatas",
    )

    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="If set, only run prediction without evaluation metrics.",
    )

    parser.add_argument(
        "--shared-only",
        action="store_true",
        help=("If set, restrict predictions/evaluation to perturbations shared between train and test (train ∩ test)."),
    )

    parser.add_argument(
        "--eval-train-data",
        action="store_true",
        help="If set, evaluate the model on the training data rather than on the test data.",
    )

    parser.add_argument(
        "--eval-genept-pert",
        action="store_true",
        help="If set, filter test set to only include perturbations present in GenePT (for fair comparison).",
    )

    parser.add_argument(
        "--count-flops",
        action="store_true",
        help="If set, count and save FLOPs performed during prediction to a JSON file.",
    )

    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="If set, limit processing to first N batches (useful for testing).",
    )

    parser.add_argument(
        "--retrieval-pool",
        choices=["full", "trainval", "train"],
        default="full",
        help=(
            "Retrieval pool used at test time when the model was trained with split-aware retrieval: "
            "'full' (all perturbations, default), 'trainval' (train and validation perturbations) "
            "or 'train' (training perturbations only)."
        ),
    )

    parser.add_argument(
        "--hard-negative-fraction",
        type=float,
        default=0.0,
        help=(
            "Robustness control: fraction of the K retrieved candidates replaced with the "
            "perturbations least similar to the query (0.0 = disabled)."
        ),
    )


def run_tx_predict(args: ap.ArgumentParser):
    import logging
    import os
    import sys

    import anndata
    import lightning.pytorch as pl
    import numpy as np
    import pandas as pd
    import torch
    import yaml

    # Cell-eval for metrics computation
    from cell_eval import MetricsEvaluator
    from cell_eval.utils import split_anndata_on_celltype
    from cell_load.data_modules import PerturbationDataModule
    from tqdm import tqdm

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    torch.multiprocessing.set_sharing_strategy("file_system")

    def get_n_params(model):
        pp=0
        for p in list(model.parameters()):
            nn=1
            for s in list(p.size()):
                nn = nn*s
            pp += nn
        return pp

    def run_test_time_finetune(model, dataloader, ft_epochs, control_pert, device):
        """
        Perform test-time fine-tuning on only control cells.
        """
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

        logger.info(f"Starting test-time fine-tuning for {ft_epochs} epoch(s) on control cells only.")
        for epoch in range(ft_epochs):
            epoch_losses = []
            pbar = tqdm(dataloader, desc=f"Finetune epoch {epoch + 1}/{ft_epochs}", leave=True)
            for batch in pbar:
                # Check if this batch contains control cells
                first_pert = (
                    batch["pert_name"][0] if isinstance(batch["pert_name"], list) else batch["pert_name"][0].item()
                )
                if first_pert != control_pert:
                    continue

                # Move batch data to device
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

                optimizer.zero_grad()
                loss = model.training_step(batch, batch_idx=0, padded=False)
                if loss is None:
                    continue
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            mean_loss = np.mean(epoch_losses) if epoch_losses else float("nan")
            logger.info(f"Finetune epoch {epoch + 1}/{ft_epochs}, mean loss: {mean_loss}")
        model.eval()

    def load_config(cfg_path: str) -> dict:
        """Load config from the YAML file that was dumped during training."""
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Could not find config file: {cfg_path}")
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        return cfg

    # 1. Load the config
    config_path = os.path.join(args.output_dir, "config.yaml")
    cfg = load_config(config_path)
    logger.info(f"Loaded config from {config_path}")

    # 2. Find run output directory & load data module
    run_output_dir = os.path.join(cfg["output_dir"], cfg["name"])
    data_module_path = os.path.join(run_output_dir, "data_module.torch")
    if not os.path.exists(data_module_path):
        raise FileNotFoundError(f"Could not find data module at {data_module_path}?")
    if cfg["model"]["name"].lower().startswith("scgpt"):
        import json
        from pathlib import Path as _Path
        from ...tx.data.scgpt_data_module import scGPTPerturbationDataModule

        pretrained_path = _Path(cfg["model"]["kwargs"]["pretrained_path"])
        vocab = json.load(open(pretrained_path / "vocab.json"))
        for s in cfg["model"]["kwargs"].get("special_tokens", []):
            if s not in vocab:
                vocab[s] = len(vocab)

        save_dict = torch.load(data_module_path, map_location="cpu")
        extra_kwargs = {
            "map_controls": save_dict.pop("map_controls", True),
            "cache_perturbation_control_pairs": save_dict.pop("cache_perturbation_control_pairs", False),
            "perturbation_features_file": save_dict.pop("perturbation_features_file", None),
            "int_counts": save_dict.pop("int_counts", False),
            "normalize_counts": save_dict.pop("normalize_counts", False),
            "store_raw_basal": save_dict.pop("store_raw_basal", False),
            "barcode": save_dict.pop("barcode", True),
        }
        perturbation_type = cfg["model"]["kwargs"].get("perturbation_type", "genetic")
        hvg_names_uns_key = (
            "hvg_names" if cfg["data"]["kwargs"].get("train_task", "replogle") != "replogle" else None
        )
        data_module = scGPTPerturbationDataModule(
            vocab=vocab,
            perturbation_type=perturbation_type,
            hvg_names_uns_key=hvg_names_uns_key,
            **save_dict,
            **extra_kwargs,
        )
        logger.info("Loaded scGPTPerturbationDataModule from %s", data_module_path)
    else:
        data_module = PerturbationDataModule.load_state(data_module_path)
    
    # Reload pert_onehot_map BEFORE setup() to preserve GenePT embeddings
    # This ensures datasets created during setup() use the correct embeddings
    pert_onehot_map_path = os.path.join(run_output_dir, "pert_onehot_map.pt")
    if os.path.exists(pert_onehot_map_path):
        data_module.pert_onehot_map = torch.load(pert_onehot_map_path, weights_only=False)
        logger.info(f"Reloaded pert_onehot_map from {pert_onehot_map_path} (dim={list(data_module.pert_onehot_map.values())[0].shape[0]})")
    
    data_module.setup(stage="test")
    
    # Filter test datasets to only include perturbations present in GenePT
    # This can be enabled for non-GenePT models too, for fair comparison
    if args.eval_genept_pert or cfg['training'].get('use_genept', False):
        logger.info("Filtering test datasets to only include GenePT perturbations...")
        
        # Load GenePT dictionary to get valid perturbations
        import pickle
        genept_path = 'genept_emb/genept_emb/GenePT_emebdding_v2/GenePT_gene_embedding_ada_text.pickle'
        if os.path.exists(genept_path):
            with open(genept_path, 'rb') as f:
                genept_data = pickle.load(f)
            valid_perts = set(genept_data.keys())
            logger.info(f"Loaded GenePT vocabulary: {len(valid_perts)} perturbations")
        else:
            logger.warning(f"GenePT file not found at {genept_path}, skipping filtering")
            valid_perts = None
        
        if valid_perts is not None:
            def filter_subset_by_perts(subset, valid_perts):
                """Filter a Subset to keep only cells with valid perturbations or controls"""
                underlying_ds = subset.dataset
                cache = underlying_ds.metadata_cache
                control_pert_code = cache.control_pert_code
                
                original_indices = subset.indices
                valid_mask = []
                for idx in original_indices:
                    pert_code = cache.pert_codes[idx]
                    if pert_code == control_pert_code:
                        valid_mask.append(True)
                    else:
                        pert_name = cache.pert_categories[pert_code]
                        # GenePT keys are strings, so convert for comparison
                        valid_mask.append(str(pert_name) in valid_perts)
                
                valid_mask = np.array(valid_mask)
                filtered_indices = original_indices[valid_mask]
                subset.indices = filtered_indices
                
                return len(original_indices), len(filtered_indices)
            
            total_removed = 0
            total_kept = 0
            
            if data_module.test_datasets:
                for i, subset in enumerate(data_module.test_datasets):
                    original_count, filtered_count = filter_subset_by_perts(subset, valid_perts)
                    removed = original_count - filtered_count
                    total_removed += removed
                    total_kept += filtered_count
                    if removed > 0:
                        logger.info(f"  test[{i}]: kept {filtered_count}/{original_count} cells ({removed} removed)")
            
            logger.info(f"✅ Test filtering complete: {total_kept} cells kept, {total_removed} removed")
    
    logger.info("Loaded data module from %s", data_module_path)

    # Seed everything
    pl.seed_everything(cfg["training"]["train_seed"])

    # 3. Load the trained model
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, args.checkpoint)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Could not find checkpoint at {checkpoint_path}.\nSpecify a correct checkpoint filename with --checkpoint."
        )
    logger.info("Loading model from %s", checkpoint_path)

    # Determine model class and load
    model_class_name = cfg["model"]["name"]
    model_kwargs = cfg["model"]["kwargs"]

    # Import the correct model class
    if model_class_name.lower() == "embedsum":
        from ...tx.models.embed_sum import EmbedSumPerturbationModel

        ModelClass = EmbedSumPerturbationModel
    elif model_class_name.lower() == "old_neuralot":
        from ...tx.models.old_neural_ot import OldNeuralOTPerturbationModel

        ModelClass = OldNeuralOTPerturbationModel
    elif model_class_name.lower() in ["neuralot", "pertsets", "state"]:
        from ...tx.models.state_transition import StateTransitionPerturbationModel

        ModelClass = StateTransitionPerturbationModel

    elif model_class_name.lower() in ["globalsimplesum", "perturb_mean"]:
        from ...tx.models.perturb_mean import PerturbMeanPerturbationModel

        ModelClass = PerturbMeanPerturbationModel
    elif model_class_name.lower() in ["celltypemean", "context_mean"]:
        from ...tx.models.context_mean import ContextMeanPerturbationModel

        ModelClass = ContextMeanPerturbationModel
    elif model_class_name.lower() == "decoder_only":
        from ...tx.models.decoder_only import DecoderOnlyPerturbationModel

        ModelClass = DecoderOnlyPerturbationModel
    elif model_class_name.lower() == "scgpt-genetic":
        from ...tx.models.scgpt.lightning_model import scGPTForPerturbation

        ModelClass = scGPTForPerturbation
    elif model_class_name.lower() == "scgpt-genetic-rag":
        from ...tx.models.scgpt.rag_model import scGPTRAGForPerturbation

        ModelClass = scGPTRAGForPerturbation
    else:
        raise ValueError(f"Unknown model class: {model_class_name}")

    var_dims_path = os.path.join(run_output_dir, "var_dims.pkl")
    if os.path.exists(var_dims_path):
        with open(var_dims_path, "rb") as f:
            var_dims = pickle.load(f)
        logger.info(f"Loaded var_dims from {var_dims_path}")
    else:
        var_dims = data_module.get_var_dims()
    model_init_kwargs = {
        "input_dim": var_dims["input_dim"],
        "hidden_dim": model_kwargs["hidden_dim"],
        "gene_dim": var_dims["gene_dim"],
        "hvg_dim": var_dims["hvg_dim"],
        "output_dim": var_dims["output_dim"],
        "pert_dim": var_dims["pert_dim"],
        **model_kwargs,
    }

    if cfg['training'].get('use_genept', 'False') and cfg['training'].get('rag', 'False'):
        model_init_kwargs['pert_dim'] = 1536
        
    model = ModelClass.load_from_checkpoint(checkpoint_path, strict=False, **model_init_kwargs)
    
    
    if cfg["training"].get("rag", False):
        import faiss
        import pickle

        split_aware_rag = cfg["training"].get("split_aware_rag", False)

        # Load name-to-index mapping (always needed for RAG)
        train_mapping_path = os.path.join(run_output_dir, "train_name_to_idx.pkl")
        if os.path.exists(train_mapping_path):
            with open(train_mapping_path, "rb") as f:
                train_name_to_idx = pickle.load(f)
            logger.info(f"Loaded train_name_to_idx with {len(train_name_to_idx)} perturbations")
        else:
            train_name_to_idx = None
            logger.warning("train_name_to_idx.pkl not found, RAG may not work correctly")

        # Load train_perts (GenePT embeddings) if using retrieve_than_predict mode
        if cfg['training'].get('retrieve_than_predict', False) and os.path.exists(pert_onehot_map_path):
            pert_map = torch.load(pert_onehot_map_path, weights_only=False)

            retrieval_pool = getattr(args, "retrieval_pool", "full")
            if split_aware_rag and retrieval_pool == "train" and train_name_to_idx is not None:
                # Restrict the test-time pool to the training perturbations
                pool_list = [None] * len(train_name_to_idx)
                for pert_name, idx in train_name_to_idx.items():
                    if pert_name in pert_map:
                        pool_list[idx] = pert_map[pert_name]
                pool = torch.stack([t for t in pool_list if t is not None])
                setattr(model, "train_perts", pool)
                setattr(model, "train_name_to_idx", train_name_to_idx)
                logger.info(f"Retrieval pool: training perturbations only ({len(train_name_to_idx)})")
            elif split_aware_rag and retrieval_pool == "trainval" and train_name_to_idx is not None:
                # Restrict the test-time pool to the training and validation perturbations
                import tomllib

                toml_path = cfg.get("data", {}).get("kwargs", {}).get("toml_config_path", None)
                if not toml_path or not os.path.exists(toml_path):
                    raise FileNotFoundError(f"Split file not found, cannot build the train+val pool: {toml_path}")
                with open(toml_path, "rb") as f:
                    split_cfg = tomllib.load(f)
                val_perts = set()
                for splits in split_cfg.get("fewshot", {}).values():
                    val_perts.update(splits.get("val", []))
                names = set(train_name_to_idx.keys()) | (val_perts & set(pert_map.keys()))
                pool_name_to_idx, pool_list = {}, []
                for pn in sorted(names):
                    if pn in pert_map and pert_map[pn].norm().item() > 0.0:
                        pool_name_to_idx[pn] = len(pool_list)
                        pool_list.append(pert_map[pn])
                setattr(model, "train_perts", torch.stack(pool_list))
                setattr(model, "train_name_to_idx", pool_name_to_idx)
                logger.info(f"Retrieval pool: training and validation perturbations ({len(pool_name_to_idx)})")
            elif split_aware_rag:
                # At prediction time use ALL perturbations (train+val+test) as the retrieval pool
                all_name_to_idx = {}
                all_perts_list = []
                for pn in sorted(pert_map.keys()):
                    all_name_to_idx[pn] = len(all_perts_list)
                    all_perts_list.append(pert_map[pn])
                all_perts = torch.stack(all_perts_list)
                setattr(model, "train_perts", all_perts)
                setattr(model, "train_name_to_idx", all_name_to_idx)
                logger.info(f"✅ Split-aware RAG prediction: using full pool of {len(all_name_to_idx)} perturbations")
            else:
                # Original behavior: rebuild pool in the same order as saved train_name_to_idx
                if train_name_to_idx is not None:
                    train_perts_list = [None] * len(train_name_to_idx)
                    for pert_name, idx in train_name_to_idx.items():
                        if pert_name in pert_map:
                            train_perts_list[idx] = pert_map[pert_name]
                    train_perts = torch.stack([t for t in train_perts_list if t is not None])
                    setattr(model, "train_perts", train_perts)
                    setattr(model, "train_name_to_idx", train_name_to_idx)
                    logger.info(f"✅ Loaded train_perts with shape {train_perts.shape}")
                else:
                    logger.warning("Cannot load train_perts: train_name_to_idx.pkl not found")
        elif train_name_to_idx is not None:
            setattr(model, "train_name_to_idx", train_name_to_idx)

        # Load FAISS index (only needed for non-retrieve_than_predict mode)
        train_index_path = os.path.join(run_output_dir, "train_faiss.index")
        if not cfg['training'].get('retrieve_than_predict', False) and os.path.exists(train_index_path):
            train_index = faiss.read_index(train_index_path)
            setattr(model, "train_index_final", train_index)
            logger.info("✅ Loaded FAISS index for RAG retrieval")
        elif not cfg['training'].get('retrieve_than_predict', False):
            logger.warning("FAISS index not found, RAG retrieval may not work")


    hard_negative_fraction = getattr(args, "hard_negative_fraction", 0.0)
    if hard_negative_fraction > 0.0:
        setattr(model, "hard_negative_fraction", hard_negative_fraction)
        logger.info(f"Replacing {hard_negative_fraction:.0%} of the retrieved candidates with hard negatives")

    model.eval()
    logger.info("Model loaded successfully.")

    # 4. Test-time fine-tuning if requested
    data_module.batch_size = 1
    if args.test_time_finetune > 0:
        control_pert = data_module.get_control_pert()
        if args.eval_train_data:
            test_loader = data_module.train_dataloader(test=True)
        else:
            test_loader = data_module.test_dataloader()

        run_test_time_finetune(
            model, test_loader, args.test_time_finetune, control_pert, device=next(model.parameters()).device
        )
        logger.info("Test-time fine-tuning complete.")

    # Pre-compute output paths so we can check for cached results.
    # Non-default evaluation options get their own results folder.
    eval_suffix = ""
    if getattr(args, "retrieval_pool", "full") != "full":
        eval_suffix += f"_pool-{args.retrieval_pool}"
    if getattr(args, "hard_negative_fraction", 0.0) > 0.0:
        eval_suffix += f"_hardneg{args.hard_negative_fraction}"
    results_dir = os.path.join(args.output_dir, "eval_" + os.path.basename(args.checkpoint) + eval_suffix)
    os.makedirs(results_dir, exist_ok=True)
    adata_pred_path = os.path.join(results_dir, "adata_pred.h5ad")
    adata_real_path = os.path.join(results_dir, "adata_real.h5ad")
    _skip_inference = os.path.exists(adata_pred_path) and os.path.exists(adata_real_path)

    if _skip_inference:
        logger.info(
            "Found cached predictions at %s – loading and skipping inference.",
            results_dir,
        )
        adata_pred = anndata.read_h5ad(adata_pred_path)
        adata_real = anndata.read_h5ad(adata_real_path)

    flop_stats = None
    all_retrieval_info = []

    if not _skip_inference:
        # 5. Run inference on test set
        data_module.setup(stage="test")
        if args.eval_train_data:
            test_loader = data_module.train_dataloader(test=True)
        else:
            test_loader = data_module.test_dataloader()

        if test_loader is None:
            logger.warning("No test dataloader found. Exiting.")
            sys.exit(0)

        num_cells = test_loader.batch_sampler.tot_num
        output_dim = var_dims["output_dim"]
        gene_dim = var_dims["gene_dim"]
        hvg_dim = var_dims["hvg_dim"]

        if args.max_batches is not None:
            logger.info(f"TESTING MODE: Processing only first {args.max_batches} batches")
            # Estimate number of cells for testing mode
            batch_size = getattr(test_loader, 'batch_size', None)
            if batch_size is None:
                # For custom samplers, use a reasonable default
                batch_size = 64  # Default batch size based on cell_sentence_len
            estimated_cells = args.max_batches * batch_size
            num_cells = min(num_cells, estimated_cells)

        logger.info("Generating predictions on test set using manual loop...")
        device = next(model.parameters()).device

        final_preds = np.empty((num_cells, output_dim), dtype=np.float32)
        final_reals = np.empty((num_cells, output_dim), dtype=np.float32)

        store_raw_expression = (
            data_module.embed_key is not None
            and data_module.embed_key != "X_hvg"
            and cfg["data"]["kwargs"]["output_space"] == "gene"
        ) or (data_module.embed_key is not None and cfg["data"]["kwargs"]["output_space"] == "all")

        final_X_hvg = None
        final_pert_cell_counts_preds = None
        if store_raw_expression:
            # Preallocate matrices of shape (num_cells, gene_dim) for decoded predictions.
            if cfg["data"]["kwargs"]["output_space"] == "gene":
                final_X_hvg = np.empty((num_cells, hvg_dim), dtype=np.float32)
                final_pert_cell_counts_preds = np.empty((num_cells, hvg_dim), dtype=np.float32)
            if cfg["data"]["kwargs"]["output_space"] == "all":
                final_X_hvg = np.empty((num_cells, gene_dim), dtype=np.float32)
                final_pert_cell_counts_preds = np.empty((num_cells, gene_dim), dtype=np.float32)

        current_idx = 0

        # Initialize aggregation variables directly
        all_pert_names = []
        all_celltypes = []
        all_gem_groups = []
        all_pert_barcodes = []
        all_ctrl_barcodes = []
    
        # Initialize retrieval info collection (for RAG models)
        all_retrieval_info = []

        # Check if we should collect retrieval info
        collect_retrieval_info = (
            cfg["training"].get("rag", False) and 
            cfg["training"].get("differentiable_rag", False)
        )
    
        # Initialize FLOP counting if requested
        flop_stats = None
        if args.count_flops:
            from torch.profiler import profile, ProfilerActivity
            flop_stats = {
                "total_flops": 0,
                "total_batches": 0,
                "total_cells": 0,
                "per_batch_flops": [],
                "device": str(device),
                "model_class": model_class_name,
                "checkpoint": args.checkpoint,
                "model_parameters": get_n_params(model),
            }
            logger.info("FLOP counting enabled")
            logger.info(f"Model has {flop_stats['model_parameters']:,} parameters")
    
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Predicting", unit="batch")):
                # Check if we've reached the maximum number of batches for testing
                if args.max_batches is not None and batch_idx >= args.max_batches:
                    logger.info(f"Stopping at batch {batch_idx} (max_batches={args.max_batches})")
                    break
                
                # Move each tensor in the batch to the model's device
                batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

                # Get predictions (with optional FLOP counting)
                if args.count_flops:
                    with profile(
                        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if device.type == "cuda" else [ProfilerActivity.CPU],
                        with_flops=True,
                        record_shapes=True,
                    ) as prof:
                        batch_preds = model.predict_step(batch, batch_idx, padded=False, collect_retrieval_info=collect_retrieval_info)
                
                    # Extract FLOP count from profiler
                    events = prof.key_averages()
                    batch_flops = sum([evt.flops for evt in events if evt.flops is not None and evt.flops > 0])
                    batch_size_actual = batch_preds["preds"].shape[0]
                
                    flop_stats["total_flops"] += batch_flops
                    flop_stats["total_batches"] += 1
                    flop_stats["total_cells"] += batch_size_actual
                    flop_stats["per_batch_flops"].append({
                        "batch_idx": batch_idx,
                        "flops": batch_flops,
                        "cells": batch_size_actual,
                        "flops_per_cell": batch_flops / batch_size_actual if batch_size_actual > 0 else 0
                    })
                else:
                    batch_preds = model.predict_step(batch, batch_idx, padded=False, collect_retrieval_info=collect_retrieval_info)
            
                # Collect retrieval info if available
                if "retrieval_info" in batch_preds and batch_preds["retrieval_info"]:
                    all_retrieval_info.append(batch_preds["retrieval_info"])

                # Extract metadata and data directly from batch_preds
                # Handle pert_name
                if isinstance(batch_preds["pert_name"], list):
                    all_pert_names.extend(batch_preds["pert_name"])
                else:
                    all_pert_names.append(batch_preds["pert_name"])

                if "pert_cell_barcode" in batch_preds:
                    if isinstance(batch_preds["pert_cell_barcode"], list):
                        all_pert_barcodes.extend(batch_preds["pert_cell_barcode"])
                        all_ctrl_barcodes.extend(batch_preds["ctrl_cell_barcode"])
                    else:
                        all_pert_barcodes.append(batch_preds["pert_cell_barcode"])
                        all_ctrl_barcodes.append(batch_preds["ctrl_cell_barcode"])

                # Handle celltype_name
                if isinstance(batch_preds["celltype_name"], list):
                    all_celltypes.extend(batch_preds["celltype_name"])
                else:
                    all_celltypes.append(batch_preds["celltype_name"])

                # Handle gem_group
                if isinstance(batch_preds["batch"], list):
                    all_gem_groups.extend([str(x) for x in batch_preds["batch"]])
                elif isinstance(batch_preds["batch"], torch.Tensor):
                    all_gem_groups.extend([str(x) for x in batch_preds["batch"].cpu().numpy()])
                else:
                    all_gem_groups.append(str(batch_preds["batch"]))

                batch_pred_np = batch_preds["preds"].cpu().numpy().astype(np.float32)
                batch_real_np = batch_preds["pert_cell_emb"].cpu().numpy().astype(np.float32)
                batch_size = batch_pred_np.shape[0]
                # Re-allocate if the model's actual output dim differs from var_dims (e.g. scGPT strips pad tokens)
                if current_idx == 0 and batch_pred_np.shape[1] != output_dim:
                    output_dim = batch_pred_np.shape[1]
                    final_preds = np.empty((num_cells, output_dim), dtype=np.float32)
                    final_reals = np.empty((num_cells, output_dim), dtype=np.float32)
                final_preds[current_idx : current_idx + batch_size, :] = batch_pred_np
                final_reals[current_idx : current_idx + batch_size, :] = batch_real_np
                current_idx += batch_size

                # Handle X_hvg for HVG space ground truth
                if final_X_hvg is not None:
                    batch_real_gene_np = batch_preds["pert_cell_counts"].cpu().numpy().astype(np.float32)
                    final_X_hvg[current_idx - batch_size : current_idx, :] = batch_real_gene_np

                # Handle decoded gene predictions if available
                if final_pert_cell_counts_preds is not None:
                    batch_gene_pred_np = batch_preds["pert_cell_counts_preds"].cpu().numpy().astype(np.float32)
                    final_pert_cell_counts_preds[current_idx - batch_size : current_idx, :] = batch_gene_pred_np

        logger.info("Creating anndatas from predictions from manual loop...")

        # Build pandas DataFrame for obs and var
        df_dict = {
            data_module.pert_col: all_pert_names,
            data_module.cell_type_key: all_celltypes,
            data_module.batch_col: all_gem_groups,
        }

        if len(all_pert_barcodes) > 0:
            df_dict["pert_cell_barcode"] = all_pert_barcodes
            df_dict["ctrl_cell_barcode"] = all_ctrl_barcodes

        obs = pd.DataFrame(df_dict)
    
        # Adjust array sizes for the actual number of processed cells
        actual_num_cells = current_idx
        final_preds = final_preds[:actual_num_cells]
        final_reals = final_reals[:actual_num_cells]
        if final_X_hvg is not None:
            final_X_hvg = final_X_hvg[:actual_num_cells]
            final_pert_cell_counts_preds = final_pert_cell_counts_preds[:actual_num_cells]
    
        if args.max_batches is not None:
            logger.warning(f"⚠️  PARTIAL RESULTS: Only processed {actual_num_cells} cells from {args.max_batches} batches")

        gene_names = var_dims["gene_names"]
        var = pd.DataFrame({"gene_names": gene_names})

        if final_X_hvg is not None:
            if len(gene_names) != final_pert_cell_counts_preds.shape[1]:
                gene_names = np.load(
                    "/large_storage/ctc/userspace/aadduri/datasets/tahoe_19k_to_2k_names.npy", allow_pickle=True
                )
                var = pd.DataFrame({"gene_names": gene_names})

            # Create adata for predictions - using the decoded gene expression values
            adata_pred = anndata.AnnData(X=final_pert_cell_counts_preds, obs=obs, var=var)
            # Create adata for real - using the true gene expression values
            adata_real = anndata.AnnData(X=final_X_hvg, obs=obs, var=var)

            # add the embedding predictions
            adata_pred.obsm[data_module.embed_key] = final_preds
            adata_real.obsm[data_module.embed_key] = final_reals
            logger.info(f"Added predicted embeddings to adata.obsm['{data_module.embed_key}']")
        else:
            # if len(gene_names) != final_preds.shape[1]:
            #     gene_names = np.load(
            #         "/large_storage/ctc/userspace/aadduri/datasets/tahoe_19k_to_2k_names.npy", allow_pickle=True
            #     )
            #     var = pd.DataFrame({"gene_names": gene_names})

            # Create adata for predictions - model was trained on gene expression space already
            # adata_pred = anndata.AnnData(X=final_preds, obs=obs, var=var)
            adata_pred = anndata.AnnData(X=final_preds, obs=obs)
            # Create adata for real - using the true gene expression values
            # adata_real = anndata.AnnData(X=final_reals, obs=obs, var=var)
            adata_real = anndata.AnnData(X=final_reals, obs=obs)

        # Optionally filter to perturbations seen in at least one training context
        if args.shared_only:
            try:
                shared_perts = data_module.get_shared_perturbations()
                if len(shared_perts) == 0:
                    logger.warning("No shared perturbations between train and test; skipping filtering.")
                else:
                    logger.info(
                        "Filtering to %d shared perturbations present in train ∩ test.",
                        len(shared_perts),
                    )
                    mask = adata_pred.obs[data_module.pert_col].isin(shared_perts)
                    before_n = adata_pred.n_obs
                    adata_pred = adata_pred[mask].copy()
                    adata_real = adata_real[mask].copy()
                    logger.info(
                        "Filtered cells: %d -> %d (kept only seen perturbations)",
                        before_n,
                        adata_pred.n_obs,
                    )
            except Exception as e:
                logger.warning(
                    "Failed to filter by shared perturbations (%s). Proceeding without filter.",
                    str(e),
                )

        # Save the AnnData objects
        adata_pred.write_h5ad(adata_pred_path)
        adata_real.write_h5ad(adata_real_path)

        logger.info(f"Saved adata_pred to {adata_pred_path}")
        logger.info(f"Saved adata_real to {adata_real_path}")
    
    # Save FLOP statistics if counting was enabled
    if flop_stats is not None:
        import json
        
        # Compute summary statistics
        flop_stats["avg_flops_per_batch"] = (
            flop_stats["total_flops"] / flop_stats["total_batches"] 
            if flop_stats["total_batches"] > 0 else 0
        )
        flop_stats["avg_flops_per_cell"] = (
            flop_stats["total_flops"] / flop_stats["total_cells"] 
            if flop_stats["total_cells"] > 0 else 0
        )
        
        # Convert large numbers to readable format
        def format_flops(flops):
            if flops >= 1e12:
                return f"{flops/1e12:.2f} TFLOPs"
            elif flops >= 1e9:
                return f"{flops/1e9:.2f} GFLOPs"
            elif flops >= 1e6:
                return f"{flops/1e6:.2f} MFLOPs"
            elif flops >= 1e3:
                return f"{flops/1e3:.2f} KFLOPs"
            else:
                return f"{flops:.2f} FLOPs"
        
        flop_stats["total_flops_readable"] = format_flops(flop_stats["total_flops"])
        flop_stats["avg_flops_per_cell_readable"] = format_flops(flop_stats["avg_flops_per_cell"])
        flop_stats["avg_flops_per_batch_readable"] = format_flops(flop_stats["avg_flops_per_batch"])
        
        flops_json_path = os.path.join(results_dir, "flops_stats.json")
        with open(flops_json_path, "w") as f:
            json.dump(flop_stats, f, indent=2)
        
        logger.info(f"Saved FLOP statistics to {flops_json_path}")
        logger.info(f"Total FLOPs: {flop_stats['total_flops_readable']}")
        logger.info(f"Avg FLOPs per cell: {flop_stats['avg_flops_per_cell_readable']}")
    
    # Save retrieval information if collected
    if len(all_retrieval_info) > 0:
        logger.info("Saving retrieval information to CSV...")
        retrieval_csv_path = os.path.join(results_dir, "retrieval_info.csv")
        
        # Compute DEGs using GenGeneEval for each unique perturbation 
        deg_dict = {}
        try:
            from gge.deg import evaluate_degs
            
            # DEG detection parameters (direct parameters, not DEGSettings)
            deg_method = "welch"        # Welch's t-test (recommended for single-cell)
            pval_threshold = 0.1        # More lenient for discovery
            lfc_threshold = 0.25        # Lenient log fold change 
            control_key = "non-targeting"  # Control condition identifier (adjusted to match the actual control)
            
            logger.info(f"Computing DEGs using GenGeneEval with settings: method={deg_method}, p<{pval_threshold}, |LFC|>{lfc_threshold}, control_key='{control_key}'")
            
            # Only compute DEGs if we have sufficient data
            # if adata_real.n_obs > 50:  # Minimum cells for meaningful DEG analysis
            #     control_pert = data_module.get_control_pert() 
            #     logger.info(f"Control perturbation identifier: '{control_pert}'")
                
            #     # Get unique perturbations (excluding control)
            #     unique_perts = adata_real.obs[data_module.pert_col].unique()
            #     non_control_perts = [p for p in unique_perts if p != control_pert]
            #     logger.info(f"Found {len(non_control_perts)} non-control perturbations for DEG analysis")
                
            #     # Process perturbations in batches for memory efficiency
            #     batch_size = 5  # Process 5 perturbations at a time to avoid memory issues
                
            #     for i in range(0, len(non_control_perts), batch_size):
            #         batch_perts = non_control_perts[i:i+batch_size]
            #         logger.info(f"Processing DEG batch {i//batch_size + 1}/{(len(non_control_perts)-1)//batch_size + 1}: {len(batch_perts)} perturbations")
                    
            #         for pert_name in batch_perts:
            #             try:
            #                 # Get cells for this perturbation vs control
            #                 pert_mask = adata_real.obs[data_module.pert_col] == pert_name
            #                 ctrl_mask = adata_real.obs[data_module.pert_col] == control_pert
                            
            #                 n_pert = pert_mask.sum()
            #                 n_ctrl = ctrl_mask.sum()
                            
            #                 if n_pert < 3 or n_ctrl < 3:
            #                     logger.warning(f"  Skipping {pert_name}: insufficient cells (pert={n_pert}, ctrl={n_ctrl})")
            #                     deg_dict[pert_name] = []
            #                     continue
                            
            #                 # Create subset with perturbation + control cells
            #                 subset_mask = pert_mask | ctrl_mask
            #                 adata_subset = adata_real[subset_mask].copy()
                            
            #                 # Convert to numpy arrays as required by evaluate_degs
            #                 real_data = adata_subset.X
            #                 if hasattr(real_data, 'toarray'):
            #                     real_data = real_data.toarray()
            #                 real_data = real_data.astype(np.float32)  # Ensure correct dtype
                            
            #                 logger.debug(f"  Computing DEGs for {pert_name} (n_pert={n_pert}, n_ctrl={n_ctrl}, shape={real_data.shape})")
                            
            #                 # Simple fallback DEG computation using fold change
            #                 try:
            #                     # Get expression data for perturbation and control
            #                     pert_data = real_data[adata_subset.obs[data_module.pert_col] == pert_name]
            #                     ctrl_data = real_data[adata_subset.obs[data_module.pert_col] == control_pert]
                                
            #                     # Compute mean expression
            #                     pert_mean = np.mean(pert_data, axis=0)
            #                     ctrl_mean = np.mean(ctrl_data, axis=0)
                                
            #                     # Compute log2 fold change (add pseudocount to avoid log(0))
            #                     pseudocount = 1e-6
            #                     log2fc = np.log2((pert_mean + pseudocount) / (ctrl_mean + pseudocount))
                                
            #                     # Find genes with significant fold change
            #                     sig_genes_idx = np.abs(log2fc) > lfc_threshold
                                
            #                     if sig_genes_idx.any():
            #                         # Load gene name mapping if available
            #                         gene_mapping = load_gene_mapping(args.output_dir)
                                    
            #                         # Get gene identifiers - this dataset uses numerical indices instead of gene names
            #                         gene_names_array = adata_subset.var_names.values
                                    
            #                         # Check if we have actual gene symbols or just indices
            #                         sample_names = gene_names_array[:5]
            #                         using_indices = all(str(name).replace('.', '').replace('-', '').isdigit() for name in sample_names)
                                    
            #                         sig_genes = gene_names_array[sig_genes_idx]
                                    
            #                         # Sort by absolute fold change (descending)
            #                         abs_fc = np.abs(log2fc[sig_genes_idx])
            #                         sorted_idx = np.argsort(abs_fc)[::-1]
                                    
            #                         # Convert to list and map to real gene names if mapping available
            #                         if using_indices and gene_mapping:
            #                             # Convert indices to real gene names using mapping
            #                             deg_gene_list = []
            #                             for gene_idx in sig_genes[sorted_idx]:
            #                                 idx_as_int = int(gene_idx)
            #                                 if idx_as_int in gene_mapping:
            #                                     deg_gene_list.append(gene_mapping[idx_as_int])
            #                                 else:
            #                                     deg_gene_list.append(f"gene_{gene_idx}")  # fallback
            #                             logger.debug(f"      Using real gene names from mapping")
            #                         elif using_indices:
            #                             deg_gene_list = [f"gene_{gene}" for gene in sig_genes[sorted_idx]]
            #                             logger.debug(f"      Using gene indices (prefixed with 'gene_')")
            #                         else:
            #                             deg_gene_list = [str(gene) for gene in sig_genes[sorted_idx]]
            #                             logger.debug(f"      Using actual gene names")
                                    
            #                         deg_dict[pert_name] = deg_gene_list
                                    
            #                         logger.debug(f"    {pert_name}: {len(deg_dict[pert_name])} DEGs found using fold change method")
            #                         if len(deg_dict[pert_name]) > 0:
            #                             logger.debug(f"      Top DEGs: {deg_dict[pert_name][:5]}")
            #                     else:
            #                         deg_dict[pert_name] = []
                                    
            #                 except Exception as deg_error:
            #                     logger.warning(f"    Fallback DEG computation failed for {pert_name}: {deg_error}")
            #                     deg_dict[pert_name] = []
                                
            #             except Exception as e:
            #                 logger.warning(f"  Failed to compute DEGs for {pert_name}: {e}")
            #                 deg_dict[pert_name] = []
                
            #     logger.info(f"✅ DEG computation completed for {len(deg_dict)} perturbations")
                
            #     # Summary statistics
            #     total_degs = sum(len(degs) for degs in deg_dict.values())
            #     avg_degs = total_degs / len(deg_dict) if deg_dict else 0
            #     logger.info(f"   Total DEGs found: {total_degs}, Average per perturbation: {avg_degs:.1f}")
                
            # else:
            #     logger.warning(f"Insufficient data for DEG analysis (only {adata_real.n_obs} cells)")
            #     deg_dict = {}
                
        except ImportError as e:
            logger.warning(f"gengeneeval.deg not available ({e}), DEGs will not be computed")
            deg_dict = {}
        except Exception as e:
            logger.warning(f"Failed to compute DEGs: {e}")
            import traceback
            logger.debug(f"DEG computation traceback: {traceback.format_exc()}")
            deg_dict = {}
        
        # Prepare data for CSV
        csv_rows = []
        
        for batch_info in all_retrieval_info:
            indices = batch_info["indices"]  # [B*S, k]
            weights = batch_info["weights"]  # [B*S, k]
            train_idx_to_name = batch_info["train_idx_to_name"]
            pert_names = batch_info["pert_names"]
            cell_types = batch_info["cell_types"]
            B = batch_info["B"]
            S = batch_info["S"]
            
            # Convert to numpy arrays BEFORE indexing
            if hasattr(indices, 'cpu'):
                indices = indices.cpu().numpy()
            if hasattr(weights, 'cpu'):
                weights = weights.cpu().numpy()
            
            # Ensure proper shape [B*S, k] by flattening first two dimensions if needed
            if indices.ndim == 3:
                indices = indices.reshape(-1, indices.shape[-1])
            if weights.ndim == 3:
                weights = weights.reshape(-1, weights.shape[-1])
            
            # Process each cell
            for cell_idx in range(indices.shape[0]):
                sample_idx = cell_idx // S
                
                # Get query perturbation name
                if pert_names is not None and sample_idx < len(pert_names):
                    query_pert = pert_names[sample_idx]
                    if isinstance(query_pert, (list, tuple)):
                        query_pert = query_pert[cell_idx % S] if cell_idx % S < len(query_pert) else query_pert[0]
                else:
                    query_pert = "unknown"
                
                # Get cell type
                if cell_types is not None and sample_idx < len(cell_types):
                    cell_type = cell_types[sample_idx]
                    if isinstance(cell_type, (list, tuple)):
                        cell_type = cell_type[cell_idx % S] if cell_idx % S < len(cell_type) else cell_type[0]
                else:
                    cell_type = "unknown"
                
                # Get retrieved perturbations and their weights for this cell
                retrieved_indices = indices[cell_idx]  # [k]
                retrieved_weights = weights[cell_idx]  # [k]
                
                # Map indices to names and build lists
                retrieved_perts = []
                selected_perts = []
                
                for k_idx in range(len(retrieved_indices)):
                    train_idx = int(retrieved_indices[k_idx])
                    weight = float(retrieved_weights[k_idx])
                    
                    pert_name = train_idx_to_name.get(train_idx, f"idx_{train_idx}")
                    
                    retrieved_perts.append(f"{pert_name}(w={weight:.2f})")
                    
                    if weight >= 0.99:  # Use >= 0.99 for float comparison
                        selected_perts.append(pert_name)
                
                # Count selected perturbations
                n_selected = len(selected_perts)
                
                # Get DEGs for this perturbation
                # degs_for_pert = deg_dict.get(str(query_pert), [])
                # degs_str = "|".join(degs_for_pert) if degs_for_pert else "none"
                
                # Create row
                csv_rows.append({
                    "cell_type": str(cell_type),
                    "pert_name": str(query_pert),
                    "retrieved_perturbations": "|".join(retrieved_perts),
                    "selected_perturbations": "|".join(selected_perts) if selected_perts else "none",
                    "n_selected": n_selected,
                    "n_retrieved": len(retrieved_perts),
                    # "degs": degs_str,
                    # "n_degs": len(degs_for_pert)
                })
        
        # Save to CSV
        import csv
        if csv_rows:
            with open(retrieval_csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "cell_type", "pert_name", "retrieved_perturbations", 
                    "selected_perturbations", "n_selected", "n_retrieved",
                    "degs", "n_degs"
                ])
                writer.writeheader()
                writer.writerows(csv_rows)
            
            logger.info(f"Saved retrieval info to {retrieval_csv_path} ({len(csv_rows)} cells)")
        else:
            logger.warning("No retrieval information to save")

    if not args.predict_only:
        # 6. Compute metrics using cell-eval
        logger.info("Computing metrics using cell-eval...")

        control_pert = data_module.get_control_pert()

        # ── Patch 1: vectorized DEG t-test (replaces slow per-gene scipy loop) ──
        def _fit_deg_fast(self, real_data):
            from scipy import stats as _stats
            real_mean = real_data.mean(axis=0)
            ctrl_mean = self.control_data.mean(axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                log2fc = np.log2(real_mean + 1) - np.log2(ctrl_mean + 1)
            if real_data.shape[0] > 1 and self.control_data.shape[0] > 1:
                _, pvalues = _stats.ttest_ind(
                    real_data, self.control_data, axis=0, equal_var=False
                )
                pvalues = np.where(np.isnan(pvalues), 1.0, pvalues)
            else:
                pvalues = np.ones(real_data.shape[1])
            is_deg = (np.abs(log2fc) >= self.config.deg_lfc) & (pvalues <= self.config.deg_pval)
            if self.config.n_top_degs is not None and is_deg.sum() > self.config.n_top_degs:
                deg_indices = np.where(is_deg)[0]
                top_idx = deg_indices[
                    np.argsort(np.abs(log2fc[deg_indices]))[::-1][: self.config.n_top_degs]
                ]
                is_deg = np.zeros(real_data.shape[1], dtype=bool)
                is_deg[top_idx] = True
            if is_deg.sum() == 0:
                is_deg = np.ones(real_data.shape[1], dtype=bool)
            self._deg_mask = is_deg
            self._deg_genes = [g for g, m in zip(self.gene_names, is_deg) if m]
            self._is_fitted = True

        from gge.spaces import SpaceTransformer as _SpaceTransformer
        _SpaceTransformer._fit_deg = _fit_deg_fast

        # ── Patch 2: re-fit DEG transformer per condition ─────────────────────
        # The library caches the transformer after the first condition, so all
        # conditions would share the same DEG mask. Resetting it per call forces
        # a fresh DEG set for each condition, which is the correct behaviour.
        # IMPORTANT: this patch must be applied exactly once. Applying it inside
        # a loop causes Python closure variable capture to create infinite
        # recursion (_orig_base_compute ends up pointing to the previous
        # iteration's closure rather than the real original).
        from gge.metrics.base_metric import BaseMetric as _BaseMetric
        _orig_base_compute = _BaseMetric.compute

        def _per_condition_compute(self, real, generated, gene_names=None,
                                   aggregate_method="mean", condition=None,
                                   split=None, control_data=None,
                                   _orig=_orig_base_compute):  # capture by value
            if self.space == "deg":
                self._transformer = None
            return _orig(
                self, real, generated, gene_names,
                aggregate_method, condition, split, control_data,
            )

        _BaseMetric.compute = _per_condition_compute

        # ── Patch 3: graceful save when per-gene array length != n_genes ──────
        # DEG-space metrics produce arrays of length n_DEGs (varies per
        # condition) which is shorter than self.gene_names → ValueError.
        from gge.results import EvaluationResult as _EvalResult
        _orig_per_gene_df = _EvalResult.to_per_gene_dataframe

        def _safe_per_gene_df(self, metric_name, _orig=_orig_per_gene_df):
            try:
                return _orig(self, metric_name)
            except Exception:
                return pd.DataFrame()

        _EvalResult.to_per_gene_dataframe = _safe_per_gene_df

        # ── GPU-accelerated EnergyDistance ────────────────────────────────────
        _cuda_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        class _GPUEnergyDistance(EnergyDistance):
            def compute_per_gene(self, real, generated):
                try:
                    from geomloss import SamplesLoss as _SL
                    real = np.asarray(real, dtype=np.float64)
                    generated = np.asarray(generated, dtype=np.float64)
                    if real.ndim == 1:
                        real = real.reshape(-1, 1)
                    if generated.ndim == 1:
                        generated = generated.reshape(-1, 1)
                    n_genes = real.shape[1]
                    distances = np.zeros(n_genes)
                    loss_fn = _SL(loss="energy", blur=0.5, backend="tensorized")
                    for i in range(n_genes):
                        r = real[:, i][~np.isnan(real[:, i])]
                        g = generated[:, i][~np.isnan(generated[:, i])]
                        if len(r) == 0 or len(g) == 0:
                            distances[i] = np.nan
                            continue
                        r_t = torch.tensor(r.reshape(-1, 1), dtype=torch.float32, device=_cuda_device)
                        g_t = torch.tensor(g.reshape(-1, 1), dtype=torch.float32, device=_cuda_device)
                        distances[i] = loss_fn(r_t, g_t).item()
                    return distances
                except Exception:
                    return super().compute_per_gene(real, generated)

        metrics = [
            # MeanPearsonCorrelation(space="deg", deg_lfc=0.05, deg_pval=0.1),
            # MeanSpearmanCorrelation(space="deg", deg_lfc=0.05, deg_pval=0.1),
            RSquared(space="deg", deg_lfc=0.05, deg_pval=0.1),
            MSEDistance(space="pca", n_components=50),
            Wasserstein2Distance(space="pca", n_components=50),
        ]

        results = evaluate_lazy(
            real_path=adata_real_path,
            generated_path=adata_pred_path,
            condition_columns=['gene', 'cell_line'],
            output_dir=results_dir+'_redundant',
            metrics=metrics,
            verbose=True,
            control_key=control_pert,
        )

        summary = results.summary()
        logger.info(
            "Evaluation complete in %s (%d split(s), %d condition column(s))",
            results_dir,
            summary.get("n_splits", 0),
            len(summary.get("condition_columns", [])),
        )

        for split_name, split_summary in summary.get("splits", {}).items():
            aggregates = split_summary.get("aggregates", {})
            if aggregates:
                logger.info("Aggregate metrics for split %s: %s", split_name, aggregates)
            else:
                logger.warning("Split %s completed without aggregate metrics", split_name)

        # backup_dir = Path(f"{results_dir}_redundant")
        # results.save(backup_dir)
        # logger.info("Saved redundant evaluation copy to %s", backup_dir)
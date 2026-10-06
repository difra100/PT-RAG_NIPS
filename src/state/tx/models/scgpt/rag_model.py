"""
scGPT extended with PT-RAG retrieval-augmentation.

Retrieval pipeline mirrors StateTransitionPerturbationModel:
  - GenePT embeddings used for cosine-similarity retrieval
  - Differentiable Gumbel-Softmax selection (PT-RAG)
  - Retrieved context injected into transformer output before the expression decoder

Three modes (controlled by constructor flags):
  1. scGPT + GenePT  : use_genept=True, rag=False
  2. scGPT + PT-RAG  : use_genept=True, rag=True, differentiable_rag=True
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lightning_model import scGPTForPerturbation
from .loss import masked_mse_loss
from .utils import map_raw_id_to_vocab_id
from ..utils import build_mlp

logger = logging.getLogger(__name__)


class scGPTRAGForPerturbation(scGPTForPerturbation):
    """scGPT with optional GenePT context injection and differentiable RAG."""

    def __init__(
        self,
        ntoken: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        nlayers_cls: int,
        n_cls: int,
        # RAG-specific
        rag: bool = False,
        topk_rag: int = 32,
        differentiable_rag: bool = False,
        gumbel_sparsity_loss: bool = False,
        gumbel_sparsity_weight: float = 0.01,
        use_genept: bool = False,
        pert_dim: int = 1536,
        
        **kwargs,
    ):
        super().__init__(
            ntoken=ntoken,
            d_model=d_model,
            nhead=nhead,
            d_hid=d_hid,
            nlayers=nlayers,
            nlayers_cls=nlayers_cls,
            n_cls=n_cls,
            **kwargs,
        )

        self.rag_mode = rag
        self.topk_rag = topk_rag
        self.differentiable_rag = differentiable_rag
        self.gumbel_sparsity_loss = gumbel_sparsity_loss
        self.gumbel_sparsity_weight = gumbel_sparsity_weight
        self.use_genept = use_genept
        self._gumbel_weights_cache: Optional[torch.Tensor] = None

        dropout = kwargs.get("dropout", 0.2)

        # GenePT → d_model projection (used by all RAG modes and plain GenePT)
        if use_genept or rag:
            self.genept_to_dmodel = build_mlp(
                in_dim=pert_dim,
                out_dim=d_model,
                hidden_dim=d_model,
                n_layers=1,
                dropout=dropout,
                activation=nn.GELU,
            )

        # Scoring & aggregation heads (shared by PT-RAG)
        if (rag and differentiable_rag):
            self.context_input_norm = nn.LayerNorm(3 * d_model)
            self.cell_pert_scorer = build_mlp(
                in_dim=3 * d_model,
                out_dim=2,
                hidden_dim=d_model,
                n_layers=2,
                dropout=dropout,
                activation=nn.GELU,
            )
            self.final_project_genept_emb = build_mlp(
                in_dim=3 * d_model,
                out_dim=d_model,
                hidden_dim=d_model,
                n_layers=1,
                dropout=dropout,
                activation=nn.GELU,
            )

    
    # ------------------------------------------------------------------
    # RAG context computation
    # ------------------------------------------------------------------

    def _compute_rag_context(
        self,
        cell_emb: torch.Tensor,
        pert_proj: torch.Tensor,
        pert_genept: torch.Tensor,
        batch: dict,
    ) -> Optional[torch.Tensor]:
        """
        Retrieve top-K similar perturbations and return a context vector.

        Args:
            cell_emb   : (B, d_model)  CLS/avg-pool cell embedding
            pert_proj  : (B, d_model)  GenePT embedding projected to d_model
            pert_genept: (B, genept_dim) raw GenePT embeddings for similarity search
            batch      : data batch dict (used to get pert_name for self-masking)

        Returns:
            context: (B, d_model) or None if no pool is available
        """
        if not (hasattr(self, "train_perts") and self.train_perts is not None):
            return None

        dev = pert_genept.device
        train_perts = self.train_perts.to(dev)  # (N, genept_dim)
        B = pert_genept.shape[0]

        # Cosine similarity retrieval
        query_norm = F.normalize(pert_genept, p=2, dim=1)     # (B, genept_dim)
        db_norm = F.normalize(train_perts, p=2, dim=1)        # (N, genept_dim)
        similarities = torch.matmul(query_norm, db_norm.t())  # (B, N)

        # Self-mask
        pert_names = batch.get("pert_name", None) if batch is not None else None
        if pert_names is not None and hasattr(self, "train_name_to_idx"):
            for i in range(B):
                pname = pert_names[i] if isinstance(pert_names[i], str) else str(pert_names[i])
                self_idx = self.train_name_to_idx.get(pname, -1)
                if self_idx >= 0:
                    similarities[i, self_idx] = float("-inf")

        _, indices = torch.topk(similarities, k=self.topk_rag, dim=1, largest=True)  # (B, K)

        closest_raw = train_perts[indices]                      # (B, K, genept_dim)
        closest_proj = self.genept_to_dmodel(closest_raw)      # (B, K, d_model)

        # Scoring pipeline (same as StateTransitionPerturbationModel)
        K = self.topk_rag
        context_input = torch.cat([
            cell_emb.unsqueeze(1).expand(-1, K, -1),   # (B, K, d_model)
            pert_proj.unsqueeze(1).expand(-1, K, -1),  # (B, K, d_model)
            closest_proj,                               # (B, K, d_model)
        ], dim=-1)  # (B, K, 3*d_model)

        context_input = self.context_input_norm(context_input)
        scores = self.cell_pert_scorer(context_input)             # (B, K, 2)
        scores = scores / scores.std().clamp(min=1e-6)
        weights = F.gumbel_softmax(scores, tau=0.5, dim=-1, hard=True)  # (B, K, 2)
        weights = weights[..., -1][..., None]                     # (B, K, 1)

        if self.training and self.gumbel_sparsity_loss:
            self._gumbel_weights_cache = weights

        embs = self.final_project_genept_emb(context_input)      # (B, K, d_model)
        context = (weights * embs).sum(dim=1)                     # (B, d_model)
        return context

    # ------------------------------------------------------------------
    # Override shared_step to inject GenePT + RAG context
    # ------------------------------------------------------------------

    def shared_step(self, batch, truncate=True):
        x_basal = batch["ctrl_cell_emb"]   # (B, n_genes)
        x_pert = batch["pert_cell_emb"]    # (B, n_genes)

        if self.perturbation_type == "chemical":
            pert_flags = torch.zeros_like(x_pert, dtype=torch.long)
        else:
            pert_flags = batch["pert_flags"]  # (B, n_genes)

        gene_ids = batch["gene_ids"][0]  # (n_genes,)
        nonpad_mask = gene_ids != self.pad_token_id

        x_basal = x_basal[:, nonpad_mask]
        x_pert = x_pert[:, nonpad_mask]
        gene_ids = gene_ids[nonpad_mask]
        pert_flags = pert_flags[:, nonpad_mask]

        batch_size, n_genes = x_basal.size()

        if self.include_zero_gene == "all":
            input_gene_ids = torch.arange(n_genes, device=x_basal.device, dtype=torch.long)
        else:
            input_gene_ids = x_basal.nonzero()[:, 1].flatten().unique().sort()[0]

        if truncate and len(input_gene_ids) > self.max_seq_len:
            input_gene_ids = torch.randperm(len(input_gene_ids), device=x_basal.device)[: self.max_seq_len]

        x_basal_seq = x_basal[:, input_gene_ids]
        x_pert_seq = x_pert[:, input_gene_ids]
        pert_flags_seq = pert_flags[:, input_gene_ids]

        mapped_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
        mapped_ids = mapped_ids.repeat(batch_size, 1)  # (B, seq_len)
        src_key_padding_mask = mapped_ids.eq(self.pad_token_id)

        # --- Encode with scGPT backbone ---
        transformer_output = self.model._encode(
            mapped_ids.long(),
            x_basal_seq,
            pert_flags_seq.long(),
            src_key_padding_mask,
        )  # (B, seq_len, d_model)

        # --- GenePT context injection ---
        pert_proj = None
        if (self.use_genept or self.rag_mode) and "pert_emb" in batch:
            pert_genept = batch["pert_emb"]             # (B, genept_dim)
            pert_proj = self.genept_to_dmodel(pert_genept)  # (B, d_model)
            transformer_output = transformer_output + pert_proj.unsqueeze(1)

        # --- RAG context injection ---
        if (self.rag_mode) and pert_proj is not None:
            cell_emb = self.model._get_cell_emb_from_layer(transformer_output, x_basal_seq)
            rag_context = self._compute_rag_context(cell_emb, pert_proj, pert_genept, batch)
            if rag_context is not None:
                transformer_output = transformer_output + rag_context.unsqueeze(1)

        # --- Decode ---
        mlm_raw = self.model.decoder(transformer_output)
        output_values = mlm_raw["pred"]  # (B, seq_len)

        masked_positions = torch.ones_like(x_basal_seq, dtype=torch.bool)
        loss = masked_mse_loss(output_values, x_pert_seq, masked_positions)

        return loss, {
            "mlm_output": output_values,
            "x_pred": output_values.float(),
            "x_true": x_pert_seq,
            "x_basal": x_basal_seq,
        }

    # ------------------------------------------------------------------
    # training_step: add Gumbel sparsity regularization
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        loss, batch_outputs = self.shared_step(batch)

        metrics = self.compute_metrics(
            x_basal=batch_outputs["x_basal"],
            x_pred=batch_outputs["x_pred"],
            x_true=batch_outputs["x_true"],
        )
        self.log("loss", loss, prog_bar=True)
        for k, v in metrics.items():
            self.log(k, v, prog_bar=True)

        if self.gumbel_sparsity_loss and self._gumbel_weights_cache is not None:
            sparsity_loss = torch.abs(self._gumbel_weights_cache).mean()
            self.log("train/gumbel_sparsity", sparsity_loss)
            self.log("train/gumbel_active_ratio", (self._gumbel_weights_cache > 0.5).float().mean())
            loss = loss + self.gumbel_sparsity_weight * sparsity_loss
            self._gumbel_weights_cache = None

        return loss

    # ------------------------------------------------------------------
    # Split-aware RAG pool swapping (identical to state_transition.py)
    # ------------------------------------------------------------------

    def _swap_rag_pool(self, pool: str) -> None:
        """Swap the active RAG retrieval pool. pool: 'train' | 'trainval' | 'all'."""
        for attr, dst in [("perts", "train_perts"), ("index", "train_index"), ("name_to_idx", "train_name_to_idx")]:
            src_key = f"_rag_{pool}_{attr}"
            if hasattr(self, src_key):
                setattr(self, dst, getattr(self, src_key))

    def on_train_epoch_start(self) -> None:
        self._swap_rag_pool("train")

    def on_validation_epoch_start(self) -> None:
        self._swap_rag_pool("trainval")

    def on_validation_epoch_end(self) -> None:
        self._swap_rag_pool("train")

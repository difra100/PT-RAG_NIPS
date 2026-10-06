from functools import partial
from pathlib import Path
from typing import Dict, Literal, Optional

from torch.utils.data import DataLoader, Dataset
from cell_load.data_modules import PerturbationDataModule
from cell_load.dataset import MetadataConcatDataset
from cell_load.data_modules.samplers import PerturbationBatchSampler

from .dataset import scGPTPerturbationDataset


class scGPTPerturbationDataModule(PerturbationDataModule):
    """
    Subclass of PerturbationDataModule that uses scGPTPerturbationDataset.

    The base DataModule always creates PerturbationDataset (no gene_ids / pert_flags)
    and uses PerturbationDataset.collate_fn.  This subclass overrides those two
    methods so that every split dataset is a proper scGPTPerturbationDataset and the
    DataLoader uses the matching collate_fn.
    """

    def __init__(
        self,
        vocab: Dict[str, int],
        perturbation_type: Literal["chemical", "genetic"] = "genetic",
        hvg_names_uns_key: Optional[str] = None,
        transform: Optional[str] = None,
        **kwargs,
    ):
        # Strip keys the base class doesn't accept that we added ourselves
        kwargs.pop("dataset_cls", None)

        super().__init__(**kwargs)

        self.vocab = vocab
        self.perturbation_type_str = perturbation_type
        self.hvg_names_uns_key = hvg_names_uns_key
        self.transform = transform

    # ------------------------------------------------------------------
    # Override: create scGPTPerturbationDataset instead of base class
    # ------------------------------------------------------------------

    def _create_base_dataset(self, dataset_name: str, fpath: Path) -> scGPTPerturbationDataset:
        mapping_kwargs = {
            "map_controls": self.map_controls,
            "cache_perturbation_control_pairs": self.cache_perturbation_control_pairs,
        }

        return scGPTPerturbationDataset(
            name=dataset_name,
            h5_path=fpath,
            mapping_strategy=self.mapping_strategy_cls(
                random_state=self.random_seed,
                n_basal_samples=self.n_basal_samples,
                **mapping_kwargs,
            ),
            embed_key=self.embed_key,
            pert_onehot_map=self.pert_onehot_map,
            batch_onehot_map=self.batch_onehot_map,
            cell_type_onehot_map=self.cell_type_onehot_map,
            pert_col=self.pert_col,
            cell_type_key=self.cell_type_key,
            batch_col=self.batch_col,
            control_pert=self.control_pert,
            random_state=self.random_seed,
            should_yield_control_cells=self.should_yield_control_cells,
            store_raw_expression=self.store_raw_expression,
            output_space=self.output_space,
            store_raw_basal=self.store_raw_basal,
            barcode=self.barcode,
            vocab=self.vocab,
            hvg_names_uns_key=self.hvg_names_uns_key,
            perturbation_type=self.perturbation_type_str,
        )

    # ------------------------------------------------------------------
    # Override: use scGPTPerturbationDataset.collate_fn
    # ------------------------------------------------------------------

    def _create_dataloader(
        self,
        datasets: list,
        test: bool = False,
        batch_size: int = None,
    ) -> DataLoader:
        collate_fn = partial(
            scGPTPerturbationDataset.collate_fn,
            transform=self.transform,
            pert_col=self.pert_col,
        )

        ds = MetadataConcatDataset(datasets)
        use_batch = self.basal_mapping_strategy == "batch"
        batch_size = batch_size or (1 if test else self.batch_size)

        sampler = PerturbationBatchSampler(
            dataset=ds,
            batch_size=batch_size,
            drop_last=self.drop_last,
            cell_sentence_len=self.cell_sentence_len,
            test=test,
            use_batch=use_batch,
        )

        return DataLoader(
            ds,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            prefetch_factor=4 if not test and self.num_workers > 0 else None,
        )

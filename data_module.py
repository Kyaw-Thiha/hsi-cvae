from __future__ import annotations

from typing import Optional, Sequence

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from dataset import HyperspectralDataset


class HyperspectralDataModule(L.LightningDataModule):
    """LightningDataModule that splits the CSV-backed dataset into train/val/test."""

    def __init__(
        self,
        csv_path: str,
        condition_columns: Sequence[str],
        spectral_range: tuple[int, int, int],
        batch_size: int = 256,
        num_workers: int = 4,
        splits: tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self.csv_path = csv_path
        self.condition_columns = list(condition_columns)
        self.spectral_range = spectral_range
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.splits = splits
        self.seed = seed
        self.pin_memory = pin_memory

        self.dataset: Optional[HyperspectralDataset] = None
        self.train_set: Optional[Dataset] = None
        self.val_set: Optional[Dataset] = None
        self.test_set: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset is not None:
            return

        self.dataset = HyperspectralDataset(
            csv_path=self.csv_path,
            condition_columns=self.condition_columns,
            spectral_range=self.spectral_range,
        )

        n_total = len(self.dataset)
        train_len = int(n_total * self.splits[0])
        val_len = int(n_total * self.splits[1])
        test_len = n_total - train_len - val_len

        self.train_set, self.val_set, self.test_set = random_split(
            self.dataset,
            lengths=[train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(self.seed),
        )

    def dataloader(self, dataset: Optional[Dataset], shuffle: bool) -> DataLoader:
        if dataset is None:
            raise RuntimeError("DataModule.setup() must be called before requesting dataloaders.")
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=self.pin_memory,
        )

    def train_dataloader(self) -> DataLoader:
        return self.dataloader(self.train_set, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self.dataloader(self.val_set, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self.dataloader(self.test_set, shuffle=False)

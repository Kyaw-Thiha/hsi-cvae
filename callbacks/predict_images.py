from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from torch import Tensor
from torchvision.utils import make_grid, save_image


def _to_01(x: Tensor) -> Tensor:
    # Map from [-1,1] to [0,1]; clamp in case
    return (x.clamp(-1, 1) + 1) * 0.5


class SavePredictionsCallback(Callback):
    """Collect predict() outputs, save individual PNGs, and optional grids."""

    def __init__(self, out_dir: str, grid_nrow: Optional[int] = None, label_key: Optional[str] = None) -> None:
        super().__init__()
        self.out_dir = out_dir
        self.grid_nrow = grid_nrow
        self.label_key = label_key
        self._imgs: List[Tensor] = []
        self._ys: List[Tensor] = []

    def on_predict_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._imgs.clear()
        self._ys.clear()
        if trainer.is_global_zero:
            Path(self.out_dir).mkdir(parents=True, exist_ok=True)
            (Path(self.out_dir) / "samples").mkdir(parents=True, exist_ok=True)
            (Path(self.out_dir) / "grids").mkdir(parents=True, exist_ok=True)

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Tensor,
        batch: Dict[str, Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._imgs.append(outputs.detach().cpu())
        if self.label_key and isinstance(batch, Dict) and self.label_key in batch:
            label_tensor = batch[self.label_key]
            if not isinstance(label_tensor, torch.Tensor):
                label_tensor = torch.as_tensor(label_tensor)
            self._ys.append(label_tensor.detach().cpu())

    def on_predict_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not trainer.is_global_zero:
            return
        imgs = _to_01(torch.cat(self._imgs, dim=0))
        labels = torch.cat(self._ys, dim=0) if self._ys else None

        # per-sample
        samp_dir = Path(self.out_dir) / "samples"
        for i in range(imgs.size(0)):
            suffix = ""
            if labels is not None:
                suffix = f"_{self._format_label(labels[i])}"
            save_image(imgs[i], str(samp_dir / f"{i:05d}{suffix}.png"))

        # per-class grids
        grid_dir = Path(self.out_dir) / "grids"
        if labels is not None and labels.ndim == 1 and torch.all(labels == labels.long()):
            unique_classes = torch.unique(labels.long())
            for c in unique_classes.tolist():
                idx = (labels.long() == c).nonzero(as_tuple=True)[0]
                if idx.numel() == 0:
                    continue
                cls_imgs = imgs[idx]
                nrow = self.grid_nrow or cls_imgs.size(0)
                grid = make_grid(cls_imgs, nrow=nrow, padding=2)
                save_image(grid, str(grid_dir / f"class_{c}.png"))
        else:
            nrow = self.grid_nrow or imgs.size(0)
            grid = make_grid(imgs, nrow=nrow, padding=2)
            save_image(grid, str(grid_dir / "all_samples.png"))

    @staticmethod
    def _format_label(label: Tensor) -> str:
        if label.ndim == 0:
            value = label.item()
            return f"label{int(value)}"
        if label.ndim == 1:
            values = "-".join(f"{float(v):.3f}" for v in label.tolist())
            return f"meta_{values}"
        return "meta"

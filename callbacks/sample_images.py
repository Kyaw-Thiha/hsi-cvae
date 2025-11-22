from __future__ import annotations
import os
import torch
from torch import Tensor
from torchvision.utils import save_image, make_grid
from lightning.pytorch.callbacks import Callback
from lightning.pytorch import LightningModule, Trainer


class SampleImages(Callback):
    """
    Lightning callback that generates and saves a grid of class-conditional samples
    at the end of each validation epoch.

    Behavior:
    - For each class label in [0, num_classes), draws `num_per_class` samples using the
      model's sampling function (`LightningModule.sample`) and arranges them into a grid.
    - Saves a PNG to `<default_root_dir>/samples/epoch_XXXX.png`.

    Requirements on the LightningModule:
    - Exposes an integer attribute `num_classes`.
    - Exposes a method `sample(n: int, y: Tensor, device: Optional[torch.device]) -> Tensor`
      returning images shaped (N, C, H, W) with values in roughly [-1, 1] (will be rescaled to [0, 1]).
    """

    def __init__(self, num_per_class: int = 8) -> None:
        super().__init__()
        self.num_per_class = num_per_class

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        Called by Lightning after each validation epoch. If a logger is present and the
        `pl_module` exposes the required API (`num_classes` and `sample`), generates a grid of
        samples and writes it to disk under `<default_root_dir>/samples/`.

        Args:
            trainer: The current Trainer instance (used for logging directory and epoch index).
            pl_module: The LightningModule being validated; expected to provide `num_classes`
                       and a `sample` method as described in the class docstring.
        """
        if not trainer.logger:
            return
        # These are provided as typed attrs by your LightningModule (see above)
        num_classes = getattr(pl_module, "num_classes", None)
        sample_fn = getattr(pl_module, "sample", None)
        if num_classes is None or sample_fn is None:
            # Nothing to do if the module doesn't expose the typed API
            return

        device = pl_module.device
        labels = torch.arange(int(num_classes), device=device).repeat_interleave(self.num_per_class)
        imgs: Tensor = sample_fn(
            n=labels.numel(), y=labels, device=device, temperature=1.0, guidance_scale=0, cond_scale=1.0
        )  # (N,C,H,W)
        imgs = (imgs.clamp(-1, 1) + 1.0) / 2.0  # [-1,1] -> [0,1]

        grid = make_grid(imgs, nrow=self.num_per_class)
        out_path = f"{trainer.default_root_dir}/samples/epoch_{trainer.current_epoch}.png"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        save_image(grid, out_path)

from __future__ import annotations

import time
from pathlib import Path

import yaml
from lightning.pytorch.callbacks import Callback


class SaveTrainingTime(Callback):
    """Save wall-clock training time to outputs/checkpoints."""

    def __init__(self, out_dir: str = "outputs/checkpoints") -> None:
        super().__init__()
        self.out_dir = Path(out_dir)
        self._start_time: float | None = None

    def on_fit_start(self, trainer, pl_module) -> None:  # type: ignore[override]
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._start_time = time.monotonic()

    def on_fit_end(self, trainer, pl_module) -> None:  # type: ignore[override]
        if self._start_time is None:
            return
        elapsed = time.monotonic() - self._start_time
        architecture = getattr(pl_module, "architecture", "model")
        payload = {
            "architecture": architecture,
            "elapsed_seconds": float(elapsed),
            "global_step": int(getattr(trainer, "global_step", 0)),
            "max_epochs": int(getattr(trainer, "max_epochs", 0)),
        }

        out_path = self.out_dir / f"training_time_{architecture}.yaml"
        with out_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)

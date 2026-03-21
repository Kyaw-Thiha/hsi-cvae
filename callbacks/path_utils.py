from __future__ import annotations

from pathlib import Path


def resolve_callback_out_dir(default_root_dir: str, out_dir: str | Path) -> Path:
    """
    Resolve callback output directories relative to trainer.default_root_dir.

    Backward compatibility:
    - Legacy configs often use `outputs/...`.
    - When default_root_dir points to a run folder (e.g. outputs/run_A), remap
      `outputs/foo` -> `<default_root_dir>/foo` instead of `<default_root_dir>/outputs/foo`.
    """
    configured = Path(out_dir)
    if configured.is_absolute():
        return configured

    root = Path(default_root_dir)
    parts = configured.parts
    if parts and parts[0] == "outputs":
        if len(parts) == 1:
            return root
        configured = Path(*parts[1:])

    return root / configured

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill CVAE setup JSON files for existing run directories.")
    parser.add_argument("--runs-root", type=str, default="outputs/ablations")
    parser.add_argument("--run-glob", type=str, default="run_*")
    parser.add_argument("--artifacts-subdir", type=str, default="artifacts")
    parser.add_argument("--filename", type=str, default="cfg_tcvae_setup.json")
    return parser.parse_args()


def latest_version_dir(run_dir: Path) -> Optional[Path]:
    logs_dir = run_dir / "lightning_logs"
    if not logs_dir.exists():
        return None
    version_dirs = [p for p in logs_dir.glob("version_*") if p.is_dir() and (p / "config.yaml").exists()]
    if not version_dirs:
        return None
    return max(version_dirs, key=lambda p: p.stat().st_mtime)


def latest_checkpoint_in_dir(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    dirpath = Path(path)
    if not dirpath.exists():
        return None
    checkpoints = [p for p in dirpath.glob("*.ckpt") if p.is_file()]
    if not checkpoints:
        return None
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    return str(latest)


def _checkpoint_dirs_from_cfg(cfg: dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    trainer_cfg = dict(cfg.get("trainer") or {})
    callbacks_cfg = list(trainer_cfg.get("callbacks") or [])
    best_dir: Optional[str] = None
    interval_dir: Optional[str] = None
    for cb in callbacks_cfg:
        if not isinstance(cb, dict):
            continue
        init_args = dict(cb.get("init_args") or {})
        dirpath = init_args.get("dirpath")
        if not dirpath:
            continue
        monitor = init_args.get("monitor")
        if monitor:
            best_dir = str(dirpath)
        else:
            interval_dir = str(dirpath)
    return best_dir, interval_dir


def _infer_run_end_ts(run_dir: Path) -> float:
    end_ts = run_dir.stat().st_mtime
    for path in run_dir.rglob("*"):
        if not path.is_file():
            continue
        if "artifacts" in path.parts:
            continue
        ts = path.stat().st_mtime
        if ts > end_ts:
            end_ts = ts
    return end_ts


def _infer_ckpt_for_run(checkpoint_dir: Optional[str], run_start_ts: float, run_end_ts: float) -> Optional[str]:
    if checkpoint_dir is None:
        return None
    dirpath = Path(checkpoint_dir)
    if not dirpath.exists():
        return None
    candidates = [p for p in dirpath.glob("*.ckpt") if p.is_file()]
    if not candidates:
        return None
    window_start = run_start_ts - 300.0
    window_end = run_end_ts + 300.0
    window = [p for p in candidates if window_start <= p.stat().st_mtime <= window_end]
    if window:
        chosen = max(window, key=lambda p: p.stat().st_mtime)
        return str(chosen)
    # Fallback: nearest checkpoint by timestamp.
    chosen = min(candidates, key=lambda p: abs(p.stat().st_mtime - run_end_ts))
    return str(chosen)


def model_block(model_cfg: dict[str, Any]) -> dict[str, Any]:
    block: dict[str, Any] = {
        "model_type": "ConditionalVAE",
        "architecture": model_cfg.get("architecture"),
        "input_dim": model_cfg.get("input_dim"),
        "condition_dim": model_cfg.get("condition_dim"),
        "latent_dim": model_cfg.get("latent_dim"),
        "hidden_dims": model_cfg.get("hidden_dims"),
        "dropout": model_cfg.get("dropout"),
        "wavelength_params": model_cfg.get("wavelength_params"),
    }
    for key in (
        "cnn_params",
        "transformer_params",
        "transformer_repeatz_params",
        "dual_path_transformer_params",
        "conformer_params",
    ):
        value = model_cfg.get(key)
        if value is not None:
            block[key] = value
    return block


def artifact_paths(artifact_dir: Path) -> dict[str, Optional[str]]:
    file_keys = {
        "norm_dict_path": "norm_dict.json",
        "minmax_stats_path": "minmax_stats.json",
        "psi1_path": "psi1_gdstreamline.csv",
        "psi2_path": "psi2_gdstreamline.csv",
        "psi1_normalized_path": "psi1_train_normalized.csv",
        "psi2_normalized_path": "psi2_test_val_normalized.csv",
        "psi1_conditions_path": "psi1_train_conditions_normalized.csv",
        "psi2_conditions_path": "psi2_test_val_conditions_normalized.csv",
        "psi1_spectra_path": "psi1_train_spectra_normalized.csv",
        "psi2_spectra_path": "psi2_test_val_spectra_normalized.csv",
        "artifact_manifest_path": "artifact_manifest.json",
    }
    result: dict[str, Optional[str]] = {}
    for key, filename in file_keys.items():
        path = artifact_dir / filename
        result[key] = str(path) if path.exists() else None
    return result


def git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def build_payload(cfg: dict[str, Any], run_dir: Path, version_dir: Path, artifact_dir: Path) -> dict[str, Any]:
    model_cfg = dict(cfg.get("model") or {})
    data_cfg = dict(cfg.get("data") or {})
    trainer_cfg = dict(cfg.get("trainer") or {})
    best_dir, interval_dir = _checkpoint_dirs_from_cfg(cfg)

    run_start_ts = (version_dir / "config.yaml").stat().st_mtime
    run_end_ts = _infer_run_end_ts(run_dir)
    best_ckpt_path = _infer_ckpt_for_run(best_dir, run_start_ts, run_end_ts)
    interval_ckpt_path = _infer_ckpt_for_run(interval_dir, run_start_ts, run_end_ts)
    if interval_ckpt_path is None:
        interval_ckpt_path = latest_checkpoint_in_dir(interval_dir)

    block = model_block(model_cfg)
    tcvae_block = {
        "model": block,
        "loss": {
            "loss_name": model_cfg.get("loss_name"),
            "loss_params": model_cfg.get("loss_params"),
        },
        "optimizer": {
            "lr": model_cfg.get("lr"),
            "weight_decay": model_cfg.get("weight_decay"),
        },
        "scheduler": model_cfg.get("scheduler_cfg"),
        "data": {
            "csv_path": data_cfg.get("csv_path"),
            "condition_columns": data_cfg.get("condition_columns"),
            "spectral_range": data_cfg.get("spectral_range"),
            "splits": data_cfg.get("splits"),
            "seed": data_cfg.get("seed"),
            "batch_size": data_cfg.get("batch_size"),
            "num_workers": data_cfg.get("num_workers"),
            "pin_memory": data_cfg.get("pin_memory"),
        },
        "sampling": {
            "temperature": model_cfg.get("temperature"),
            "guidance_scale": model_cfg.get("guidance_scale"),
            "condition_scale": model_cfg.get("condition_scale"),
            "condition_dropout": model_cfg.get("condition_dropout"),
        },
        "artifacts": {
            "run_dir": str(run_dir),
            "config_yaml": str(version_dir / "config.yaml"),
            "hparams_yaml": str(version_dir / "hparams.yaml"),
            "resume_from_ckpt_path": cfg.get("ckpt_path"),
            "checkpoint_dirs": {
                "best": best_dir,
                "interval": interval_dir,
            },
            "best_ckpt_path": best_ckpt_path,
            "interval_latest_ckpt_path": interval_ckpt_path,
            **artifact_paths(artifact_dir),
        },
        "provenance": {
            "subcommand": cfg.get("subcommand"),
            "seed_everything": cfg.get("seed_everything"),
            "trainer": {
                "max_epochs": trainer_cfg.get("max_epochs"),
                "precision": trainer_cfg.get("precision"),
                "default_root_dir": trainer_cfg.get("default_root_dir"),
            },
            "exported_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(),
        },
    }
    return {
        "cvae_type": block.get("model_type", "ConditionalVAE"),
        "cfg_tcvae": tcvae_block,
        "cfg_cvae": tcvae_block,
    }


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root)
    run_dirs = sorted([p for p in runs_root.glob(args.run_glob) if p.is_dir()], key=lambda p: p.name)

    written = 0
    skipped = 0
    for run_dir in run_dirs:
        version_dir = latest_version_dir(run_dir)
        if version_dir is None:
            skipped += 1
            print(f"[skip] {run_dir}: no lightning_logs/version_*/config.yaml")
            continue

        config_path = version_dir / "config.yaml"
        with config_path.open("r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle) or {}

        artifact_dir = run_dir / args.artifacts_subdir
        artifact_dir.mkdir(parents=True, exist_ok=True)
        payload = build_payload(cfg, run_dir, version_dir, artifact_dir)
        out_path = artifact_dir / args.filename
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        written += 1
        print(f"[ok] {out_path}")

    print(f"done: wrote={written}, skipped={skipped}")


if __name__ == "__main__":
    main()

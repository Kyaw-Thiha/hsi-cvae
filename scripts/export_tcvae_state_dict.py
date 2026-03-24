from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export plain model.state_dict() from a Lightning tcVAE checkpoint.")
    parser.add_argument("--ckpt", type=Path, required=True, help="Path to Lightning .ckpt file.")
    parser.add_argument("--out", type=Path, required=True, help="Output path for plain tcVAE state dict.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state_dict = payload.get("state_dict", payload)

    model_state = {key.removeprefix("model."): value for key, value in state_dict.items() if key.startswith("model.")}
    if not model_state:
        raise SystemExit("No model.* weights found in checkpoint; expected a Lightning checkpoint with model submodule weights.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model_state, args.out)
    print(f"Exported tcVAE state dict to {args.out}")


if __name__ == "__main__":
    main()

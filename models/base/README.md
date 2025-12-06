# MLP (Base) CVAE

The base architecture is a symmetric multilayer perceptron used as both encoder and decoder. It’s the simplest option when you want a lightweight baseline without convolutional or attention layers.

## Encoder
- Concatenates the spectrum vector with the conditioning vector along the feature dimension.
- Passes the joint vector through an MLP built via `build_mlp` (Linear → BatchNorm → ReLU [+ Dropout]).
- Splits into two linear heads to produce `mu` and `logvar` for the latent distribution.

## Decoder
- Concatenates latent sample `z` with the conditioning vector.
- Runs the mirrored MLP (hidden dims reversed) and projects back to `input_dim`, followed by `sigmoid` to keep outputs in `[0,1]`.

## Default Hyperparameters
- `hidden_dims = [512, 256, 128]`
- `dropout = 0.0` (override via CLI if needed)
These defaults live in `config/base.yaml` and `model.py`; change them by editing the config or passing `model.hidden_dims=[...]` when launching the CLI.

## Usage
```bash
python main.py fit --config config/base.yaml
```
To switch from MLP to other architectures, stack an additional config (e.g., `config/models/transformer.yaml`).

## Sampling Notes
- Conditioning vectors must match `condition_dim`; `sample()` can generate random simplex weights if none provided.
- `decode(z, cond)` assumes inputs are in the normalized range used during training; outputs are `[0,1]` and later rescaled to `[-1,1]` inside the Lightning module.

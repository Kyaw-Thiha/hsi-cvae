# CNN CVAE

This variant replaces the MLP encoder/decoder with 1D convolutional stacks so the model can capture local spectral structure while still operating on the full wavelength range.

## Encoder
- Treats the spectrum as a single-channel 1D signal (`(batch, 1, seq_len)`), then concatenates projected conditioning maps (see below) to form the input channels.
- Applies a stack of `ConvBlock`s (Conv1d → BatchNorm → ReLU [+ Dropout]) defined by `conv_channels`.
- Flattens the final feature map and feeds it to linear heads to produce `mu` and `logvar`.

## Conditioning Projection
`ConditionInjector` (`models/cnn/cvae.py:22-40`) linearly maps the conditioning vector to `(batch, cond_channels, seq_len)` so each conv block sees conditioning information as extra feature maps. Set `cond_channels` > 0 to enable; otherwise the spectrum is processed alone.

## Decoder
- Concatenates latent + condition vectors, projects them to the first decoder channel map, and reshapes to `(batch, channels, seq_len)`.
- Runs the reversed convolutional stack, ending with a `1x1` convolution and `sigmoid` to produce spectra in `[0,1]`.

## Defaults & Configuration
- `model.py` requires `cnn_params.conv_channels` and optionally accepts `dropout` and `cond_channels`.
- Typical config snippet (`config/base.yaml`):
  ```yaml
  model:
    architecture: cnn
    cnn_params:
      conv_channels: [16, 32, 64]
      cond_channels: 1
      dropout: 0.1
  ```
Adjust `conv_channels` to control depth/width; increase `cond_channels` if you want richer conditioning maps.

## Usage
```bash
python main.py fit --config config/base.yaml --config config/models/cnn.yaml  # or inline overrides
```
Ensure your dataset’s `condition_dim` matches the conditioning vector shape expected by `ConditionInjector`.

## Sampling Notes
- `decode(z, cond)` expects the same conditioning dimensionality as training; `sample()` will generate random/one-hot conditions if none are provided.
- Outputs remain in `[0,1]`; the Lightning module rescales to `[-1,1]` before logging/saving.

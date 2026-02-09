# Dual-Path Hierarchical Transformer CVAE

This variant splits decoding into two paths:
- Global path from condition embedding only
- Local path from latent code (repeat-z tokens + transformer blocks)

Final reconstruction:
`output = sigmoid(global_scale * (1 - w) * global + w * local)`,
where `w` is a learnable local-path mix weight with a configurable minimum floor.

## Encoder
- Linear projection to `d_model`
- Encoder block x6:
  - RMSNorm -> Self-Attention (8 heads) -> Residual
  - RMSNorm -> Cross-Attention with shared condition embedding -> Residual
  - RMSNorm -> FFN (GELU, 4x expansion) -> Residual
- Mean pool over sequence
- Linear heads to `mu` and `logvar`

## Decoder
- Global condition path: FFN (GELU) + dropout -> linear to full spectrum
- Local latent path:
  - latent projection to `d_model`
  - repeat across sequence
  - add sinusoidal positional encoding
  - Decoder block x2:
    - RMSNorm -> Self-Attention (8 heads) -> Residual
    - Gated FiLM condition injection (sigmoid gate, init 0.1)
    - RMSNorm -> FFN (GELU, 4x) -> Residual
  - Linear projection to spectrum
- Fuse global/local with a learnable sigmoid-constrained local mix weight
- Warm up decoder by holding global-path scale at 0, then linearly ramping to 1

## Config
Enable with:
`--config config/models/dual_path_transformer.yaml`

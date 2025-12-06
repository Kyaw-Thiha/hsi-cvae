# Conformer CVAE

The Conformer variant augments the Transformer-based CVAE with Macaron feed-forward blocks, depthwise convolution modules, and optional relative positional attention. It captures local spectral patterns via depthwise convolutions while still modeling long-range dependencies through self-attention. The architecture draws heavily from the original Conformer paper ([Gulati et al., 2020](https://arxiv.org/pdf/2005.08100)).

## Encoder
- Input projection + conditioning broadcast identical to the transformer model, followed by sinusoidal positional encoding.
- Each layer follows the Conformer sandwich: half-step FFN → (relative) multi-head self-attention → depthwise conv module → half-step FFN → layer norm.
- A learned CLS token is prepended; its final state feeds the `mu/logvar` heads, while the per-wavelength token states are retained for the decoder.

## Decoder & Hybrid Memory
The decoder mirrors the encoder blocks but operates as a Transformer decoder: it starts from a learned query template, performs self-attention, cross-attends to a memory sequence, then applies the convolution module and FFN. Memory comes from:
- **Reconstruction**: encoder token states (positional encoding reapplied) so wavelength-specific context is available.
- **Sampling/predict**: `_latent_to_memory(z, cond)` projects the latent+condition vector to `(batch, seq_len, d_model)` and adds positional encoding so cross-attention remains meaningful without encoder input.

```python
queries = pos_enc(torch.zeros(batch, seq_len, d_model))
if encoder_memory is None:
    memory = latent_to_memory(z, cond)
else:
    memory = memory_pos_enc(encoder_memory)
for layer in conformer_layers:
    queries = layer(queries, memory=memory)
recon = sigmoid(output_head(queries))
```

## Positional Encoding & Conditioning
- Absolute sinusoidal positional encoding is used for queries/memory by default, while the attention block can switch between absolute (`nn.MultiheadAttention`) and relative bias via `use_relative_pos`.
- Conditioning features are injected by adding a learned projection of the condition vector to every timestep, just like the transformer model.

## Default Hyperparameters
Defined in `model.py` when `architecture: conformer`:
- `d_model = 256`
- `n_heads = 4`
- `n_layers = 4`
- `dropout = 0.1`
- `ffn_expansion = 4`
- `conv_kernel_size = 17`
- `use_relative_pos = True`

Adjust these through `config/models/conformer.yaml` or via CLI overrides.

## Configuration & Usage
Enable the conformer CVAE with:
```bash
python main.py fit --config config/base.yaml --config config/models/conformer.yaml
```
The config exposes all conformer-specific parameters (`d_model`, `n_heads`, `n_layers`, `conv_kernel_size`, `ffn_expansion`, `use_relative_pos`). Customize there or stack additional configs for experiments.

## Sampling Notes
- `decode(z, cond)` without `encoder_memory` synthesizes decoder memory from latent+condition, mirroring the transformer hybrid flow.
- `forward(...)` / `test_step` pass the encoder memory through, so reconstructions benefit from real per-wavelength context.
- Decoder outputs stay in `[0,1]` thanks to `sigmoid`; higher-level modules rescale to `[-1,1]` when needed.

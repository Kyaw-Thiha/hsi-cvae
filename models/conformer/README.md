# Conformer CVAE

The Conformer variant augments the Transformer-based CVAE with Macaron feed-forward blocks, depthwise convolution modules, and optional relative positional attention. It captures local spectral patterns via depthwise convolutions while still modeling long-range dependencies through self-attention. The architecture draws heavily from the original Conformer paper ([Gulati et al., 2020](https://arxiv.org/pdf/2005.08100)).

## Encoder
- Input projection + conditioning broadcast identical to the transformer model, followed by sinusoidal positional encoding.
- Each layer follows the Conformer sandwich: half-step FFN → (relative) multi-head self-attention → depthwise conv module → half-step FFN → layer norm.
- A learned CLS token is prepended; its final state feeds the `mu/logvar` heads, while the per-wavelength token states are retained for the decoder.

## Decoder & Hybrid Memory
The decoder mirrors the encoder blocks but operates as a Transformer decoder: it starts from a learned query template, performs self-attention, cross-attends to a memory sequence, then applies the convolution module and FFN. During training it blends encoder token states with a latent-derived memory using an adaptive gate plus an annealed scalar so gradients flow through both paths. At sampling time the encoder branch disappears, leaving only the latent memory that the schedule has trained up.

```python
queries = pos_enc(torch.zeros(batch, seq_len, d_model))
latent_mem = latent_to_memory(z, cond)
if encoder_memory is None:
    memory = latent_mem
else:
    enc_mem = memory_pos_enc(encoder_memory)
    adaptive = sigmoid(fusion_gate([z, cond]))  # (B, 1, 1)
    weight = schedule + (1 - schedule) * adaptive  # schedule from config
    memory = weight * latent_mem + (1 - weight) * enc_mem
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
The config exposes all conformer-specific parameters (`d_model`, `n_heads`, `n_layers`, `conv_kernel_size`, `ffn_expansion`, `use_relative_pos`) plus the optional `latent_gate` block (`start`, `end`, `warmup_steps`) that controls how quickly the decoder fades encoder memory out in favor of the latent sequence. Customize there or stack additional configs for experiments.

## Sampling Notes
- `decode(z, cond)` without `encoder_memory` synthesizes decoder memory from latent+condition, mirroring the transformer hybrid flow.
- `forward(...)` / `test_step` pass the encoder memory through, so reconstructions benefit from real per-wavelength context.
- Decoder outputs stay in `[0,1]` thanks to `sigmoid`; higher-level modules rescale to `[-1,1]` when needed.

# Transformer CVAE

This architecture wraps a conditional variational autoencoder with Transformer encoder/decoder stacks so it can model long-range spectral structure while conditioning on geophysical fractions.

## Encoder
- Projects each scalar wavelength to a `d_model`-dim vector via a linear layer and adds sinusoidal absolute positional encodings.
- Broadcasts the conditioning vector across the sequence (optional linear projection) so every token sees the conditioning signal.
- Prepends a learned CLS token; after the encoder stack the CLS output feeds the `mu/logvar` heads while the remaining token states are retained for the decoder.

## Decoder & Hybrid Memory
The decoder starts from a learned query template (zero tensor + positional encoding) and applies Transformer decoder layers with cross-attention. During training/reconstruction it mixes encoder token states with a latent-derived memory using an adaptive gate plus a scheduled blend so gradients flow through both paths. During sampling the encoder path is absent, so only the latent memory remains—matching the behavior the schedule anneals toward late in training.

```python
queries = pos_enc(torch.zeros(batch, seq_len, d_model))
if encoder_memory is None:
    memory = latent_to_memory(z, cond)  # project to (B, L, d_model) + pos enc
else:
    latent_mem = latent_to_memory(z, cond)
    enc_mem = memory_pos_enc(encoder_memory)
    adaptive = sigmoid(fusion_gate([z, cond]))  # (B, 1, 1)
    schedule = latent_blend  # annealed scalar from config
    weight = schedule + (1 - schedule) * adaptive
    memory = weight * latent_mem + (1 - weight) * enc_mem
output = decoder(queries, memory)
recon = sigmoid(output_head(output))
```
This hybrid setup lets reconstructions benefit from real encoder context while keeping unconditional sampling cheap.

## Positional Encoding & Conditioning
- Absolute sinusoidal encoding (`models/transformer/cvae.py:8-24`) for both encoder and decoder queries.
- Conditioning vectors are added to every token through a learned projection (acts like FiLM for spectra).

## Default Hyperparameters
Defined in `model.py` when `architecture: transformer`:
- `d_model = 128`
- `n_heads = 4`
- `n_layers = 4`
- `dropout = 0.1`
Override via CLI config (next section) if you need deeper/wider stacks.

## Configuration & Usage
Enable the transformer CVAE by chaining the config:
```bash
python main.py fit --config config/models/transformer.yaml
```
The model-specific config (`config/models/transformer.yaml`) exposes `d_model`, `n_heads`, `n_layers`, `dropout`, plus the optional `latent_gate` block (`start`, `end`, `warmup_steps`) that controls how quickly the decoder fades encoder memory out in favor of the latent projection. Adjust these values or copy the file if you need multiple presets.

## Sampling Notes
- `CVAELightningModule.sample` calls `self.model.decode(z, cond)` without encoder memory, so the decoder uses the synthesized latent memory. That guarantees outputs even when no input spectrum is provided.
- `predict_step` behaves the same; to get reconstructions faithful to a given spectrum, run `forward`/`test_step` so encoder memory is available.
- The decoder outputs `sigmoid`-squashed spectra in `[0,1]`; training rescales to `[-1,1]` elsewhere, so keep that in mind if you manually call `decode`.

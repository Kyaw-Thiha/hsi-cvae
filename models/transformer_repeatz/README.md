# Transformer Repeat-Z CVAE

This variant uses a custom Transformer encoder/decoder stack with:
- Encoder: 6 blocks
- Decoder: 4 blocks
- Block order: RMSNorm -> MHA -> Dropout -> Residual -> RMSNorm -> FFN(4x, GELU) -> Dropout -> Residual
- Encoder-to-latent via CLS token
- Latent-to-decoder by repeating latent tokens across sequence positions
- Wavelength-aware positional encoding from wavelength values
- FiLM conditioning in every block (attention path + FFN path)

Scale-VAE behavior is reused from `model.py` (no duplicate Scale-VAE logic inside this model).

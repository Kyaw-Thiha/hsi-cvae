# Scale-VAE in This Project
NOTE: The final didn't end up using `Scale-VAE`` and instead just used `Beta-VAE`.
So, the optimal hyperparams for `Scale-VAE` were not found.

## Background (from the paper)

Variational Autoencoders (VAEs) can suffer from **posterior collapse**: the decoder learns to reconstruct without using latent variables, so `z` carries little information.

Scale-VAE addresses this by making latent means more separable across samples while keeping the KL regularization on the original posterior.

### Core definitions

Given encoder output:
- `q_phi(z|x) = N(mu_x, sigma_x^2)`

For each latent dimension `d`, compute a scale factor:

```text
f_d = des_std / Std(mu_{X,d})
```

where:
- `Std(mu_{X,d})` is the std of encoder means over data (practically, over mini-batch / epoch estimate)
- `des_std` is a target spread hyperparameter

Build scaled posterior used for reconstruction sampling:

```text
q_hat_phi(z|x) = N(mu_hat_x, sigma_x^2)
mu_hat_{x,d} = f_d * mu_{x,d}
```

Objective:

```text
L_scale = E_{q_hat_phi(z|x)}[log p_theta(x|z)] - KL(q_phi(z|x) || p(z))
```

Key point: only the reconstruction path uses scaled means; KL still uses original `mu/logvar`.

### Mini-batch stability from the paper

Because `f` from one mini-batch can be noisy, the paper uses:
- First `f_epo` epochs: use each batch's own `f`
- Later epochs: use epoch-average `f_bar` (average of all batch factors)

### Inference in the paper

Sample `z ~ N(0, I)`, then scale before decoding:

```text
z_scaled = f * z
```

This keeps generation behavior aligned with training.

---

## How We Integrated It Here

## Files changed

- `model.py`
- `models/losses.py`
- `config/losses/scale_vae.yaml`

## What happens in training (`model.py`)

When `loss_name: scale_vae`:

1. Encode as usual to get `mu`, `logvar` (and memory for transformer/conformer).
2. Compute per-dim scaling factor from batch mean spread:
   - `f_batch = des_std / std(mu, dim=0)` with `scale_eps` as denominator floor for stability.
3. Choose factor by epoch rule:
   - if `current_epoch <= f_epo`: use `f_batch`
   - else: use stored epoch-average `scale_f_bar`
4. Sample reconstruction latent with scaled mean:
   - `z = (mu * f_used) + eps * exp(0.5 * logvar)`
5. Decode from `z`.
6. Compute loss with:
   - reconstruction from scaled sample
   - KL from original `mu/logvar`
   - objective form: `recon + KL` (no beta weighting, no gradient auxiliary term for Scale-VAE mode)

Epoch hooks:
- `on_train_epoch_start`: reset accumulators
- `on_train_epoch_end`: update `scale_f_bar` from average batch factors

## What happens in inference / sampling (`model.py`)

For `predict_step` and `sample`:
- Draw `z ~ N(0, I)`
- Apply learned scale: `z <- z * scale_f_bar`
- Decode as usual

So generation is consistent with Scale-VAE training.
For validation/test in Scale-VAE mode, `scale_f_bar` is also used (deterministic, aligned with generation).

## Loss entry (`models/losses.py`)

Added `scale_vae_loss` in `LOSS_REGISTRY`.
It computes `recon + KL` directly.  
Scale-VAE's distinct behavior still comes from latent-mean scaling in `model.py` before decoding.

---

## Config and usage

Use:
- `config/losses/scale_vae.yaml`

Current defaults:
- `des_std: 0.5`
- `f_epo: 100`
- `scale_eps: 1e-6`
- `recon: mse`

Note: `scale_min`, `scale_max`, `beta`, and gradient-loss knobs are not part of paper-faithful Scale-VAE mode here.

Example:

```bash
python main.py fit \
  --config config/models/mlp.yaml \
  --config config/losses/scale_vae.yaml
```

For restricted environments (no multiprocessing workers):

```bash
python main.py fit \
  --config config/models/mlp.yaml \
  --config config/losses/scale_vae.yaml \
  --data.num_workers=0
```

Switch losses by config (kept fully config-driven):

```bash
# Vanilla VAE
python main.py fit --config config/models/mlp.yaml --config config/losses/vanilla.yaml

# Beta-VAE
python main.py fit --config config/models/mlp.yaml --config config/losses/beta_vae.yaml

# Scale-VAE (paper-faithful in this repo)
python main.py fit --config config/models/mlp.yaml --config config/losses/scale_vae.yaml
```

---

## Notes

- Formulas are written in plain Markdown-friendly text (no LaTeX required).
- `des_std` strongly affects latent spread; tune it with validation metrics.
- Very small batch sizes can make factor estimates noisy; `f_epo` and epoch averaging help stabilize this.

# Dual-Path Transformer CVAE

## Intro
> [TBD by author]

---

## I/O and Notation
- Input spectrum: $x \in \mathbb{R}^{B \times L}$
- Condition vector: $c \in \mathbb{R}^{B \times C}$
- Latent sample: $z \in \mathbb{R}^{B \times d_z}$
- Reconstruction: $\hat{x} \in \mathbb{R}^{B \times L}$
- $B$: batch size, $L$: spectral length (e.g., 210), $C$: condition dimension (e.g., 3)

---

## Architecture Details

### 1. Condition Embedding (shared)
A shared MLP embeds class/composition condition:

$$
e_c = \mathrm{MLP}(c) \in \mathbb{R}^{B \times d_{\text{model}}}
$$

Used by:
- Encoder cross-attention
- Decoder FiLM
- Decoder global path

### 2. Encoder
1. Per-wavelength tokenization by linear projection: $x \rightarrow \mathbb{R}^{B \times L \times d_{\text{model}}}$
2. Encoder Transformer Block $\times N_e$
3. Mean pooling over sequence dimension
4. Two heads to Gaussian posterior:
   - $\mu = W_\mu h$
   - $\log \sigma^2 = W_{\logvar} h$

#### Encoder Transformer Block
1. RMSNorm
2. Multi-head self-attention
3. Residual
4. RMSNorm
5. Cross-attention (queries from token sequence, keys/values from condition token)
6. Residual
7. RMSNorm
8. FFN (GELU, 4x expansion)
9. Residual

### 3. Latent Sampling
Reparameterization:

$$
z = \mu + \epsilon \odot \exp(0.5 \log\sigma^2), \quad \epsilon \sim \mathcal{N}(0, I)
$$

### 4. Decoder (Dual Path)

#### 4.1 Global Path (condition-driven)

$$
g = \mathrm{Linear}(\mathrm{Dropout}(\mathrm{GELU}(\mathrm{Linear}(e_c)))) \in \mathbb{R}^{B \times L}
$$

#### 4.2 Local Path (latent-driven)
1. Project latent to model dim: $z \rightarrow \mathbb{R}^{B \times d_{\text{model}}}$
2. Repeat across sequence: $\mathbb{R}^{B \times L \times d_{\text{model}}}$
3. Add sinusoidal positional encoding
4. Decoder Transformer Block $\times N_d$
5. Linear projection to local spectrum:

$$
\ell \in \mathbb{R}^{B \times L}
$$

#### Decoder Transformer Block
1. RMSNorm
2. Multi-head self-attention
3. Residual
4. Gated FiLM condition injection
5. RMSNorm
6. FFN (GELU, 4x expansion)
7. Residual

#### Gated FiLM
- $\gamma = W_\gamma e_c$, $\beta = W_\beta e_c$
- Learnable scalar gate $s=\sigma(\alpha)$
- Modulation:

$$
u' = u \odot (1 + s\gamma) + s\beta
$$

### 5. Fusion and Output
Local-path mix weight $w$ is learned (sigmoid-constrained with configurable minimum floor).  
Global path has warmup scale $s_g$ (hold then ramp during training).

$$
f = s_g(1-w)g + w\ell
$$

$$
\hat{x} = \sigma(\kappa f)
$$

where $\kappa$ is decoder logit gain.

---

## Forward Pass (Step-by-Step)
1. Normalize condition vector (row-wise sum-to-1 in training module).
2. Apply condition dropout during training (drop whole condition vectors with Bernoulli mask).
3. Encode $(x, c)\rightarrow (\mu,\logvar)$.
4. Sample $z$ via reparameterization.
5. Decode $(z,c)$ through dual-path decoder.
6. Compute loss (beta-VAE + optional gradient terms + masking + free bits).

---

## Loss and Training Objective

### Beta-VAE objective used

$$
\mathcal{L} = \mathcal{L}_{\text{recon}} + \lambda_{\text{grad}}\mathcal{L}_{\text{grad}} + \beta\,\mathcal{L}_{\text{KL-freebits}}
$$

- Reconstruction metric: MSE (or L1 by config)
- KL with free bits per latent dimension:

$$
\mathcal{L}_{\text{KL-freebits}} = \sum_j \max(\mathrm{KL}_j - \tau, 0),\quad \tau=\frac{\text{free\_bits\_total}}{d_z}
$$

- Optional gradient matching loss on spectral finite differences (orders configurable)
- Optional wavelength weighting mask (e.g., reduced weight for water absorption bands)

---

## Hyperparams Details

### A. Core Architecture (best-performing K-lat12 family)
- `latent_dim = 12`
- `d_model = 128`
- `n_heads = 8`
- `encoder_layers = 6`
- `decoder_layers = 2`
- `dropout = 0.0`

### B. Conditioning / anti-bypass controls
- `condition_dropout = 0.35`
- `decoder_use_film = true`
- `gated_film_init = 0.20`
- `latent_fuse_weight = 0.85` (init)
- `latent_fuse_weight_min = 0.75`
- `latent_fuse_weight_learnable = true`
- `global_path_hidden_dim = 32`
- `global_path_dropout = 0.50`
- `global_path_warmup_hold_epochs = 20`
- `global_path_warmup_ramp_epochs = 30`
- `decoder_logit_gain = 1.0`

### C. Objective / optimization
- Loss: `beta_vae`
- `recon = mse`
- `beta = 0.02`
- `free_bits_total = 2.0`
- `grad_weight`: phase-1 `0.0`, phase-2 `1.0`
- `grad_diff_orders = [1,2]`
- `grad_order_weights = [1.0,0.1]`
- KL anneal: enabled (`linear`, start 0.0, warmup ratio 0.5)
- Optimizer: Adam, `lr = 2e-4`, `weight_decay = 0.0`
- Scheduler: cosine annealing

### D. Data/training setup used in ablation
- `input_dim = 210` (400-2490 nm, step 10 nm)
- `condition_dim = 3`
- batch size: `32`
- split: `0.8 / 0.1 / 0.1`
- seed: `42`
- epoch schedules used across runs:
  - single-stage `100` epochs (A/B)
  - two-stage `40 -> 120` (C/D/C_fixed/D_fixed/F/G/H/I)
  - two-stage `40 -> 80` (J/K variants)

### E. Ablation hyperparameters actually used (A-K)
- Shared fixed-template backbone (C_fixed onward):
  - `condition_dropout=0.35`, `d_model=128`, `n_heads=8`, `encoder_layers=6`, `decoder_layers=2`, `dropout=0.0`
  - `latent_fuse_weight=0.85`, `latent_fuse_weight_min=0.75`, `latent_fuse_weight_learnable=true`
  - `global_path_hidden_dim=32`, `global_path_dropout=0.50`
  - `global_path_warmup_hold_epochs=20`, `global_path_warmup_ramp_epochs=30`
  - `decoder_logit_gain=1.0`
- A/B (single-stage, old composition path): `max_epochs=100`, `gated_film_init=0.20` (A) or `0.001` (B)
- C/D (old composition path, two-stage): phase-1 `max_epochs=40`, phase-2 `max_epochs=120`; FiLM init as A/B
- C_fixed/D_fixed/F (two-stage):
  - phase-1: `grad_weight=0.0`, `free_bits_total=0.0`
  - phase-2: `grad_weight=8.0`, `free_bits_total=0.0`
  - FiLM variants: `gated_film_init=0.20` (C_fixed), `0.001` (D_fixed), `decoder_use_film=false` (F)
- G/H/I (two-stage):
  - `latent_dim=128` (G), `32` (H), `16` (I)
  - phase-1: `beta=0.02`, `free_bits_total=2.0`, `grad_weight=0.0`
  - phase-2: `beta=0.02`, `free_bits_total=2.0`, `grad_weight=2.0`
  - epoch schedule: phase-1 `40`, phase-2 `120`
- J variants (`latent_dim=16`):
  - base + phase-1 common: `beta=0.02`, `free_bits_total=2.0`, `grad_weight=0.0`, `max_epochs=40`
  - phase-2 (`max_epochs=80`) tested:
    - `grad_weight=0.5`, `beta=0.02`
    - `grad_weight=1.0`, `beta=0.02`
    - `grad_weight=1.0`, `beta=0.015`
  - all J phase-2 runs keep `free_bits_total=2.0`
- K variants (`latent_dim=12`):
  - base + phase-1 common: `beta=0.02`, `free_bits_total=2.0`, `grad_weight=0.0`, `max_epochs=40`
  - phase-2 (`max_epochs=80`) in current config set: `grad_weight=1.0`, `beta=0.02`, `free_bits_total=2.0`

For all ablation runs, loss base was `beta_vae` with `recon=mse`, KL annealing enabled (linear), gradient orders `[1,2]` with weights `[1.0, 0.1]`, and wavelength mask settings from `config/losses/beta_vae.yaml`.

---

## Design Rationale (Concise)
- Dual-path decoder separates condition-driven coarse structure (global path) from latent-driven details (local path).
- Warmup on global path and high local-path floor are used to discourage latent bypass/collapse.
- Small latent sizes (12-16) and moderate gradient regularization improved latent usage stability in ablations.

---

## Ablation-Sensitive Knobs
Most impactful knobs observed:
1. `latent_dim`
2. `condition_dropout`
3. `latent_fuse_weight` / `latent_fuse_weight_min`
4. `global_path_warmup_hold_epochs` / `global_path_warmup_ramp_epochs`
5. `beta`, `free_bits_total`
6. phase-2 `grad_weight`

---

## Limitations / Failure Modes
- Posterior collapse is mitigated, not universally solved across all settings.
- Stability across multiple random seeds still needs broader confirmation.
- Early families (A-H) frequently converged to near-zero active latent dimensions.

---

## Reproducibility Recipe
- Base model config: `config/models/dual_path_transformer.yaml`
- Base loss config: `config/losses/beta_vae.yaml`
- Best-run overrides:
  - `config/experiments/run_K_lat12_base.yaml`
  - `config/experiments/run_K_phase1_common.yaml`
  - `config/experiments/run_K_phase2_grad1_beta002.yaml`
- Ablation summary: `docs/latent_usage_ablation.md`

Canonical training:

```bash
python main.py fit \
  --config config/models/dual_path_transformer.yaml \
  --config config/losses/beta_vae.yaml \
  --config <experiment_yaml> \
  --trainer.default_root_dir <run_dir>
```

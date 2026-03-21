# Latent Usage Ablation (Dual-Path Transformer)

> This document records posterior-collapse ablation experiments conducted on the **dual-path-transformer** CVAE.

## Main Idea & Results
### TL;DR
- Goal: improve latent usage and reduce posterior collapse in dual-path-transformer.
- Approach: staged ablations across architecture knobs and training/loss hyperparameters.
- Current outcome: runs in the I/J/K family improved latent usage substantially versus A-H, with `run_K_lat12` giving the best overall tradeoff.

### Snapshot 
| Item | Status |
|---|---|
| Primary model under test | dual-path-transformer |
| Posterior collapse fully solved? | Partially (mitigated in I/J/K; not robust across all settings) |
| Best run by validation | `run_K_lat12` (`best val_loss=0.0024`) |
| Best checkpoint | `outputs/checkpoints/best/e_epoch=37-l_val_loss=0.0024.ckpt` |

---

## Background Info
- Problem observed: model often produced realistic outputs while latent KL objective collapsed toward zero.
- Hypothesis: strong condition pathway enabled decoder bypass of latent variables.
- Why this ablation started: quantify which controls (conditioning strength, warmup, loss terms, latent size) improve latent usage without hurting reconstruction.

---

## Experimental Setup
### Data / Split
- Dataset: `data/simpler_data_rwc.csv`
- Split: `0.8 / 0.1 / 0.1` (train/val/test)
- Batch size: `32`
- Seeds used: `42`

### Architecture Scope
- Architecture tested: **dual-path-transformer only**
- Shared baseline family:
- `config/models/dual_path_transformer.yaml`
- `config/losses/beta_vae.yaml`
- `config/base.yaml` data/trainer defaults
- Run-specific overrides in `config/experiments/*.yaml`

### Metrics Tracked
- `train_kl_loss`
- `train_kl_objective_loss`
- `train_active_dims`
- `train_mu_std`
- `train_recon_loss`
- `val_loss`
- `val_recon_loss`
- `val_kl_loss`
- `val_kl_objective_loss`
- `train_dual_path_local_weight`
- `train_dual_path_global_scale`

### Collapse Criteria
- Collapse risk signals:
- `train_active_dims -> 0`
- `train_kl_objective_loss -> ~0` for sustained epochs
- `mu_std` shrinking abnormally
- "Healthy latent usage" signals:
- Non-trivial active dims over late training
- Non-zero KL objective while preserving validation quality

---

## Hyperparams Being Tuned
- Conditioning / decoder bypass controls:
- `condition_dropout`
- `gated_film_init`
- `decoder_use_film`
- `latent_fuse_weight`
- `latent_fuse_weight_min`
- `global_path_hidden_dim`
- `global_path_dropout`
- `global_path_warmup_hold_epochs`
- `global_path_warmup_ramp_epochs`
- Optimization/loss controls:
- `beta`
- `free_bits_total`
- `grad_weight` (phase schedule)
- Model capacity:
- `latent_dim`
- `d_model`
- `decoder_layers`
- Effective ranges tested:
- `latent_dim`: `128, 32, 16, 12`
- `beta`: `0.05, 0.02, 0.015`
- `free_bits_total`: `0.0, 2.0`
- `grad_weight`: `0.0, 0.5, 1.0, 2.0, 8.0`
- `gated_film_init`: `0.2, 0.001`
- `decoder_use_film`: `true, false`

---

## Experiments
### Run Groups
- Group 1: A/B/C/D (initial anti-bypass + schedule variants)
- Group 2: C_fixed/D_fixed/F (corrected config composition + FiLM-off test)
- Group 3: G/H/I (beta/free-bits/grad schedule with latent size sweep)
- Group 4: J/K refinements (shorter phase-2 variants around best regime)

### Run Registry
| Run | Main change | Phase schedule | Notes |
|---|---|---|---|
| A | FiLM gate init `0.2` | single run (`max_epochs=100`) | early anti-bypass attempt |
| B | FiLM gate init `0.001` | single run (`max_epochs=100`) | FiLM suppression test |
| C | A + phase split (`40 -> 120`) | phase1/phase2 | still old composition path |
| D | B + phase split (`40 -> 120`) | phase1/phase2 | still old composition path |
| C_fixed | Correct config merge + stronger bottleneck (`d_model=128`, `decoder_layers=2`, `global_hidden=32`) | phase1 (`grad_weight=0`) + phase2 (`grad_weight=8`) | `free_bits_total=0` |
| D_fixed | C_fixed + low FiLM init (`0.001`) | phase1 (`grad_weight=0`) + phase2 (`grad_weight=8`) | `free_bits_total=0` |
| F | C_fixed but `decoder_use_film=false` | phase1 (`grad_weight=0`) + phase2 (`grad_weight=8`) | FiLM removed from decoder |
| G | C_fixed template + `latent_dim=128`, `beta=0.02`, `free_bits=2`, `grad_weight=2` in phase2 | phase1+phase2 | high latent capacity |
| H | same as G but `latent_dim=32` | phase1+phase2 | medium latent capacity |
| I | same as G but `latent_dim=16` | phase1+phase2 | low latent capacity |
| J variants | I baseline + phase2 short sweep (`grad_weight=0.5/1.0`, `beta=0.02/0.015`) | phase2 to `max_epochs=80` | latent=16 refinement |
| K variants | K base (`latent_dim=12`) + phase2 short sweep (`grad_weight=1.0`, `beta=0.02`) | phase2 to `max_epochs=80` | latent=12 refinement |

### Repro Commands
- Canonical command pattern:
```bash
python main.py fit \
  --config config/models/dual_path_transformer.yaml \
  --config config/losses/beta_vae.yaml \
  --config <experiment_yaml> \
  --trainer.default_root_dir <run_dir>
```

- Resume pattern:
```bash
python main.py fit \
  --ckpt_path <phase1_ckpt> \
  --config config/models/dual_path_transformer.yaml \
  --config config/losses/beta_vae.yaml \
  --config <phase2_yaml> \
  --trainer.default_root_dir <run_dir>
```

---

## Results
### Aggregate Results Table
| Run | Best `val_loss` | Final `val_loss` | Final `val_kl_loss` | Final `val_kl_objective_loss` | Final `train_active_dims` | Collapse verdict |
|---|---:|---:|---:|---:|---:|---|
| A | 0.0102 | 0.0102 | 1.0122 | 0.000039 | 0 | Collapsed |
| B | 0.0102 | 0.0103 | 1.0206 | 0.000057 | 0 | Collapsed |
| C | 0.0031 | 0.0103 | 0.8823 | 0.000489 | 0 | Collapsed |
| D | 0.0033 | 0.0103 | 0.8401 | 0.000502 | 0 | Collapsed |
| C_fixed | 0.0038 | 0.0115 | 0.0001 | 0.000100 | 0 | Collapsed |
| D_fixed | 0.0050 | 0.0121 | 0.0001 | 0.000094 | 0 | Collapsed |
| F | 0.0049 | 0.0121 | 0.0003 | 0.000325 | 0 | Collapsed |
| G | 0.0029 | 0.0043 | 1.0817 | 0.000209 | 0 | Collapsed |
| H | 0.0026 | 0.0044 | 1.7748 | 0.000382 | 0 | Collapsed |
| I | 0.0025 | 0.0041 | 1.8518 | 0.000921 | 16 | Mitigated |
| J_lat16 | 0.0025 | 0.0028 | 1.3639 | 0.000114 | 4 | Partial |
| J_lat16_g1_b002 | 0.0030 | 0.0034 | 1.8798 | 0.000566 | 16 | Mitigated |
| J_lat16_g05_b002 | 0.0025 | 0.0029 | 1.8587 | 0.000135 | 16 | Mitigated |
| J_lat16_g1_b0015 | 0.0030 | 0.0034 | 1.8781 | 0.000770 | 16 | Mitigated |
| K_lat12 | 0.0024 | 0.0026 | 1.4789 | 0.001195 | 10 | Mitigated (best overall) |
| K_lat12_g1_b002 | 0.0031 | 0.0031 | 1.8821 | 0.000179 | 12 | Mitigated |

### Best Checkpoints
| Run | Checkpoint path | Selection rationale |
|---|---|---|
| K_lat12 | `outputs/checkpoints/best/e_epoch=37-l_val_loss=0.0024.ckpt` | best observed `val_loss` among A-K |
| J_lat16_g05_b002 | `outputs/checkpoints/best/e_epoch=73-l_val_loss=0.0025.ckpt` | strong latent usage (`active_dims=16`) with near-best validation |

### Key Curves to Inspect
- TensorBoard run comparison under `outputs/ablations/*/lightning_logs`:
- `train_active_dims`
- `train_kl_objective_loss` and `val_kl_objective_loss`
- `val_loss` and `val_recon_loss`

---

## Analysis
### What Worked
- Reducing latent size from `128 -> 16/12` helped prevent full latent shutoff.
- Lowering phase-2 gradient loss strength (`grad_weight=0.5 or 1.0`) was more stable than large values (`8.0`).
- `run_K_lat12` produced the best balance: lowest `val_loss` and non-trivial late-training active dims.

### What Failed
- FiLM suppression alone (`gated_film_init=0.001`) did not prevent collapse.
- Turning decoder FiLM off entirely (`run_F_film_off`) did not solve collapse by itself.
- Large latent capacity with strong conditioning path (A-H, especially latent 128/32) frequently ended with `train_active_dims=0`.
- Very high phase-2 `grad_weight` correlated with brittle behavior and weaker latent retention.

### Posterior Collapse Verdict
- Final verdict: **Partially solved / strongly mitigated, not globally solved.**
- Evidence:
- `train_active_dims`: A-H mostly finished at `0`; I/J/K successful variants finished at `10-16`.
- `train_kl_objective_loss`: successful variants stayed non-zero (for example `run_K_lat12`: `0.001150`).
- `val_kl_objective_loss`: non-zero in successful variants (`run_K_lat12`: `0.001195`; J variants: `1e-4` to `7e-4`).
- Caveats:
- Results are from single-seed runs; stability across seeds is not yet confirmed.
- Some earlier runs (A-D) used an older composition path and are not directly comparable to fixed-template runs.

---

## Next Steps
- Decision:
- Use `run_K_lat12` as the current working recipe and treat this ablation as complete for first-pass posterior-collapse mitigation.

- If proceeding:
- Run multi-seed stability on `run_K_lat12` (at least 3 seeds).
- Add early stopping around the observed best epoch window (`~35-40` for latent-12 recipe).
- Freeze one production config from K (and keep J_g05 as backup).
- Evaluate downstream sample quality/diversity vs. prior baseline.

- If done:
- Mark this ablation as closed in README and link final config + checkpoint.

---

## Appendix
### Config Files Used
- `config/experiments/pc_base_shared.yaml`
- `config/experiments/film_strong.yaml`
- `config/experiments/film_weak.yaml`
- `config/experiments/grad_off_phase1.yaml`
- `config/experiments/grad_on_phase2.yaml`
- `config/experiments/run_C_fixed_phase1.yaml`
- `config/experiments/run_C_fixed_phase2.yaml`
- `config/experiments/run_D_fixed_phase1.yaml`
- `config/experiments/run_D_fixed_phase2.yaml`
- `config/experiments/run_F_film_off_phase1.yaml`
- `config/experiments/run_F_film_off_phase2.yaml`
- `config/experiments/run_G_lat128_phase1.yaml`
- `config/experiments/run_G_lat128_phase2.yaml`
- `config/experiments/run_H_lat32_phase1.yaml`
- `config/experiments/run_H_lat32_phase2.yaml`
- `config/experiments/run_I_lat16_phase1.yaml`
- `config/experiments/run_I_lat16_phase2.yaml`
- `config/experiments/run_J_lat16_base.yaml`
- `config/experiments/run_J_phase1_common.yaml`
- `config/experiments/run_J_phase2_grad05_beta002.yaml`
- `config/experiments/run_J_phase2_grad1_beta002.yaml`
- `config/experiments/run_J_phase2_grad1_beta0015.yaml`
- `config/experiments/run_K_lat12_base.yaml`
- `config/experiments/run_K_phase1_common.yaml`
- `config/experiments/run_K_phase2_grad1_beta002.yaml`
- `config/models/dual_path_transformer.yaml`
- `config/losses/beta_vae.yaml`

### Run Directories
- `outputs/ablations/run_A`
- `outputs/ablations/run_B`
- `outputs/ablations/run_C`
- `outputs/ablations/run_D`
- `outputs/ablations/run_C_fixed`
- `outputs/ablations/run_D_fixed`
- `outputs/ablations/run_F_film_off`
- `outputs/ablations/run_G_lat128`
- `outputs/ablations/run_H_lat32`
- `outputs/ablations/run_I_lat16`
- `outputs/ablations/run_J_lat16`
- `outputs/ablations/run_J_lat16_g1_b002`
- `outputs/ablations/run_J_lat16_g05_b002`
- `outputs/ablations/run_J_lat16_g1_b0015`
- `outputs/ablations/run_K_lat12`
- `outputs/ablations/run_K_lat12_g1_b002`

### Notes
- Metrics were extracted from TensorBoard event files under each run's `lightning_logs`.
- For mixed phase runs, reported values are taken from the final available scalar points.

# tcVAE vs Diffusion Artifact Differences

Date: 2026-03-21

Scope:
- `FINCH-Science_SyntheticData/synthesis/isprs/diffusion/*`
- `FINCH-Science_SyntheticData/synthesis/isprs/tcvae/*`
- upstream generators in this repo outside `FINCH-Science_SyntheticData`

## Compared artifact files

Diffusion:
- `FINCH-Science_SyntheticData/synthesis/isprs/diffusion/psi1_gdstreamline.csv`
- `FINCH-Science_SyntheticData/synthesis/isprs/diffusion/psi2_gdstreamline.csv`
- `FINCH-Science_SyntheticData/synthesis/isprs/diffusion/cfg_diffusion_setup.json`
- `FINCH-Science_SyntheticData/synthesis/isprs/diffusion/synthesis_cfg.yaml`

tcVAE:
- `FINCH-Science_SyntheticData/synthesis/isprs/tcvae/psi1_train_normalized.csv`
- `FINCH-Science_SyntheticData/synthesis/isprs/tcvae/psi2_test_val_normalized.csv`
- `FINCH-Science_SyntheticData/synthesis/isprs/tcvae/psi1_train_conditions_normalized.csv`
- `FINCH-Science_SyntheticData/synthesis/isprs/tcvae/psi2_test_val_conditions_normalized.csv`
- `FINCH-Science_SyntheticData/synthesis/isprs/tcvae/psi1_train_spectra_normalized.csv`
- `FINCH-Science_SyntheticData/synthesis/isprs/tcvae/psi2_test_val_spectra_normalized.csv`
- `FINCH-Science_SyntheticData/synthesis/isprs/tcvae/artifact_manifest.json`
- `FINCH-Science_SyntheticData/synthesis/isprs/tcvae/cfg_tcvae_setup.json`
- `FINCH-Science_SyntheticData/synthesis/isprs/tcvae/synthesis_cfg.yaml`

## Main artifact differences

### 1. Diffusion `psi` CSVs are row-labeled, tcVAE `psi` CSVs are not

Diffusion `psi1` and `psi2` begin with:

- `orig_index`
- `Spectra`
- `gv_fraction`
- `npv_fraction`
- `soil_fraction`
- wavelength columns

tcVAE `psi1` and `psi2` begin directly with:

- `gv_fraction`
- `npv_fraction`
- `soil_fraction`
- wavelength columns

Impact:
- tcVAE artifacts do not preserve row identity.
- A reader cannot reconcile a row back to the source dataset by using the artifact file alone.

This is the specific issue raised in feedback and it is valid.

### 2. tcVAE auxiliary CSVs are also unlabeled

The following tcVAE files contain no per-row identifier columns:

- `psi1_train_conditions_normalized.csv`
- `psi2_test_val_conditions_normalized.csv`
- `psi1_train_spectra_normalized.csv`
- `psi2_test_val_spectra_normalized.csv`

Impact:
- even the split-out condition-only and spectra-only exports are not self-reconcilable.

### 3. Column counts differ

Observed:

- diffusion `psi1_gdstreamline.csv`: 1500 rows, 215 columns
- diffusion `psi2_gdstreamline.csv`: 223 rows, 215 columns
- tcVAE `psi1_train_normalized.csv`: 1378 rows, 213 columns
- tcVAE `psi2_test_val_normalized.csv`: 345 rows, 213 columns

The 2-column difference is exactly the missing:

- `orig_index`
- `Spectra`

### 4. File naming differs

Diffusion publishes:

- `psi1_gdstreamline.csv`
- `psi2_gdstreamline.csv`

tcVAE publishes:

- `psi1_train_normalized.csv`
- `psi2_test_val_normalized.csv`

And tcVAE adds extra files:

- `artifact_manifest.json`
- condition-only CSVs
- spectra-only CSVs

Impact:
- tcVAE does not look like the same artifact family as diffusion.

### 5. Split definitions differ

Diffusion artifacts:

- `psi1`: synthesized training artifact
- `psi2`: held-out artifact

tcVAE artifacts:

- `psi1_train_normalized.csv`: train split
- `psi2_test_val_normalized.csv`: test + val concatenated

Observed sizes:

- diffusion `psi1`: 1500
- diffusion `psi2`: 223
- tcVAE `psi1`: 1378
- tcVAE `psi2`: 345

Impact:
- these are not only format differences; the split composition is also different.

### 6. Value scaling differs

Observed numeric ranges:

- diffusion `psi1`: min about `-2.0188365`, max about `6.0864725`
- diffusion `psi2`: min about `-1.9365187`, max about `6.003995`
- tcVAE `psi1`: min `0.0`, max `1.0`
- tcVAE `psi2`: min `0.0`, max `1.0`

Impact:
- diffusion artifacts are not represented in the same numeric domain as tcVAE artifacts.
- even with identical schema, the files would still not be directly comparable.

### 7. tcVAE condition rows are explicitly renormalized on export

The exporter normalizes each condition row to sum to 1 before writing:

- `callbacks/data_artifacts.py`

Impact:
- source condition values can shift slightly due to normalization / floating-point formatting.
- example observed from source vs tcVAE export:
  - source: `0.9621`, `0.038`
  - export: `0.96200383`, `0.03799620`

### 8. tcVAE artifacts are not independently reconcilable

tcVAE row identity can only be reconstructed by combining:

- the original dataset CSV
- the split index file
- the knowledge of train/test/val concatenation order

Relevant files:

- `data/simpler_data_rwc.csv`
- `outputs/checkpoints/data_42.json`

Impact:
- the artifacts alone are insufficient for future readers.
- this is the core usability issue.

## Upstream generator findings

### tcVAE source data does contain labels

`data/simpler_data_rwc.csv` includes:

- `Spectra`
- condition columns
- wavelength columns

It does not include `orig_index` as a stored CSV column, but row index is available from row position and from the saved split indices.

### tcVAE split indices are available

`outputs/checkpoints/data_42.json` stores:

- `train_indices`
- `val_indices`
- `test_indices`

This means:

- `orig_index` can be exported
- `Spectra` can be exported
- row mapping is available at artifact-generation time

### The tcVAE exporter is where labels are dropped

The current exporter is:

- `callbacks/data_artifacts.py`

What it does now:

- extracts numeric conditions from `dataset.conditions`
- extracts normalized spectra from `dataset.spectra`
- writes plain numeric matrices with `np.savetxt`

What it does not do:

- prepend `orig_index`
- prepend `Spectra`
- write any row mapping sidecar

The key write sites are the exports of:

- `psi1_train_normalized.csv`
- `psi2_test_val_normalized.csv`
- condition-only CSVs
- spectra-only CSVs

### Dataset loader currently does not preserve metadata for artifact export

The dataset loader:

- reads the full CSV
- extracts condition columns
- infers spectral columns
- stores normalized spectra tensor and condition tensor

But it does not expose artifact-ready metadata columns such as:

- `Spectra`
- original row index column / source row index array

Relevant file:

- `dataset.py`

## Config/setup artifact inconsistencies

### 1. tcVAE synthesis config points to diffusion-style setup filename

In:

- `FINCH-Science_SyntheticData/synthesis/isprs/tcvae/synthesis_cfg.yaml`

Observed:

- `architecture: CCVAEncoder`
- `cfg_model_setup_path: 'cfg_diffusion_setup.json'`

But the tcVAE folder actually contains:

- `cfg_tcvae_setup.json`

Impact:
- tcVAE artifact packaging is internally inconsistent.

### 2. Exported setup JSON structure name differs from checked-in tcVAE artifact

Current exporter code writes:

- top-level `cfg_cvae`

Checked-in tcVAE artifact uses:

- top-level `cfg_tcvae`

Relevant files:

- `callbacks/cvae_setup_json.py`
- `scripts/export_cvae_setup_json.py`
- `FINCH-Science_SyntheticData/synthesis/isprs/tcvae/cfg_tcvae_setup.json`

Impact:
- the generated setup artifact family is not aligned with the checked-in tcVAE artifact naming.

## Specific evidence for the missing-label problem

The first few rows of tcVAE `psi1_train_normalized.csv` match the train indices and source rows from:

- `outputs/checkpoints/data_42.json`
- `data/simpler_data_rwc.csv`

Example:

- first exported tcVAE row corresponds to source row index `1667`
- source `Spectra` value is `SSC_BAPI08_20111107`

But the tcVAE artifact row only contains:

- condition values
- wavelength values

Therefore the label exists upstream and is simply discarded during export.

## Practical change list needed to make tcVAE artifacts much closer to diffusion

### Must-have

- prepend `orig_index` to tcVAE `psi1` and `psi2`
- prepend `Spectra` to tcVAE `psi1` and `psi2`
- preserve the same row order currently used for splits
- ideally prepend the same identifier columns to condition-only and spectra-only auxiliary CSVs too

### Strongly recommended

- rename tcVAE main exports toward diffusion-style names, or add diffusion-style aliases
- add row-mapping fields to `artifact_manifest.json`
- align tcVAE setup/config artifact naming so it no longer references diffusion file names

### Separate decision

- decide whether tcVAE artifacts should remain normalized in `[0,1]` or be exported in the same value domain expected by the diffusion/testing pipeline

This is not just formatting. It affects compatibility.

## Notes

- `FINCH-Science_SyntheticData/synthesis/isprs/diffusion/generated_gdstreamline_3000.csv`
- `FINCH-Science_SyntheticData/synthesis/isprs/diffusion/generated_gdstreamline_9000.csv`

These are Git LFS pointer files in the current workspace, so they were not useful for direct content comparison here.

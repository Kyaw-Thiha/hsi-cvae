# tcVAE to Diffusion-Compatible Artifact Checklist

Date: 2026-03-21

Goal:
- make tcVAE artifacts match the diffusion folder format closely enough that downstream integration is straightforward
- preserve row identity and make artifacts self-reconcilable

Related notes:
- `docs/tmp_tcvae_vs_diffusion_artifact_diffs.md`

## Phase 1: Lock the artifact contract

Define the tcVAE integration artifact schema to match diffusion exactly:

- `orig_index`
- `Spectra`
- `gv_fraction`
- `npv_fraction`
- `soil_fraction`
- wavelength columns `400` to `2490`

Define exact column order as identical to diffusion.

Define row-order rules:

- `psi1` uses train indices in saved split order
- `psi2` uses `test` followed by `val`, unless intentionally changed

Decide filename policy:

- Option A: emit diffusion-style names directly
- Option B: emit current tcVAE names plus diffusion-compatible aliases

Decide value-domain policy:

- keep integration files normalized to `[0,1]`
- or convert integration files into the same numeric domain used by diffusion/testing

## Phase 2: Preserve metadata in the dataset layer

Files:

- `dataset.py`
- `data_module.py`

Tasks:

- expose source row indices from the original CSV
- expose `Spectra` values from the original CSV
- keep these accessible on the dataset object so callbacks can export them
- optionally expose other source metadata if useful later, but do not add extra integration columns unless needed

## Phase 3: Rewrite artifact export to produce diffusion-compatible main files

File:

- `callbacks/data_artifacts.py`

Tasks:

- build labeled export rows using:
  - split indices
  - source row index as `orig_index`
  - source `Spectra`
  - condition columns
  - spectra columns
- replace the current plain matrix export for the main `psi` files with a labeled tabular export
- ensure `psi1` and `psi2` main files are self-contained and reconcilable
- keep deterministic formatting for numeric columns
- keep `orig_index` integer-like and `Spectra` string-like

## Phase 4: Decide what to do with auxiliary files

File:

- `callbacks/data_artifacts.py`

Recommended:

- keep auxiliary files, but label them too

For condition-only and spectra-only CSVs, prepend:

- `orig_index`
- `Spectra`

Tasks:

- update:
  - `psi1_train_conditions_normalized.csv`
  - `psi2_test_val_conditions_normalized.csv`
  - `psi1_train_spectra_normalized.csv`
  - `psi2_test_val_spectra_normalized.csv`
- ensure each auxiliary file can be joined back to the main `psi` file row-for-row

## Phase 5: Align naming and manifest metadata

Files:

- `callbacks/data_artifacts.py`
- generated `artifact_manifest.json`

Tasks:

- update manifest fields to describe:
  - labeled schema
  - exact split definitions
  - row order
  - whether values are normalized or transformed
- add fields for identifier columns:
  - `id_columns: ["orig_index", "Spectra"]`
- if using alias filenames, record both canonical and compatibility filenames

## Phase 6: Fix setup/config artifact inconsistencies

Files:

- `callbacks/cvae_setup_json.py`
- `scripts/export_cvae_setup_json.py`
- `FINCH-Science_SyntheticData/synthesis/isprs/tcvae/synthesis_cfg.yaml`

Tasks:

- decide whether top-level setup key should be `cfg_tcvae` or `cfg_cvae`
- make exporter and checked-in artifact agree
- fix `synthesis_cfg.yaml` so it references the correct tcVAE setup file, not `cfg_diffusion_setup.json`
- fix the reported architecture name if needed

## Phase 7: Add a format validator

Recommended new script:

- `scripts/validate_artifact_format.py`

Checks:

- header matches expected diffusion-compatible schema
- `orig_index` exists
- `Spectra` exists
- `orig_index` is unique within each file
- `Spectra` is non-empty
- auxiliary files align row-for-row with main files
- row counts match expected split sizes
- numeric columns are parseable
- optional: assert exact filename set exists

## Phase 8: Regenerate and compare

Tasks:

- regenerate tcVAE artifacts from one run
- compare headers against diffusion `psi1_gdstreamline.csv` and `psi2_gdstreamline.csv`
- spot-check first few rows against:
  - `data/simpler_data_rwc.csv`
  - `outputs/checkpoints/data_42.json`
- verify that a future reader can reconcile any exported row without external guesswork

## Suggested work order

1. Decide filename policy and numeric-domain policy.
2. Update dataset metadata exposure.
3. Update artifact exporter for labeled main files.
4. Label auxiliary files.
5. Fix manifest.
6. Fix setup/config naming inconsistencies.
7. Add validator.
8. Regenerate and diff-check outputs.

## Recommended implementation stance

- make the main tcVAE integration files diffusion-compatible immediately
- keep tcVAE-specific normalized auxiliaries, but label them too
- add compatibility aliases if a low-risk transition is preferred
- treat numeric-domain alignment as a separate decision after schema alignment

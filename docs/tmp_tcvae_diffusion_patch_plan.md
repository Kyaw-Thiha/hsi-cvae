# tcVAE to Diffusion-Compatible Artifact Patch Plan

Date: 2026-03-21

Goal:
- turn the checklist into a concrete file-by-file implementation plan
- make tcVAE artifacts match the diffusion artifact contract as closely as practical

Assumption for this plan:
- main integration-facing tcVAE artifacts should become diffusion-compatible in schema
- tcVAE-specific auxiliary artifacts can remain, but they should also be labeled
- value-domain alignment is called out explicitly as a decision point

## Proposed output contract

Main CSV artifacts should have this exact leading column order:

1. `orig_index`
2. `Spectra`
3. `gv_fraction`
4. `npv_fraction`
5. `soil_fraction`
6. spectral columns `400` through `2490`

Main outputs:

- `psi1_gdstreamline.csv` or compatible alias
- `psi2_gdstreamline.csv` or compatible alias

Transition-safe approach:

- keep current tcVAE filenames
- add diffusion-compatible aliases
- make both point to the same tabular content format

## File-by-file patch plan

## 1. `dataset.py`

Purpose:
- preserve source metadata needed for export

Current gap:
- dataset loads the full CSV but only stores numeric tensors for conditions and spectra
- `Spectra` and source row indices are not preserved as artifact-ready fields

Planned changes:

- add optional metadata capture during dataset initialization
- store:
  - `self.row_indices`
  - `self.spectrum_ids`
  - optionally `self.source_df_columns` if useful
- derive `row_indices` as the original row positions from the loaded CSV
- if `Spectra` exists, store its string values in `self.spectrum_ids`
- if `Spectra` does not exist, decide on a fallback such as empty strings or generated labels, but prefer failing clearly if required for artifact export

Concrete edits:

- after reading `df = pd.read_csv(...)`, add:
  - `self.row_indices = np.arange(len(df), dtype=np.int64)`
  - `self.spectrum_ids = df["Spectra"].astype(str).tolist()` if column exists
- keep this data available without affecting training behavior

Reason:
- exporter needs direct access to source identity without reconstructing from scratch

## 2. `data_module.py`

Purpose:
- expose artifact-related metadata consistently through the datamodule

Current gap:
- split indices are saved, but no artifact-specific metadata contract is exposed

Planned changes:

- add lightweight accessors or attributes so callbacks can rely on:
  - `dataset.row_indices`
  - `dataset.spectrum_ids`
  - split ordering semantics
- optionally expose:
  - `self.spectrum_id_column = "Spectra"`
  - `self.id_columns = ["orig_index", "Spectra"]`

Concrete edits:

- no structural rewrite needed
- after dataset creation, verify `Spectra` exists if labeled artifact export is enabled
- optionally add a config-driven toggle later, but default should support labeled exports

Reason:
- keep artifact callback logic simpler and less brittle

## 3. `callbacks/data_artifacts.py`

Purpose:
- this is the main patch site
- make exporter produce diffusion-compatible labeled artifacts

Current gap:
- writes only numeric matrices with `np.savetxt`
- drops `orig_index` and `Spectra`
- main `psi` outputs are not self-reconcilable

Planned changes:

### 3a. Add labeled export helpers

Replace or supplement `_write_matrix_csv(...)` with a helper that can write mixed-type rows:

- `_write_labeled_csv(path, header, rows)`

Recommended implementation:

- use Python `csv.writer` or pandas DataFrame export
- do not use `np.savetxt` for mixed string + numeric tables

### 3b. Build per-split identity arrays

Use:

- `dataset.row_indices`
- `dataset.spectrum_ids`
- `train_idx`
- `test_idx`
- `val_idx`

Build:

- `psi1_orig_index`
- `psi1_spectra`
- `psi2_orig_index`
- `psi2_spectra`

Where:

- `psi1` follows `train_idx`
- `psi2` follows `test_idx` then `val_idx`

### 3c. Build labeled main `psi` rows

For each split row, emit:

- `orig_index`
- `Spectra`
- normalized conditions
- normalized or transformed spectra

Header:

- `["orig_index", "Spectra"] + condition_columns + spectral_columns`

### 3d. Decide main filename behavior

Recommended transition approach:

- continue writing:
  - `psi1_train_normalized.csv`
  - `psi2_test_val_normalized.csv`
- additionally write diffusion-compatible aliases:
  - `psi1_gdstreamline.csv`
  - `psi2_gdstreamline.csv`

If you want harder convergence:

- stop emitting the old main names later

### 3e. Label auxiliary files too

Update auxiliary CSVs to prepend:

- `orig_index`
- `Spectra`

Affected files:

- `psi1_train_conditions_normalized.csv`
- `psi2_test_val_conditions_normalized.csv`
- `psi1_train_spectra_normalized.csv`
- `psi2_test_val_spectra_normalized.csv`

Headers become:

- conditions-only:
  - `orig_index`, `Spectra`, condition columns
- spectra-only:
  - `orig_index`, `Spectra`, spectral columns

### 3f. Keep NPY export policy explicit

NPY files can remain pure numeric arrays if desired, but note:

- they will not contain row labels
- manifest should state that clearly

Optional improvement:

- add companion metadata arrays for NPY export

### 3g. Update manifest generation

Extend manifest fields to include:

- `id_columns`
- `main_schema`
- `row_order_definition`
- whether main files are normalized or transformed
- compatibility aliases if emitted

Suggested manifest additions:

- `"id_columns": ["orig_index", "Spectra"]`
- `"psi1_row_order": "train_indices in saved split order"`
- `"psi2_row_order": "test_indices followed by val_indices"`
- `"main_export_domain": "normalized_0_1"` or whatever is chosen
- `"compatibility_aliases": {...}`

### 3h. Value-domain decision point inside exporter

Two paths:

- Path A:
  - keep tcVAE main artifacts normalized
  - only align schema and labeling
- Path B:
  - transform main artifacts into the same numeric domain diffusion/testing expects
  - keep normalized tcVAE-specific files as auxiliaries

Recommendation:

- implement Path A first
- leave code structured so Path B is easy to add

## 4. `callbacks/cvae_setup_json.py`

Purpose:
- align setup artifact naming with tcVAE artifact family

Current gap:
- exporter writes top-level `cfg_cvae`
- checked-in tcVAE artifact uses `cfg_tcvae`

Planned changes:

- decide on one name and use it consistently

Recommended:

- switch exporter to emit `cfg_tcvae` if that is the desired external contract for this folder

Concrete edits:

- rename the nested top-level key from `cfg_cvae` to `cfg_tcvae`
- verify any downstream reader expectations before changing
- update artifact path field names only if necessary

Also update artifact references if new filenames are added:

- `psi1_path`
- `psi2_path`
- possibly compatibility alias paths

## 5. `scripts/export_cvae_setup_json.py`

Purpose:
- keep backfilled setup JSON generation consistent with runtime callback export

Current gap:
- mirrors current `cfg_cvae` structure

Planned changes:

- mirror the same key-name decision made in `callbacks/cvae_setup_json.py`
- update emitted artifact path references to the new main filenames if needed
- ensure setup JSON for old runs can still be backfilled into the new structure

Concrete edits:

- rename top-level nested key if switching to `cfg_tcvae`
- update `artifact_paths(...)` to reflect any new compatibility files

## 6. `config/base.yaml`

Purpose:
- optionally add configuration for labeled artifact export behavior

Current gap:
- callback is wired in, but no explicit artifact-format options exist

Planned changes:

Optional but useful:

- extend `callbacks.data_artifacts.SaveDataArtifacts` init args with:
  - `emit_diffusion_compatible_names: true`
  - `label_auxiliary_files: true`
  - `require_spectrum_ids: true`
  - optionally `main_export_domain: normalized`

This is not strictly required for the first patch, but it will make behavior explicit.

## 7. `FINCH-Science_SyntheticData/synthesis/isprs/tcvae/synthesis_cfg.yaml`

Purpose:
- fix checked-in artifact inconsistency

Current gap:

- says `architecture: CCVAEncoder`
- points to `cfg_diffusion_setup.json`

Planned changes:

- update to the actual tcVAE setup artifact
- ensure architecture name matches the intended tcVAE class/config

Concrete edits:

- change `cfg_model_setup_path` to `cfg_tcvae_setup.json` if that is the target contract
- correct architecture field if the checked-in value is stale

Note:
- if this file is a snapshot rather than generated code, patch it after generator behavior is finalized

## 8. New validator script: `scripts/validate_artifact_format.py`

Purpose:
- prevent drift after patching

Planned behavior:

- load a target artifact directory
- validate main files
- validate auxiliary files
- optionally compare schema against a reference diffusion file

Checks:

- required files exist
- header matches expected contract
- `orig_index` exists and is unique
- `Spectra` exists and is non-empty
- numeric columns are parseable
- auxiliary rows align with main rows
- row counts match split expectations

Optional comparison mode:

- compare tcVAE header exactly against diffusion `psi1_gdstreamline.csv`

## 9. Documentation note updates

Files:

- `docs/tmp_tcvae_vs_diffusion_artifact_diffs.md`
- `docs/tmp_tcvae_diffusion_artifact_checklist.md`

Planned changes:

- after implementation, append what was actually changed
- note any deviations from the initial plan

## Suggested implementation sequence

### Step 1

Patch `dataset.py` to preserve:

- `row_indices`
- `spectrum_ids`

### Step 2

Patch `callbacks/data_artifacts.py` to:

- build labeled split tables
- write main labeled `psi` files
- write labeled auxiliary files
- update manifest

### Step 3

Patch setup JSON exporters:

- `callbacks/cvae_setup_json.py`
- `scripts/export_cvae_setup_json.py`

### Step 4

Patch checked-in tcVAE synthesis config snapshot:

- `FINCH-Science_SyntheticData/synthesis/isprs/tcvae/synthesis_cfg.yaml`

### Step 5

Add validator:

- `scripts/validate_artifact_format.py`

### Step 6

Regenerate one tcVAE artifact set and validate it against the contract

## Recommended concrete policy choices

Recommended choices for minimal disruption:

- keep current tcVAE filenames
- add diffusion-compatible alias filenames
- label all CSV artifacts
- keep main files normalized for the first patch
- leave numeric-domain transformation as a second patch if integration still requires it

Recommended choices for strongest convergence:

- make main exported filenames diffusion-style
- label everything
- align main numeric domain too
- keep normalized tcVAE-only outputs as secondary auxiliary artifacts

## Risks to watch

- changing setup JSON top-level key may affect downstream loaders if they already expect `cfg_cvae`
- if `Spectra` is missing in any future source CSV, labeled export should fail clearly rather than silently dropping identity
- if main filenames are changed without aliases, existing tcVAE consumers may break
- numeric-domain changes are more likely to affect downstream models than schema-only changes

## Definition of done

The patch is done when:

- tcVAE main `psi` artifacts have the same visible schema as diffusion
- every exported CSV row is self-reconcilable using only the artifact file
- auxiliary files can be joined back to the main files row-for-row
- setup/config artifacts no longer reference diffusion filenames incorrectly
- validator passes on a regenerated tcVAE artifact directory

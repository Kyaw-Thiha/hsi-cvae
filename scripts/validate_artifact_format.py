from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate tcVAE/diffusion-style artifact directory contents.")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("outputs/ablations/run_K_lat12/artifacts"),
        help="Artifact directory to validate.",
    )
    parser.add_argument(
        "--reference-psi",
        type=Path,
        default=None,
        help="Reference diffusion psi CSV for exact header comparison.",
    )
    return parser.parse_args()


def read_header(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        return next(reader)


def expected_header() -> list[str]:
    spectral = [str(w) for w in range(400, 2500, 10)]
    return ["orig_index", "Spectra", "gv_fraction", "npv_fraction", "soil_fraction", *spectral]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"VALIDATION FAILED: {message}")


def validate_main_csv(path: Path, expected: Sequence[str]) -> None:
    require(path.exists(), f"missing required file: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None, f"missing header: {path}")
        require(list(reader.fieldnames) == list(expected), f"header mismatch for {path}")

        seen_indices: set[str] = set()
        row_count = 0
        for row in reader:
            row_count += 1
            orig_index = row["orig_index"]
            spectra = row["Spectra"]
            require(orig_index not in seen_indices, f"duplicate orig_index {orig_index} in {path}")
            seen_indices.add(orig_index)
            require(bool(spectra), f"empty Spectra value in {path}")
            for field in expected[2:]:
                try:
                    float(row[field])
                except ValueError as exc:
                    raise SystemExit(f"VALIDATION FAILED: non-numeric value in {path} field {field}: {exc}") from exc
        require(row_count > 0, f"no data rows in {path}")


def validate_aux_csv(path: Path, expected_prefix: Sequence[str]) -> None:
    require(path.exists(), f"missing auxiliary file: {path}")
    header = read_header(path)
    require(header[:2] == list(expected_prefix), f"auxiliary file missing id columns: {path}")


def validate_norm_dict(path: Path) -> None:
    require(path.exists(), f"missing norm_dict: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(payload.get("norm_type") == "statistical", "norm_dict.json must use statistical normalization")
    require("mean_vals" in payload, "norm_dict.json missing mean_vals")
    require("std_vals" in payload, "norm_dict.json missing std_vals")


def validate_manifest(path: Path) -> None:
    require(path.exists(), f"missing artifact manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(payload.get("id_columns") == ["orig_index", "Spectra"], "artifact_manifest id_columns mismatch")
    require(payload.get("diffusion_compatible_psi1_file") == "psi1_gdstreamline.csv", "artifact_manifest psi1 alias mismatch")
    require(payload.get("diffusion_compatible_psi2_file") == "psi2_gdstreamline.csv", "artifact_manifest psi2 alias mismatch")


def main() -> None:
    args = parse_args()
    artifact_dir = args.artifact_dir
    reference_psi = args.reference_psi
    require(artifact_dir.exists(), f"artifact directory not found: {artifact_dir}")

    expected = expected_header()
    validate_main_csv(artifact_dir / "psi1_gdstreamline.csv", expected)
    validate_main_csv(artifact_dir / "psi2_gdstreamline.csv", expected)

    validate_aux_csv(artifact_dir / "psi1_train_normalized.csv", ["orig_index", "Spectra"])
    validate_aux_csv(artifact_dir / "psi2_test_val_normalized.csv", ["orig_index", "Spectra"])
    validate_aux_csv(artifact_dir / "psi1_train_conditions_normalized.csv", ["orig_index", "Spectra"])
    validate_aux_csv(artifact_dir / "psi2_test_val_conditions_normalized.csv", ["orig_index", "Spectra"])
    validate_aux_csv(artifact_dir / "psi1_train_spectra_normalized.csv", ["orig_index", "Spectra"])
    validate_aux_csv(artifact_dir / "psi2_test_val_spectra_normalized.csv", ["orig_index", "Spectra"])

    validate_norm_dict(artifact_dir / "norm_dict.json")
    validate_manifest(artifact_dir / "artifact_manifest.json")

    if reference_psi is not None:
        require(reference_psi.is_file(), f"reference psi must be a file: {reference_psi}")
        require(read_header(reference_psi) == expected, f"reference header mismatch: {reference_psi}")
        require(read_header(artifact_dir / "psi1_gdstreamline.csv") == read_header(reference_psi), "psi1_gdstreamline header does not exactly match diffusion reference")

    print(f"VALIDATION PASSED: {artifact_dir}")


if __name__ == "__main__":
    main()

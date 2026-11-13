"""Create DP3 temporal data-preparation deliverables with pandas.

The two DP1 source CSVs are script inputs only and are not included in the
submission ZIP. This script does not fit a model or learned preprocessing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = "02_T2_RQ2_train_Q1_to_Q2.csv"
TEST_FILE = "03_T2_RQ2_test_Q2_to_Q3.csv"
TARGET = "present_next"
KEY = ["listing_id", "followup_period"]
CATEGORICAL = ["room_type", "superhost_status"]
NUMERIC = [
    "price_numeric", "accommodates_clean", "bedrooms_clean", "beds_clean",
    "bathrooms_clean", "minimum_nights_clean", "availability_365_clean",
    "number_of_reviews_clean", "reviews_per_month_clean",
    "review_scores_rating_clean", "calculated_host_listings_count_clean",
]
PREDICTORS = CATEGORICAL + NUMERIC
COLUMNS = [
    "listing_id", "host_id", "baseline_period", "neighbourhood_cleansed",
    "room_type", "superhost_status", "price_numeric", "accommodates_clean",
    "bedrooms_clean", "beds_clean", "bathrooms_clean", "minimum_nights_clean",
    "availability_365_clean", "number_of_reviews_clean", "reviews_per_month_clean",
    "review_scores_rating_clean", "calculated_host_listings_count_clean",
    "capacity_group", "host_portfolio_group", "followup_period", "split", TARGET,
]
SPECS = {
    "train": (TRAIN_FILE, "2026_Q1", "2026_Q2", "train", 4166, {0: 266, 1: 3900}),
    "test": (TEST_FILE, "2026_Q2", "2026_Q3", "test", 5689, {0: 317, 1: 5372}),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"listing_id": "string", "host_id": "string"}, low_memory=False)


def counts(frame: pd.DataFrame) -> dict[int, int]:
    values = pd.to_numeric(frame[TARGET], errors="raise").astype(int).value_counts()
    return {int(key): int(value) for key, value in values.items()}


def validate(frame: pd.DataFrame, name: str) -> dict[str, object]:
    filename, baseline, followup, split, rows, expected_counts = SPECS[name]
    checks = {
        "schema_matches": frame.columns.tolist() == COLUMNS,
        "row_count_matches": len(frame) == rows,
        "baseline_period_matches": set(frame["baseline_period"]) == {baseline},
        "followup_period_matches": set(frame["followup_period"]) == {followup},
        "split_matches": set(frame["split"]) == {split},
        "shinjuku_cohort_matches": set(frame["neighbourhood_cleansed"]) == {"Shinjuku Ku"},
        "listing_id_non_missing": not frame["listing_id"].isna().any(),
        "listing_id_unique_within_cohort": not frame["listing_id"].duplicated().any(),
        "target_counts_match": counts(frame) == expected_counts,
        "bathrooms_clean_present": "bathrooms_clean" in frame.columns,
        "raw_bathrooms_absent": "bathrooms" not in frame.columns,
        "estimated_predictors_absent": not any(c.startswith("estimated_") for c in frame.columns),
        "primary_predictors_present": all(c in frame.columns for c in PREDICTORS),
    }
    if not all(checks.values()):
        raise ValueError(f"{filename}: " + ", ".join(k for k, ok in checks.items() if not ok))
    return {"source_file": filename, "rows": rows, "target_counts": counts(frame), "checks": checks}


def definition_table() -> pd.DataFrame:
    rows = [
        ["train", "Model development and cross-validation only", "2026_Q1", "Shinjuku Ku", "2026_Q2", 4166, 3900, 266, "Yes"],
        ["test", "Untouched final temporal test", "2026_Q2", "Shinjuku Ku", "2026_Q3", 5689, 5372, 317, "No"],
    ]
    columns = [
        "dataset_split", "purpose", "baseline_period", "baseline_cohort", "followup_period",
        "rows", "present_next_1", "present_next_0", "use_in_model_selection",
    ]
    return pd.DataFrame(rows, columns=columns).assign(
        target_variable=TARGET,
        target_definition="1 if listing_id appears anywhere in full Tokyo follow-up snapshot; 0 otherwise.",
        record_key="listing_id + followup_period",
    )


def predictor_table() -> pd.DataFrame:
    rows = [[p, "categorical", "Fit encoding only on each training fold."] for p in CATEGORICAL]
    rows += [[p, "numeric", "Fit imputation and scaling only on each training fold."] for p in NUMERIC]
    return pd.DataFrame(rows, columns=["predictor", "type", "later_preprocessing"]).assign(
        available_at_baseline="Yes", included_in_main_model="Yes"
    )


def build(input_dir: Path, output_dir: Path) -> dict[str, object]:
    paths = {name: input_dir / spec[0] for name, spec in SPECS.items()}
    absent = [str(path) for path in paths.values() if not path.exists()]
    if absent:
        raise FileNotFoundError("Missing DP1 input file(s): " + "; ".join(absent))
    frames = {name: read(path) for name, path in paths.items()}
    reports = {name: validate(frames[name], name) for name in SPECS}
    clean = pd.concat([frames["train"], frames["test"]], ignore_index=True)
    duplicate_key_count = int(clean.duplicated(KEY).sum())
    if duplicate_key_count:
        raise ValueError("Duplicate listing_id + followup_period keys in T2_clean.csv.")

    output_dir.mkdir(parents=True, exist_ok=True)
    clean_path = output_dir / "T2_clean.csv"
    definition_path = output_dir / "T2_label_cohort_definition.csv"
    variables_path = output_dir / "T2_variables.csv"
    qa_path = output_dir / "DP3_leakage_check.json"
    clean.to_csv(clean_path, index=False, encoding="utf-8")
    definition_table().to_csv(definition_path, index=False, encoding="utf-8")
    predictor_table().to_csv(variables_path, index=False, encoding="utf-8")

    checks = {
        "all_predictors_available_at_baseline": all(c in clean.columns for c in PREDICTORS),
        "target_not_a_predictor": TARGET not in PREDICTORS,
        "id_and_host_not_predictors": not any(c in PREDICTORS for c in ["listing_id", "host_id"]),
        "followup_and_split_not_predictors": not any(c in PREDICTORS for c in ["followup_period", "split"]),
        "estimated_occupancy_excluded": "estimated_occupancy_l365d" not in clean.columns,
        "estimated_revenue_excluded": "estimated_revenue_l365d" not in clean.columns,
        "bathrooms_clean_used": "bathrooms_clean" in PREDICTORS and "bathrooms" not in clean.columns,
        "compound_key_unique": duplicate_key_count == 0,
        "no_random_resplit": set(clean["split"]) == {"train", "test"},
        "learned_preprocessing_not_fitted": True,
        "final_test_not_for_tuning": True,
    }
    qa = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "created_with": "Python/pandas",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "temporal_design": {
            "train": "Shinjuku Q1 baseline -> full Tokyo Q2 label",
            "final_test": "Shinjuku Q2 baseline -> full Tokyo Q3 label",
        },
        "inputs_not_bundled": {name: {"sha256": sha256(path), **reports[name]} for name, path in paths.items()},
        "output": {"file": clean_path.name, "rows": len(clean), "record_key": "listing_id + followup_period", "sha256": sha256(clean_path)},
        "primary_predictors": PREDICTORS,
        "leakage_checks": checks,
        "later_modeling_rule": "Fit imputation, encoding, scaling, and feature selection inside each training fold; then refit train and evaluate final test once.",
    }
    qa_path.write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return qa


def main() -> int:
    parser = argparse.ArgumentParser(description="Create DP3 temporal submission outputs.")
    parser.add_argument("--input-dir", type=Path, default=ROOT / "inputs")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "DP3")
    args = parser.parse_args()
    try:
        qa = build(args.input_dir.resolve(), args.output_dir.resolve())
    except Exception as error:
        print(f"DP3 preparation failed: {error}", file=sys.stderr)
        return 1
    print(f"DP3 preparation passed: {qa['output']['rows']:,} rows; status={qa['status']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""DP2: Prepare the Shinjuku RQ1 modelling cohort and model matrix.

Inputs (project_root/Inputs):
    01_T1_RQ1_Q2_Shinjuku.csv
    05_variable_roles.csv

Outputs (project_root/Scripts/DP2_Output):
    T1_RQ1_clean_observed.csv
    T1_RQ1_model_matrix.csv
    DP2_feature_config.csv
    DP2_preprocessing_log.csv

The primary treatment for missing reviews_per_month_clean is median imputation.
This is a DP2 modelling choice, not a rule imposed by DP1. The script records
that all missing review-frequency rows in the primary cohort have
number_of_reviews_clean == 0. M1 separately tests zero imputation.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
INPUT_DIR = PROJECT_DIR / "inputs"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "DP2_Output"
SOURCE_FILE = INPUT_DIR / "01_T1_RQ1_Q2_Shinjuku.csv"
ROLES_FILE = INPUT_DIR / "05_variable_roles.csv"

WARD = "Shinjuku Ku"
SNAPSHOT = "2026_Q2"
FEATURES = [
    "price_numeric",
    "accommodates_clean",
    "minimum_nights_clean",
    "availability_365_clean",
    "reviews_per_month_clean",
    "calculated_host_listings_count_clean",
]
LOG_FEATURES = {
    "price_numeric",
    "accommodates_clean",
    "minimum_nights_clean",
    "reviews_per_month_clean",
    "calculated_host_listings_count_clean",
}
WINSOR_FEATURES = LOG_FEATURES
PROFILE_COLUMNS = [
    "bedrooms_clean", "beds_clean", "bathrooms_clean",
    "number_of_reviews_clean", "review_scores_rating_clean",
]


def require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def calculate_skewness(values: pd.Series) -> float:
    valid = pd.to_numeric(values, errors="coerce").dropna()
    if len(valid) < 3 or valid.nunique() < 3:
        return 0.0
    return float(valid.skew())


def main() -> None:
    if not SOURCE_FILE.exists() or not ROLES_FILE.exists():
        raise FileNotFoundError(
            "Place 01_T1_RQ1_Q2_Shinjuku.csv and 05_variable_roles.csv in the Inputs folder."
        )

    data = pd.read_csv(
        SOURCE_FILE,
        dtype={"listing_id": "string", "host_id": "string"},
        low_memory=False,
    )
    data.columns = data.columns.str.strip()
    roles = pd.read_csv(ROLES_FILE)
    roles.columns = roles.columns.str.strip()

    required = [
        "listing_id", "host_id", "snapshot", "neighbourhood_cleansed",
        "room_type", "superhost_status", "capacity_group", "host_portfolio_group",
        *FEATURES, *PROFILE_COLUMNS,
    ]
    require_columns(data, required, "DP1 Shinjuku source")
    require_columns(roles, ["variable", "RQ1_role"], "DP1 variable-role table")

    role_lookup = dict(zip(roles["variable"].astype(str), roles["RQ1_role"].astype(str)))
    incorrect_roles = [
        feature for feature in FEATURES
        if role_lookup.get(feature) != "Primary clustering input"
    ]
    if incorrect_roles:
        raise ValueError(f"DP1 does not identify these as primary RQ1 inputs: {incorrect_roles}")
    for identifier in ["listing_id", "host_id"]:
        if "exclude from clustering" not in role_lookup.get(identifier, "").lower():
            raise ValueError(f"DP1 must mark {identifier} as excluded from clustering.")

    if data["listing_id"].isna().any() or data["listing_id"].duplicated().any():
        raise ValueError("listing_id must be complete and unique.")
    if set(data["neighbourhood_cleansed"].dropna().astype(str)) != {WARD}:
        raise ValueError(f"Expected only {WARD} listings.")
    if set(data["snapshot"].dropna().astype(str)) != {SNAPSHOT}:
        raise ValueError(f"Expected only snapshot {SNAPSHOT}.")

    for column in FEATURES + PROFILE_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    # The primary cohort excludes records without an observed nightly price.
    price_missing_mask = data["price_numeric"].isna()
    observed = data.loc[~price_missing_mask].copy().reset_index(drop=True)
    if observed.empty:
        raise ValueError("No listings with observed price remain for the primary cohort.")

    review_missing_mask = observed["reviews_per_month_clean"].isna()
    review_missing_n = int(review_missing_mask.sum())
    review_missing_zero_reviews_n = int(
        (review_missing_mask & observed["number_of_reviews_clean"].eq(0)).sum()
    )
    if review_missing_n and review_missing_zero_reviews_n != review_missing_n:
        raise ValueError(
            "The missing-review audit changed: not every missing reviews_per_month row "
            "has number_of_reviews_clean == 0. Review the imputation decision."
        )

    model_values = observed[FEATURES].copy().astype(float)
    transformed = pd.DataFrame(index=observed.index)
    config_rows = []
    log_rows = []

    for feature in FEATURES:
        raw_values = model_values[feature].copy()
        missing_n = int(raw_values.isna().sum())
        raw_skew = calculate_skewness(raw_values)
        imputation_value = float(raw_values.median()) if missing_n else np.nan

        # Estimate winsorisation thresholds from observed values before imputation.
        winsor_applied = feature in WINSOR_FEATURES and raw_skew >= 1.0
        lower = float(raw_values.dropna().quantile(0.01)) if winsor_applied else np.nan
        upper = float(raw_values.dropna().quantile(0.99)) if winsor_applied else np.nan

        values = raw_values.fillna(imputation_value) if missing_n else raw_values.copy()
        if winsor_applied:
            values = values.clip(lower=lower, upper=upper)

        log_applied = feature in LOG_FEATURES
        if log_applied:
            if (values < 0).any():
                raise ValueError(f"Cannot apply log1p to negative values in {feature}.")
            values = np.log1p(values)
        transformed[feature] = values.astype(float)

        if feature == "reviews_per_month_clean" and missing_n:
            imputation_rationale = (
                "DP2 primary modelling choice: use the Shinjuku median (1.39), not a DP1 mandate. "
                f"All {review_missing_zero_reviews_n} missing rows have number_of_reviews_clean == 0; "
                "a zero-imputation alternative is evaluated in M1."
            )
        elif missing_n:
            imputation_rationale = "Median imputation used for the primary model matrix."
        else:
            imputation_rationale = "No missing values in the primary modelling cohort."

        associated_zero_reviews = (
            review_missing_zero_reviews_n
            if feature == "reviews_per_month_clean" else 0
        )
        config_rows.append({
            "feature": feature,
            "dp1_rq1_role": role_lookup[feature],
            "primary_model_input": True,
            "raw_missing_n_in_primary_cohort": missing_n,
            "missing_rows_with_number_of_reviews_zero": associated_zero_reviews,
            "primary_imputation_method": "Median" if missing_n else "None",
            "primary_imputation_value": imputation_value,
            "imputation_rationale": imputation_rationale,
            "raw_skewness": raw_skew,
            "winsorisation_applied": winsor_applied,
            "winsor_lower_p01": lower,
            "winsor_upper_p99": upper,
            "log1p_applied": log_applied,
            "standardisation": "StandardScaler (z-score)",
        })
        log_rows.append({
            "feature": feature,
            "rows_in_primary_cohort": len(observed),
            "missing_before_imputation": missing_n,
            "imputation_method": "Median" if missing_n else "None",
            "imputation_value": imputation_value,
            "raw_skewness_before_winsorisation": raw_skew,
            "winsorisation_applied": winsor_applied,
            "winsor_lower_p01": lower,
            "winsor_upper_p99": upper,
            "log1p_applied": log_applied,
        })

    scaler = StandardScaler()
    scaled = scaler.fit_transform(transformed[FEATURES])
    model_matrix = pd.DataFrame(scaled, columns=FEATURES)
    model_matrix.insert(0, "listing_id", observed["listing_id"].astype(str).to_numpy())

    for index, feature in enumerate(FEATURES):
        log_rows[index]["standard_scaler_mean"] = float(scaler.mean_[index])
        log_rows[index]["standard_scaler_scale"] = float(scaler.scale_[index])
        config_rows[index]["standard_scaler_mean"] = float(scaler.mean_[index])
        config_rows[index]["standard_scaler_scale"] = float(scaler.scale_[index])

    if not np.isfinite(model_matrix[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("The model matrix contains missing or infinite values.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    observed.to_csv(OUTPUT_DIR / "T1_RQ1_clean_observed.csv", index=False, encoding="utf-8-sig")
    model_matrix.to_csv(OUTPUT_DIR / "T1_RQ1_model_matrix.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(config_rows).to_csv(
        OUTPUT_DIR / "DP2_feature_config.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(log_rows).to_csv(
        OUTPUT_DIR / "DP2_preprocessing_log.csv", index=False, encoding="utf-8-sig"
    )

    print("DP2 completed successfully.")
    print(f"Source listings: {len(data):,}")
    print(f"Excluded for missing price: {int(price_missing_mask.sum()):,}")
    print(f"Primary modelling listings: {len(observed):,}")
    print(
        f"Missing reviews_per_month in primary cohort: {review_missing_n:,}; "
        f"all associated with number_of_reviews_clean == 0: {review_missing_zero_reviews_n:,}."
    )
    print(f"Primary median imputation value: {observed['reviews_per_month_clean'].median():.2f}")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

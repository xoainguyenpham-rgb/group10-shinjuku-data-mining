"""M1: Evaluate K-Means, sensitivity scenarios, and the final Orange K=5 solution.

Inputs:
    ../inputs/01_T1_RQ1_Q2_Shinjuku.csv
    ../inputs/Orange_K5_final_output.tab
    ../outputs/DP2_Output/T1_RQ1_model_matrix.csv
    ../outputs/DP2_Output/T1_RQ1_clean_observed.csv
    ../outputs/DP2_Output/DP2_feature_config.csv
    ../outputs/DP2_Output/DP2_preprocessing_log.csv

Outputs (../outputs/M1_Output):
    M1_evaluation.csv
    M1_sensitivity_analysis.csv
    M1_sensitivity_assignments.csv
    M1_K3_K4_K5_cluster_profiles.csv
    M1_K4_to_K5_transition.csv
    Q1_id_cluster.csv
    M1_Silhouette_by_K.png
    M1_Stability_by_K.png
    M1_result_interpretation.txt

All ARI comparisons between different modelling populations are calculated
on the common listing_id set after a validated one-to-one ID join. The fixed
Silhouette sample is reproducible and its random seed and ID hash are recorded.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
import hashlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
INPUT_DIR = PROJECT_DIR / "inputs"
DP2_DIR = PROJECT_DIR / "outputs" / "DP2_Output"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "M1_Output"
SOURCE_FILE = INPUT_DIR / "01_T1_RQ1_Q2_Shinjuku.csv"
ORANGE_FILE = INPUT_DIR / "Orange_K5_final_output.tab"

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
K_VALUES = range(2, 9)
FINAL_K = 5
N_SEEDS = 10
N_INIT = 10
SILHOUETTE_SAMPLE_SIZE = 5000
SILHOUETTE_SAMPLE_SEED = 42
WARD_SAMPLE_SIZE = 2000
WARD_SAMPLE_SEED = 43

CLUSTER_NAMES = {
    1: "Extended-Stay Listings with Large Tokyo-Scoped Host Listing Counts",
    2: "Higher-Availability Standard Listings with Smaller Tokyo-Scoped Host Listing Counts",
    3: "Lower-Availability Standard Listings with Multi-Listing Hosts (Tokyo Scope)",
    4: "Mid-Capacity Listings with Very Large Tokyo-Scoped Host Listing Counts",
    5: "Higher-Priced, Large-Capacity Listings",
}

PROFILE_NUMERIC = [
    "price_numeric",
    "accommodates_clean",
    "minimum_nights_clean",
    "availability_365_clean",
    "reviews_per_month_clean",
    "calculated_host_listings_count_clean",
]


def read_csv_required(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")
    return pd.read_csv(path, low_memory=False, **kwargs)


def sha256_ids(values: pd.Series | list[str]) -> str:
    payload = "\n".join(sorted(pd.Series(values).astype(str).tolist())).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def run_seed_ensemble(matrix: np.ndarray, k: int) -> dict:
    labels_by_seed: list[np.ndarray] = []
    inertias: list[float] = []
    with threadpool_limits(limits=1):
        for seed in range(N_SEEDS):
            model = KMeans(
                n_clusters=k,
                init="k-means++",
                n_init=N_INIT,
                max_iter=300,
                random_state=seed,
                algorithm="lloyd",
            )
            model.fit(matrix)
            labels_by_seed.append(model.labels_.copy())
            inertias.append(float(model.inertia_))

    pairwise_ari = [
        adjusted_rand_score(labels_by_seed[i], labels_by_seed[j])
        for i, j in combinations(range(N_SEEDS), 2)
    ]
    best_index = int(np.argmin(inertias))
    labels = labels_by_seed[best_index]
    counts = np.bincount(labels, minlength=k)
    return {
        "labels": labels,
        "best_seed": best_index,
        "inertia": inertias[best_index],
        "mean_seed_ari": float(np.mean(pairwise_ari)),
        "min_seed_ari": float(np.min(pairwise_ari)),
        "counts": counts,
    }


def load_orange_labels(matrix: np.ndarray, expected_ids: pd.Series | None = None) -> tuple[np.ndarray, float]:
    if not ORANGE_FILE.exists():
        raise FileNotFoundError(
            "Place the final Orange K=5 export at Inputs/Orange_K5_final_output.tab."
        )
    orange = pd.read_csv(
        ORANGE_FILE,
        sep="\t",
        skiprows=[1, 2],
        dtype={"listing_id": "string", "Cluster": "string"},
        low_memory=False,
    )
    required = ["Cluster", *FEATURES]
    missing = [column for column in required if column not in orange.columns]
    if missing:
        raise ValueError(f"Orange export is missing columns: {missing}")
    orange_matrix = orange[FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if len(orange_matrix) != len(matrix):
        raise ValueError(f"Orange has {len(orange_matrix):,} rows; DP2 has {len(matrix):,}.")
    if np.isnan(orange_matrix).any():
        raise ValueError("The Orange export contains missing feature values.")
    if expected_ids is not None and "listing_id" in orange.columns:
        # Orange rounds long IDs (float precision), so compare as float64, row by row.
        orange_ids = pd.to_numeric(orange["listing_id"], errors="coerce").to_numpy(float)
        dp2_ids = pd.to_numeric(expected_ids, errors="coerce").to_numpy(float)
        if not np.array_equal(orange_ids, dp2_ids):
            raise ValueError("Orange listing_id order does not match DP2 (compared as float64).")
    max_difference = float(np.nanmax(np.abs(orange_matrix - matrix)))
    if not np.isfinite(max_difference) or max_difference > 1e-5:
        raise ValueError(
            "Orange features do not match the DP2 matrix row by row. "
            f"Maximum absolute difference: {max_difference:.8g}."
        )
    labels = orange["Cluster"].astype("string")
    if labels.isna().any() or labels.nunique() != FINAL_K:
        raise ValueError("The Orange export must contain exactly five non-missing clusters.")
    return labels.astype(str).to_numpy(), max_difference


def build_scenario_matrix(
    frame: pd.DataFrame,
    features: list[str],
    bounds: dict[str, tuple[float, float]],
    log_flags: dict[str, bool],
    imputation_values: dict[str, float] | None = None,
    preserve_zero_feature: str | None = None,
    disable_log: bool = False,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Apply DP2 transformations to a specified cohort and feature set."""
    imputation_values = imputation_values or {}
    working = frame.copy().reset_index(drop=True)
    transformed = pd.DataFrame(index=working.index)

    for feature in features:
        values = pd.to_numeric(working[feature], errors="coerce").astype(float)
        original_missing = values.isna().to_numpy()
        if values.isna().any():
            if feature not in imputation_values or pd.isna(imputation_values[feature]):
                raise ValueError(f"No imputation value supplied for {feature}.")
            values = values.fillna(float(imputation_values[feature]))

        if feature in bounds:
            lower, upper = bounds[feature]
            zero_mask = values.eq(0.0) if preserve_zero_feature == feature else None
            values = values.clip(lower=lower, upper=upper)
            if zero_mask is not None and zero_mask.any():
                values.loc[zero_mask] = 0.0

        if not disable_log and log_flags.get(feature, False):
            if (values < 0).any():
                raise ValueError(f"Cannot apply log1p to negative values in {feature}.")
            values = np.log1p(values)
        transformed[feature] = values.astype(float)

    matrix = StandardScaler().fit_transform(transformed[features])
    if not np.isfinite(matrix).all():
        raise ValueError("A sensitivity matrix contains missing or infinite values.")
    return working[["listing_id"]].copy(), matrix


def merge_labels_to_primary(
    primary: pd.DataFrame,
    alternative: pd.DataFrame,
    label_column: str,
    expected_common_n: int | None = None,
) -> pd.DataFrame:
    if primary["listing_id"].duplicated().any() or alternative["listing_id"].duplicated().any():
        raise ValueError("Duplicate listing_id found before ARI comparison.")
    merged = primary[["listing_id", "orange_cluster", "primary_cluster"]].merge(
        alternative[["listing_id", label_column]],
        on="listing_id",
        how="inner",
        validate="one_to_one",
    )
    if expected_common_n is not None and len(merged) != expected_common_n:
        raise AssertionError(
            f"Expected {expected_common_n:,} shared IDs for {label_column}; found {len(merged):,}."
        )
    if merged["listing_id"].nunique() != len(merged):
        raise AssertionError("The ARI comparison contains duplicate shared IDs.")
    return merged


def rank_labels_by_price(frame: pd.DataFrame, labels: np.ndarray) -> tuple[np.ndarray, dict]:
    temp = pd.DataFrame({
        "listing_id": frame["listing_id"].astype(str).to_numpy(),
        "price_numeric": pd.to_numeric(frame["price_numeric"], errors="coerce").to_numpy(),
        "raw_label": labels,
    })
    median_price = temp.groupby("raw_label")["price_numeric"].median().sort_values()
    mapping = {raw_label: rank for rank, raw_label in enumerate(median_price.index, start=1)}
    ranked = np.array([mapping[value] for value in labels], dtype=int)
    return ranked, mapping


def save_m1_charts(evaluation: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(evaluation["k"], evaluation["silhouette_sample"], marker="o", linewidth=2)
    selected = evaluation.loc[evaluation["k"].eq(FINAL_K), "silhouette_sample"].iloc[0]
    ax.scatter([FINAL_K], [selected], marker="*", s=190, zorder=5, label=f"Selected K={FINAL_K}")
    ax.axvline(FINAL_K, linestyle="--", linewidth=1, alpha=0.6)
    for k, score in zip(evaluation["k"], evaluation["silhouette_sample"]):
        ax.annotate(f"{score:.3f}", (k, score), xytext=(0, 8), textcoords="offset points", ha="center")
    ax.set_title("Silhouette Score by Number of Clusters")
    ax.set_xlabel("Number of clusters (K)")
    ax.set_ylabel("Mean silhouette score")
    ax.set_xticks(evaluation["k"].tolist())
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "M1_Silhouette_by_K.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6.0))
    ax.plot(evaluation["k"], evaluation["mean_ari_across_10_seeds"], marker="o", linewidth=2, label="Mean pairwise ARI")
    ax.plot(evaluation["k"], evaluation["minimum_ari_across_10_seeds"], marker="s", linestyle="--", linewidth=1.8, label="Minimum pairwise ARI")
    ax.axvline(FINAL_K, linestyle=":", linewidth=1, alpha=0.7, label=f"Selected K={FINAL_K}")
    lower_bound = min(-0.05, float(evaluation["minimum_ari_across_10_seeds"].min()) - 0.05)
    ax.set_ylim(lower_bound, 1.03)
    ax.set_title("K-Means Stability Across 10 Random Seeds")
    ax.set_xlabel("Number of clusters (K)")
    ax.set_ylabel("Adjusted Rand Index (ARI)")
    ax.set_xticks(evaluation["k"].tolist())
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="lower left")
    fig.text(
        0.5, 0.015,
        "Higher ARI indicates greater agreement between seed solutions; 1 indicates identical partitions up to label permutation.",
        ha="center", va="bottom", fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUTPUT_DIR / "M1_Stability_by_K.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def create_cluster_profiles(
    source_observed: pd.DataFrame,
    label_frame: pd.DataFrame,
    k_value: int,
    cluster_column: str,
    cluster_name_map: dict[int, str] | None = None,
) -> pd.DataFrame:
    merged = source_observed.merge(
        label_frame[["listing_id", cluster_column]],
        on="listing_id", how="inner", validate="one_to_one",
    )
    rows = []
    for cluster, group in merged.groupby(cluster_column, sort=True):
        row = {
            "k": k_value,
            "cluster": int(cluster),
            "cluster_name": (cluster_name_map or {}).get(int(cluster), f"K={k_value} Segment {int(cluster)}"),
            "listing_count": len(group),
            "listing_share_pct": round(len(group) / len(merged) * 100, 2),
        }
        for feature in PROFILE_NUMERIC:
            values = pd.to_numeric(group[feature], errors="coerce")
            row[f"{feature}_valid_n"] = int(values.notna().sum())
            row[f"{feature}_missing_n"] = int(values.isna().sum())
            row[f"{feature}_median"] = float(values.median()) if values.notna().any() else np.nan
        host_count_median = row["calculated_host_listings_count_clean_median"]
        row["host_listing_count_scope_note"] = (
            "Number of listings associated with the host within the Tokyo scope; not an ownership count."
        )
        for category in ["1", "2-5", "6+"]:
            row[f"host_portfolio_group_{category.replace('-', '_').replace('+', 'plus')}_pct"] = round(
                group["host_portfolio_group"].astype("string").eq(category).mean() * 100, 2
            )
        for category in ["Entire home/apt", "Private room", "Hotel room", "Shared room"]:
            safe = category.lower().replace("/", "_").replace(" ", "_")
            row[f"room_type_{safe}_pct"] = round(group["room_type"].astype("string").eq(category).mean() * 100, 2)
        row["dominant_host_portfolio_group"] = (
            group["host_portfolio_group"].mode().iloc[0]
            if group["host_portfolio_group"].notna().any() else "Missing"
        )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matrix_frame = read_csv_required(DP2_DIR / "T1_RQ1_model_matrix.csv", dtype={"listing_id": "string"})
    observed = read_csv_required(DP2_DIR / "T1_RQ1_clean_observed.csv", dtype={"listing_id": "string", "host_id": "string"})
    source = read_csv_required(SOURCE_FILE, dtype={"listing_id": "string", "host_id": "string"})
    source.columns = source.columns.str.strip()

    for column in FEATURES + ["price_numeric", "number_of_reviews_clean"]:
        if column in source.columns:
            source[column] = pd.to_numeric(source[column], errors="coerce")
    if matrix_frame["listing_id"].duplicated().any() or observed["listing_id"].duplicated().any():
        raise ValueError("Duplicate listing_id found in DP2 outputs.")
    if not matrix_frame["listing_id"].astype(str).equals(observed["listing_id"].astype(str)):
        raise ValueError("DP2 observed data and model matrix are not aligned by source row order.")

    matrix = matrix_frame[FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(matrix).all():
        raise ValueError("Primary DP2 model matrix contains missing or infinite values.")
    orange_labels, orange_max_diff = load_orange_labels(matrix, matrix_frame["listing_id"])

    primary_ids = matrix_frame[["listing_id"]].copy()
    primary_ids["orange_cluster"] = orange_labels
    observed_price = observed[["listing_id", "price_numeric"]].copy()
    primary_ids = primary_ids.merge(observed_price, on="listing_id", validate="one_to_one")
    orange_medians = primary_ids.groupby("orange_cluster")["price_numeric"].median().sort_values()
    orange_to_rank = {label: rank for rank, label in enumerate(orange_medians.index, start=1)}
    primary_ids["primary_cluster"] = primary_ids["orange_cluster"].map(orange_to_rank).astype(int)
    primary_ids["cluster_name"] = primary_ids["primary_cluster"].map(CLUSTER_NAMES)

    dp2_config = read_csv_required(DP2_DIR / "DP2_feature_config.csv")
    dp2_log = read_csv_required(DP2_DIR / "DP2_preprocessing_log.csv").set_index("feature")
    bounds: dict[str, tuple[float, float]] = {}
    dp2_log_flags: dict[str, bool] = {}
    primary_imputation: dict[str, float] = {}
    for feature in FEATURES:
        lower, upper = dp2_log.loc[feature, "winsor_lower_p01"], dp2_log.loc[feature, "winsor_upper_p99"]
        if pd.notna(lower) and pd.notna(upper):
            bounds[feature] = (float(lower), float(upper))
        dp2_log_flags[feature] = bool(dp2_log.loc[feature, "log1p_applied"])
        value = dp2_config.loc[dp2_config["feature"].eq(feature), "primary_imputation_value"]
        primary_imputation[feature] = float(value.iloc[0]) if len(value) and pd.notna(value.iloc[0]) else np.nan

    # A deterministic 5,000-ID sample is used for all main K Silhouette values.
    primary_sample_positions = np.random.RandomState(SILHOUETTE_SAMPLE_SEED).choice(
        len(matrix), size=min(SILHOUETTE_SAMPLE_SIZE, len(matrix)), replace=False
    )
    primary_sample_ids = matrix_frame.iloc[primary_sample_positions]["listing_id"].astype(str).tolist()
    sample_hash = sha256_ids(primary_sample_ids)

    # Evaluate K=2..8 using the DP2 primary model matrix.
    evaluation_rows = []
    primary_runs: dict[int, dict] = {}
    for k in K_VALUES:
        run = run_seed_ensemble(matrix, k)
        primary_runs[k] = run
        silhouette = float(silhouette_score(matrix[primary_sample_positions], run["labels"][primary_sample_positions]))
        evaluation_rows.append({
            "k": k,
            "silhouette_sample": silhouette,
            "silhouette_sample_n": len(primary_sample_positions),
            "silhouette_sample_seed": SILHOUETTE_SAMPLE_SEED,
            "silhouette_sample_id_sha256": sample_hash,
            "mean_ari_across_10_seeds": run["mean_seed_ari"],
            "minimum_ari_across_10_seeds": run["min_seed_ari"],
            "best_seed": run["best_seed"],
            "best_inertia": run["inertia"],
            "cluster_counts": " | ".join(str(int(value)) for value in run["counts"]),
            "minimum_cluster_share_pct": float(run["counts"].min() / len(matrix) * 100),
            "maximum_cluster_share_pct": float(run["counts"].max() / len(matrix) * 100),
            "selected_for_final_market_segmentation": k == FINAL_K,
        })
    evaluation = pd.DataFrame(evaluation_rows)
    k5_python_orange_ari = float(adjusted_rand_score(orange_labels, primary_runs[FINAL_K]["labels"]))
    orange_silhouette = float(silhouette_score(matrix[primary_sample_positions], orange_labels[primary_sample_positions]))
    evaluation["orange_k5_silhouette_sample"] = np.nan
    evaluation["python_best_seed_vs_orange_k5_ari"] = np.nan
    evaluation.loc[evaluation["k"].eq(FINAL_K), "orange_k5_silhouette_sample"] = orange_silhouette
    evaluation.loc[evaluation["k"].eq(FINAL_K), "python_best_seed_vs_orange_k5_ari"] = k5_python_orange_ari
    evaluation.to_csv(OUTPUT_DIR / "M1_evaluation.csv", index=False, encoding="utf-8-sig")
    save_m1_charts(evaluation)

    # Ranked K=3 and K=4 benchmark assignments. K=5 remains the official Orange solution.
    k_assignments: dict[int, pd.DataFrame] = {}
    for k in [3, 4]:
        ranked, _ = rank_labels_by_price(observed, primary_runs[k]["labels"])
        k_assignments[k] = pd.DataFrame({"listing_id": observed["listing_id"].astype(str), f"k{k}_cluster": ranked})
    k5_assignments = primary_ids[["listing_id", "primary_cluster"]].rename(columns={"primary_cluster": "k5_cluster"})
    k345_profiles = []
    for k in [3, 4]:
        profile = create_cluster_profiles(observed, k_assignments[k], k, f"k{k}_cluster")
        k345_profiles.append(profile)
    k5_profile = create_cluster_profiles(
        observed,
        k5_assignments.rename(columns={"k5_cluster": "k5_cluster"}),
        5,
        "k5_cluster",
        CLUSTER_NAMES,
    )
    k345_profiles.append(k5_profile)
    pd.concat(k345_profiles, ignore_index=True).to_csv(
        OUTPUT_DIR / "M1_K3_K4_K5_cluster_profiles.csv", index=False, encoding="utf-8-sig"
    )

    k4_k5 = k_assignments[4].merge(k5_assignments, on="listing_id", how="inner", validate="one_to_one")
    k4_counts = k4_k5["k4_cluster"].value_counts().to_dict()
    transition = (
        k4_k5.groupby(["k4_cluster", "k5_cluster"], as_index=False)
        .size().rename(columns={"size": "listing_count"})
    )
    transition["k4_cluster_total"] = transition["k4_cluster"].map(k4_counts)
    transition["share_within_k4_cluster_pct"] = (
        transition["listing_count"] / transition["k4_cluster_total"] * 100
    ).round(2)
    transition["share_of_full_sample_pct"] = (transition["listing_count"] / len(k4_k5) * 100).round(2)
    transition["k5_cluster_name"] = transition["k5_cluster"].map(CLUSTER_NAMES)
    transition["common_id_count"] = len(k4_k5)
    transition["join_method"] = "Inner one-to-one listing_id join"
    transition.to_csv(OUTPUT_DIR / "M1_K4_to_K5_transition.csv", index=False, encoding="utf-8-sig")

    # Create the sensitivity model matrices from the DP1 source using documented DP2 bounds.
    source_primary = source.loc[source["price_numeric"].notna()].copy().reset_index(drop=True)
    source_primary = source_primary.merge(
        observed[["listing_id"]], on="listing_id", how="inner", validate="one_to_one"
    )
    if len(source_primary) != len(observed):
        raise AssertionError("The source primary cohort does not match the 5,328 observed-price cohort.")

    scenario_specs: list[dict] = []
    scenario_labels: dict[str, pd.DataFrame] = {}
    scenario_metrics: dict[str, dict] = {}

    def evaluate_scenario(
        scenario_id: str,
        scenario_name: str,
        frame: pd.DataFrame,
        features: list[str],
        imputation_values: dict[str, float] | None = None,
        preserve_zero_feature: str | None = None,
        disable_log: bool = False,
        scenario_note: str = "",
        seed_ensemble: bool = True,
    ) -> None:
        ids, scenario_matrix = build_scenario_matrix(
            frame,
            features,
            bounds,
            dp2_log_flags,
            imputation_values=imputation_values,
            preserve_zero_feature=preserve_zero_feature,
            disable_log=disable_log,
        )
        if seed_ensemble:
            model_run = run_seed_ensemble(scenario_matrix, FINAL_K)
            raw_labels = model_run["labels"]
            mean_seed_ari = model_run["mean_seed_ari"]
            min_seed_ari = model_run["min_seed_ari"]
            best_seed = model_run["best_seed"]
            inertia = model_run["inertia"]
            counts = model_run["counts"]
        else:
            raise ValueError("Use the dedicated Ward evaluation block for hierarchical clustering.")

        label_column = f"{scenario_id}_cluster_raw"
        alt = ids.copy()
        alt[label_column] = raw_labels
        common = merge_labels_to_primary(primary_ids, alt, label_column)
        ari_vs_primary = float(adjusted_rand_score(common["orange_cluster"], common[label_column]))

        scenario_labels[scenario_id] = alt
        scenario_metrics[scenario_id] = {
            "scenario": scenario_name,
            "scenario_note": scenario_note,
            "modelled_rows": len(ids),
            "common_ids_with_primary": len(common),
            "ari_vs_orange_primary": ari_vs_primary,
            "mean_ari_between_seed_solutions": mean_seed_ari,
            "minimum_ari_between_seed_solutions": min_seed_ari,
            "best_seed": best_seed,
            "best_inertia": inertia,
            "cluster_counts": " | ".join(str(int(value)) for value in counts),
            "common_listing_ids_sha256": sha256_ids(common["listing_id"]),
            "join_method": "Inner one-to-one merge on listing_id",
        }

    # No-log sensitivity: same 5,328 IDs and same winsorisation bounds; skip log1p only.
    evaluate_scenario(
        "no_log",
        "No log transformation",
        source_primary,
        FEATURES,
        imputation_values=primary_imputation,
        disable_log=True,
        scenario_note="Same primary cohort and DP2 winsorisation bounds; omit log1p, then refit StandardScaler.",
    )

    # Drop host listing count: five remaining primary features, same primary cohort.
    no_host_features = [feature for feature in FEATURES if feature != "calculated_host_listings_count_clean"]
    evaluate_scenario(
        "no_host_count",
        "Exclude calculated host listing count",
        source_primary,
        no_host_features,
        imputation_values=primary_imputation,
        scenario_note="Same primary cohort and transformations; omit the host listing-count feature and refit StandardScaler.",
    )

    # Full-sample price sensitivity: all 5,689 rows, impute price and reviews using full-cohort medians.
    full_price_frame = source.copy().reset_index(drop=True)
    full_price_median = float(full_price_frame["price_numeric"].median())
    full_review_median = float(full_price_frame["reviews_per_month_clean"].median())
    evaluate_scenario(
        "price_full_median",
        "Retain all 5,689 rows; median-impute price and review frequency",
        full_price_frame,
        FEATURES,
        imputation_values={**primary_imputation, "price_numeric": full_price_median, "reviews_per_month_clean": full_review_median},
        scenario_note=f"Price median={full_price_median:.3f}; reviews/month median re-estimated on full cohort={full_review_median:.2f}. ARI is on the 5,328 common primary IDs.",
    )

    # Price-only control: keep reviews/month imputation fixed at the primary-cohort median.
    primary_review_median = primary_imputation["reviews_per_month_clean"]
    evaluate_scenario(
        "price_full_hold_review",
        "Retain all 5,689 rows; impute price and hold review median at primary value",
        full_price_frame,
        FEATURES,
        imputation_values={**primary_imputation, "price_numeric": full_price_median, "reviews_per_month_clean": primary_review_median},
        scenario_note=f"Price median={full_price_median:.3f}; reviews/month held at primary DP2 median={primary_review_median:.2f}. ARI is on the 5,328 common primary IDs.",
    )

    # Review complete-case sensitivity: exclude only the 638 missing-review rows from the primary cohort.
    review_complete = source_primary.loc[source_primary["reviews_per_month_clean"].notna()].copy().reset_index(drop=True)
    evaluate_scenario(
        "review_complete_case",
        "Exclude records with missing reviews_per_month",
        review_complete,
        FEATURES,
        scenario_note="Complete cases for review frequency within the 5,328 price-observed primary cohort; ARI is on the shared IDs.",
    )

    # Review zero-imputation sensitivity: all missing review frequency rows have zero total reviews.
    review_zero_frame = source_primary.copy().reset_index(drop=True)
    missing_reviews = review_zero_frame["reviews_per_month_clean"].isna()
    structural_zero = review_zero_frame.loc[missing_reviews, "number_of_reviews_clean"].eq(0)
    if not structural_zero.all():
        raise ValueError("The zero-imputation scenario is invalid: some missing review rows have non-zero total reviews.")
    review_zero_frame["reviews_per_month_clean"] = review_zero_frame["reviews_per_month_clean"].fillna(0.0)
    evaluate_scenario(
        "review_zero",
        "Set missing reviews_per_month to zero",
        review_zero_frame,
        FEATURES,
        preserve_zero_feature="reviews_per_month_clean",
        scenario_note=f"Set all {int(missing_reviews.sum())} missing review-frequency values to zero; each has number_of_reviews_clean=0. ARI is on all 5,328 primary IDs.",
    )

    # Ward validation: deterministic 2,000-ID sample, explicitly recorded and ID-matched.
    ward_positions = np.random.RandomState(WARD_SAMPLE_SEED).choice(
        len(matrix), size=min(WARD_SAMPLE_SIZE, len(matrix)), replace=False
    )
    ward_labels = AgglomerativeClustering(n_clusters=FINAL_K, linkage="ward").fit_predict(matrix[ward_positions])
    ward_frame = pd.DataFrame({
        "listing_id": matrix_frame.iloc[ward_positions]["listing_id"].astype(str).to_numpy(),
        "ward_cluster_raw": ward_labels,
    })
    ward_common = merge_labels_to_primary(primary_ids, ward_frame, "ward_cluster_raw", expected_common_n=len(ward_positions))
    ward_ari = float(adjusted_rand_score(ward_common["orange_cluster"], ward_common["ward_cluster_raw"]))
    ward_silhouette = float(silhouette_score(matrix[ward_positions], ward_labels))
    ward_row = {
        "scenario": "Ward hierarchical clustering (method comparison)",
        "scenario_note": "Ward linkage fitted on a deterministic sample; not a seed-based K-Means sensitivity.",
        "modelled_rows": len(ward_positions),
        "common_ids_with_primary": len(ward_common),
        "ari_vs_orange_primary": ward_ari,
        "mean_ari_between_seed_solutions": np.nan,
        "minimum_ari_between_seed_solutions": np.nan,
        "best_seed": np.nan,
        "best_inertia": np.nan,
        "cluster_counts": " | ".join(str(int(value)) for value in np.bincount(ward_labels, minlength=FINAL_K)),
        "common_listing_ids_sha256": sha256_ids(ward_common["listing_id"]),
        "join_method": "Inner one-to-one merge on listing_id",
        "silhouette_sample": ward_silhouette,
        "silhouette_sample_n": len(ward_positions),
        "silhouette_sample_seed": WARD_SAMPLE_SEED,
        "silhouette_sample_id_sha256": sha256_ids(ward_common["listing_id"]),
    }

    # A common sensitivity silhouette sample is drawn from IDs available in the review complete-case scenario.
    complete_ids = set(review_complete["listing_id"].astype(str))
    common_sens_sample = [value for value in primary_sample_ids if value in complete_ids]
    sensitivity_sample_n = min(4500, len(common_sens_sample))
    sensitivity_sample_ids = np.random.RandomState(SILHOUETTE_SAMPLE_SEED).choice(
        np.array(common_sens_sample, dtype=object), size=sensitivity_sample_n, replace=False
    ).tolist()
    sensitivity_sample_hash = sha256_ids(sensitivity_sample_ids)
    primary_position_by_id = {
        value: index for index, value in enumerate(matrix_frame["listing_id"].astype(str))
    }
    primary_sensitivity_positions = np.array(
        [primary_position_by_id[value] for value in sensitivity_sample_ids], dtype=int
    )
    primary_sensitivity_silhouette = float(
        silhouette_score(
            matrix[primary_sensitivity_positions],
            orange_labels[primary_sensitivity_positions],
        )
    )

    # Silhouette values are calculated on the same sensitivity IDs for every applicable scenario.
    sensitivity_rows = []
    primary_sens_labels = primary_ids[["listing_id", "orange_cluster"]].copy()
    sensitivity_rows.append({
        "scenario": "Primary Orange K=5 reference",
        "method": "Orange K-Means",
        "modelled_rows": len(primary_ids),
        "common_ids_with_primary": len(primary_ids),
        "ari_vs_orange_primary": np.nan,
        "mean_ari_between_seed_solutions": np.nan,
        "minimum_ari_between_seed_solutions": np.nan,
        "ari_interpretation": "Reference solution; not compared against itself.",
        "silhouette": primary_sensitivity_silhouette,
        "silhouette_sample_n": len(sensitivity_sample_ids),
        "silhouette_sample_seed": SILHOUETTE_SAMPLE_SEED,
        "silhouette_sample_id_sha256": sensitivity_sample_hash,
        "modelled_cluster_counts": " | ".join(str(int(value)) for value in primary_ids["orange_cluster"].value_counts().sort_index().values),
        "join_method": "Orange row order validated against all six DP2 features; exact IDs restored from DP1.",
    })

    for scenario_id, metrics in scenario_metrics.items():
        alt = scenario_labels[scenario_id]
        sample_ids_for_scenario = sensitivity_sample_ids
        joined = primary_ids[["listing_id", "orange_cluster"]].merge(
            alt, on="listing_id", how="inner", validate="one_to_one"
        )
        if len(joined) != metrics["common_ids_with_primary"]:
            raise AssertionError("Sensitivity ARI join count changed after scenario evaluation.")

        # Reconstruct transformed scenario features in exactly the same row order as its label frame.
        if scenario_id in {"no_log", "no_host_count"}:
            frame = source_primary
            features = FEATURES if scenario_id == "no_log" else no_host_features
            impute = primary_imputation
            _, scenario_matrix = build_scenario_matrix(
                frame, features, bounds, dp2_log_flags, imputation_values=impute,
                disable_log=(scenario_id == "no_log"),
            )
        elif scenario_id == "price_full_median":
            _, scenario_matrix = build_scenario_matrix(
                full_price_frame, FEATURES, bounds, dp2_log_flags,
                imputation_values={**primary_imputation, "price_numeric": full_price_median, "reviews_per_month_clean": full_review_median},
            )
        elif scenario_id == "price_full_hold_review":
            _, scenario_matrix = build_scenario_matrix(
                full_price_frame, FEATURES, bounds, dp2_log_flags,
                imputation_values={**primary_imputation, "price_numeric": full_price_median, "reviews_per_month_clean": primary_review_median},
            )
        elif scenario_id == "review_complete_case":
            _, scenario_matrix = build_scenario_matrix(
                review_complete, FEATURES, bounds, dp2_log_flags,
            )
        elif scenario_id == "review_zero":
            _, scenario_matrix = build_scenario_matrix(
                review_zero_frame, FEATURES, bounds, dp2_log_flags,
                preserve_zero_feature="reviews_per_month_clean",
            )
        else:
            raise ValueError(f"Unknown sensitivity scenario: {scenario_id}")

        id_to_pos = {value: index for index, value in enumerate(alt["listing_id"].astype(str))}
        positions = np.array([id_to_pos[value] for value in sample_ids_for_scenario], dtype=int)
        label_map = alt.set_index("listing_id")[f"{scenario_id}_cluster_raw"]
        sample_labels = pd.Series(sample_ids_for_scenario).map(label_map).to_numpy()
        silhouette = float(silhouette_score(scenario_matrix[positions], sample_labels))
        sensitivity_rows.append({
            "scenario": metrics["scenario"],
            "method": "K-Means, K=5, 10 seeds, n_init=10",
            "modelled_rows": metrics["modelled_rows"],
            "common_ids_with_primary": metrics["common_ids_with_primary"],
            "ari_vs_orange_primary": metrics["ari_vs_orange_primary"],
            "mean_ari_between_seed_solutions": metrics["mean_ari_between_seed_solutions"],
            "minimum_ari_between_seed_solutions": metrics["minimum_ari_between_seed_solutions"],
            "ari_interpretation": "ARI vs Orange primary uses common listing_id values; seed ARIs are pairwise comparisons among the 10 solutions within this scenario.",
            "silhouette": silhouette,
            "silhouette_sample_n": len(sample_ids_for_scenario),
            "silhouette_sample_seed": SILHOUETTE_SAMPLE_SEED,
            "silhouette_sample_id_sha256": sensitivity_sample_hash,
            "modelled_cluster_counts": metrics["cluster_counts"],
            "join_method": metrics["join_method"],
            "scenario_note": metrics["scenario_note"],
            "best_seed": metrics["best_seed"],
        })

    sensitivity_rows.append({
        "scenario": ward_row["scenario"],
        "method": "Agglomerative clustering, Ward linkage, K=5",
        "modelled_rows": ward_row["modelled_rows"],
        "common_ids_with_primary": ward_row["common_ids_with_primary"],
        "ari_vs_orange_primary": ward_row["ari_vs_orange_primary"],
        "mean_ari_between_seed_solutions": np.nan,
        "minimum_ari_between_seed_solutions": np.nan,
        "ari_interpretation": "Ward is not seed-based; ARI is calculated on the same 2,000 listing IDs used to fit Ward.",
        "silhouette": ward_row["silhouette_sample"],
        "silhouette_sample_n": ward_row["silhouette_sample_n"],
        "silhouette_sample_seed": WARD_SAMPLE_SEED,
        "silhouette_sample_id_sha256": ward_row["silhouette_sample_id_sha256"],
        "modelled_cluster_counts": ward_row["cluster_counts"],
        "join_method": ward_row["join_method"],
        "scenario_note": ward_row["scenario_note"],
    })

    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(OUTPUT_DIR / "M1_sensitivity_analysis.csv", index=False, encoding="utf-8-sig")

    # Export ID-level labels for all K=5 sensitivity tests.
    assignment_export = source[["listing_id"]].copy()
    assignment_export["orange_primary_cluster"] = assignment_export["listing_id"].map(primary_ids.set_index("listing_id")["orange_cluster"])
    assignment_export["primary_ranked_cluster"] = assignment_export["listing_id"].map(primary_ids.set_index("listing_id")["primary_cluster"])
    for scenario_id, alt in scenario_labels.items():
        column = f"{scenario_id}_cluster_raw"
        assignment_export[column] = assignment_export["listing_id"].map(alt.set_index("listing_id")[column])
    assignment_export["ward_cluster_raw"] = assignment_export["listing_id"].map(ward_frame.set_index("listing_id")["ward_cluster_raw"])
    assignment_export.to_csv(OUTPUT_DIR / "M1_sensitivity_assignments.csv", index=False, encoding="utf-8-sig")

    # K4-to-K5 transition evidence, based on an ID-validated one-to-one join.
    # Re-create the assignment file after all labels are available.
    k4_k5 = k_assignments[4].merge(k5_assignments, on="listing_id", how="inner", validate="one_to_one")

    # Final Q1 mapping uses Orange K=5 labels, price-ranked for readable M2 labels.
    final_assignments = primary_ids[["listing_id", "orange_cluster", "primary_cluster"]].copy()
    final_assignments = final_assignments.rename(columns={"primary_cluster": "cluster"})
    final_assignments["cluster_name"] = final_assignments["cluster"].map(CLUSTER_NAMES)
    final_assignments.to_csv(OUTPUT_DIR / "Q1_id_cluster.csv", index=False, encoding="utf-8-sig")

    # Guard: CLUSTER_NAMES are assigned by price rank; confirm the profile still matches them.
    k5_check = k5_profile.set_index("cluster")
    checks = {
        "cluster 1 has the highest median minimum nights": k5_check["minimum_nights_clean_median"].idxmax() == 1,
        "cluster 5 has the highest median capacity": k5_check["accommodates_clean_median"].idxmax() == 5,
        "cluster 4 has the highest median host listing count": k5_check["calculated_host_listings_count_clean_median"].idxmax() == 4,
        "cluster 2 has higher median availability than cluster 3": k5_check.loc[2, "availability_365_clean_median"] > k5_check.loc[3, "availability_365_clean_median"],
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise AssertionError(f"CLUSTER_NAMES no longer match the cluster profiles: {failed}")

    # Correct interpretation language for the final M1 report.
    k3 = evaluation.loc[evaluation["k"].eq(3)].iloc[0]
    k4 = evaluation.loc[evaluation["k"].eq(4)].iloc[0]
    k5 = evaluation.loc[evaluation["k"].eq(5)].iloc[0]
    k6_8_mean = float(evaluation.loc[evaluation["k"].between(6, 8), "mean_ari_across_10_seeds"].mean())
    k6_min_share = float(evaluation.loc[evaluation["k"].eq(6), "minimum_cluster_share_pct"].iloc[0])
    def scen_ari(prefix: str) -> float:
        return float(sensitivity.loc[sensitivity["scenario"].str.startswith(prefix), "ari_vs_orange_primary"].iloc[0])
    sensitivity_line = (
        "Feature sensitivity (ARI vs Orange K=5): "
        f"no log transform={scen_ari('No log'):.3f}; exclude host listing count={scen_ari('Exclude calculated host'):.3f}; "
        f"Ward method (2,000-ID sample)={scen_ari('Ward'):.3f}. Seed stability does not imply robustness to feature choice."
    )
    interpretation = [
        "M1 RESULT INTERPRETATION",
        "========================",
        "Final clustering tool: Orange Data Mining; final assignment: Orange K=5 labels.",
        f"Silhouette sample: n={len(primary_sample_positions):,}; random seed={SILHOUETTE_SAMPLE_SEED}; SHA-256={sample_hash}.",
        f"K=3: Silhouette={k3['silhouette_sample']:.4f}; mean pairwise seed ARI={k3['mean_ari_across_10_seeds']:.4f}; maximum cluster share={k3['maximum_cluster_share_pct']:.2f}%.",
        f"K=4: Silhouette={k4['silhouette_sample']:.4f}; mean pairwise seed ARI={k4['mean_ari_across_10_seeds']:.4f}; maximum cluster share={k4['maximum_cluster_share_pct']:.2f}%.",
        f"K=5: Silhouette={k5['silhouette_sample']:.4f}; mean pairwise seed ARI={k5['mean_ari_across_10_seeds']:.4f}; minimum cluster share={k5['minimum_cluster_share_pct']:.2f}%.",
        "Higher ARI indicates greater agreement between seed solutions; 1 indicates identical partitions up to label permutation.",
        f"K=5 mean pairwise seed ARI={k5['mean_ari_across_10_seeds']:.4f}; the mean over K=6-8 is {k6_8_mean:.4f} (lower).",
        f"Silhouette ranking: K=3 {k3['silhouette_sample']:.4f} > K=4 {k4['silhouette_sample']:.4f} > K=5 {k5['silhouette_sample']:.4f}. K=5 does not maximise Silhouette.",
        f"At K=6, the smallest cluster represented {k6_min_share:.2f}% of the sample. This is a point for interpretability review, not automatic evidence that the model is poor.",
        "K=5 is selected for market-segmentation utility, not because it maximises Silhouette. K=3 is the strongest statistical baseline but leaves a broad majority segment; K=4 mainly separates standard listings by availability; K=5 provides additional differentiation in Tokyo-scoped host listing count and capacity/price profiles.",
        f"Python best-seed K=5 vs Orange K=5 ARI={k5_python_orange_ari:.6f}; Orange assignments remain the authoritative final labels.",
        "Sensitivity ARI columns are distinct: ari_vs_orange_primary compares a scenario's best partition with Orange K=5 labels on common IDs; mean/minimum pairwise ARI describe agreement among the 10 seed solutions within that scenario.",
        "For scenarios with different row counts, the ARI comparison uses an inner one-to-one listing_id join and only the common IDs. The 5,689-row price-imputation scenario is compared with the primary model on the 5,328 shared IDs.",
        sensitivity_line,
        "Silhouette values in M1_sensitivity_analysis.csv come from different feature spaces and are not comparable across scenarios. Sensitivity Silhouettes use IDs with observed reviews_per_month only.",
        "Host listing count is interpreted as the number of listings associated with a host within the Tokyo scope, not as a count of listings owned by the host.",
        "Cluster 5 is named Higher-Priced, Large-Capacity Listings. The name describes observed price and capacity; it does not claim premium quality.",
        "Availability is not occupancy; review frequency is not booking volume. These descriptive segments do not establish profitability, listing activity, regulatory compliance, or host success.",
    ]
    (OUTPUT_DIR / "M1_result_interpretation.txt").write_text("\n".join(interpretation) + "\n", encoding="utf-8")

    # Validation checks.
    if final_assignments["listing_id"].duplicated().any() or len(final_assignments) != len(matrix):
        raise AssertionError("Final Q1 assignments do not have exactly one row per primary listing ID.")
    if len(k4_k5) != len(matrix) or k4_k5["listing_id"].duplicated().any():
        raise AssertionError("K4-to-K5 transition is not a one-to-one full-cohort comparison.")

    print("M1 completed successfully.")
    print(f"Primary listings: {len(matrix):,}; final K={FINAL_K} Orange assignments.")
    print(f"Silhouette sample seed: {SILHOUETTE_SAMPLE_SEED}; n={len(primary_sample_positions):,}.")
    print(f"Python best-seed vs Orange K=5 ARI: {k5_python_orange_ari:.6f}.")
    print(f"Sensitivity scenarios: {len(sensitivity)} rows; outputs include ID-level assignments and K=3/4/5 profiles.")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

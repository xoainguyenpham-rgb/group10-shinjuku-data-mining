"""M2: Profile the final Orange K=5 market segments using observed values.

Inputs:
    DP2_Output/T1_RQ1_clean_observed.csv
    M1_Output/Q1_id_cluster.csv

Outputs:
    M2_Output/M2_cluster_profile.csv
    M2_Output/M2_cluster_profile.png

All numeric summaries use observed values. Missing values are reported rather
than replaced. Host listing count is described as the number of listings
associated with a host within the Tokyo scope, not as an ownership count.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DP2_DIR = PROJECT_DIR / "outputs" / "DP2_Output"
M1_DIR = PROJECT_DIR / "outputs" / "M1_Output"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "M2_Output"

NUMERIC_COLUMNS = [
    "price_numeric", "accommodates_clean", "minimum_nights_clean",
    "availability_365_clean", "reviews_per_month_clean",
    "calculated_host_listings_count_clean", "bedrooms_clean", "beds_clean",
    "bathrooms_clean", "number_of_reviews_clean", "review_scores_rating_clean",
]
CATEGORICAL_COLUMNS = [
    "room_type", "superhost_status", "capacity_group", "host_portfolio_group",
]
HOST_SCOPE_NOTE = (
    "Number of listings associated with the host within the Tokyo scope; "
    "not a count of listings owned by the host."
)
SHORT_NAMES = {
    1: "Extended stay",
    2: "Higher availability",
    3: "Lower availability",
    4: "Large host listing count",
    5: "Higher price, large capacity",
}


def add_numeric_profile(row: dict, group: pd.DataFrame, column: str) -> None:
    values = pd.to_numeric(group[column], errors="coerce")
    valid = values.dropna()
    row[f"{column}_valid_n"] = int(valid.size)
    row[f"{column}_missing_n"] = int(values.isna().sum())
    row[f"{column}_missing_pct"] = round(float(values.isna().mean() * 100), 2)
    row[f"{column}_median"] = float(valid.median()) if not valid.empty else np.nan
    row[f"{column}_mean"] = float(valid.mean()) if not valid.empty else np.nan


# Cells outlined on the slide: the values behind the three findings beside the chart.
# (feature column, final cluster number C1..C5)
KEY_CELLS = {
    ("price_numeric_median", 1), ("price_numeric_median", 5),
    ("availability_365_clean_median", 2), ("availability_365_clean_median", 3),
    ("calculated_host_listings_count_clean_median", 4),
}


def plot_heatmap(profile: pd.DataFrame) -> None:
    """Original heatmap layout with a calmer palette and light outlines on key values."""
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle

    plot_features = [
        ("price_numeric_median", "Median price (JPY/night)", "{:,.0f}"),
        ("accommodates_clean_median", "Median capacity (guests)", "{:,.0f}"),
        ("minimum_nights_clean_median", "Median minimum nights", "{:,.0f}"),
        ("availability_365_clean_median", "Median availability (days)", "{:,.0f}"),
        ("reviews_per_month_clean_median", "Median reviews per month", "{:,.2f}"),
        ("calculated_host_listings_count_clean_median", "Median host listing count (Tokyo)", "{:,.0f}"),
    ]
    table = profile.set_index("cluster")[[c for c, _, _ in plot_features]].T
    z_values = table.sub(table.mean(axis=1), axis=0).div(
        table.std(axis=1).replace(0, np.nan), axis=0
    ).fillna(0)
    # Same blue-low / red-high logic as before, with lower saturation.
    calm = LinearSegmentedColormap.from_list("calm_diverging", ["#7FA6CC", "#F6F5F2", "#D08A82"])

    fig, ax = plt.subplots(figsize=(7.5, 4.95), dpi=200)
    image = ax.imshow(z_values.to_numpy(), cmap=calm, vmin=-1.7, vmax=1.7, aspect="auto")
    ax.set_xticks(range(len(profile)))
    ax.set_xticklabels([
        f"Cluster {int(r.cluster)}\n{int(r.listing_count):,} ({r.listing_share_pct:.1f}%)"
        for r in profile.itertuples()
    ], fontsize=8.5, color="#102B46")
    ax.set_yticks(range(len(plot_features)))
    ax.set_yticklabels([label for _, label, _ in plot_features], fontsize=9, color="#102B46")
    for i, (column, _, fmt) in enumerate(plot_features):
        for j, cluster_id in enumerate(table.columns):
            value = table.loc[column, cluster_id]
            key = (column, int(cluster_id)) in KEY_CELLS
            ax.text(j, i, "NA" if pd.isna(value) else fmt.format(value), ha="center", va="center",
                    fontsize=9.5 if key else 8.5, fontweight="bold" if key else "normal",
                    color="#102B46" if key else "#3A4654")
            if key:  # thin, slightly inset outline in a soft navy
                ax.add_patch(Rectangle((j - 0.46, i - 0.46), 0.92, 0.92, fill=False,
                                       ec="#5876A0", lw=1.25, zorder=10))
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-.5, len(profile), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(plot_features), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.2)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.suptitle("Shinjuku Airbnb Market Segments (K=5)", fontsize=10.5, fontweight="bold",
                 color="#102B46", y=0.998)
    fig.text(0.5, 0.935, "Observed medians; colour shows relative position within each row. "
             "Outlined cells = values behind the three findings.", ha="center", fontsize=7.5, color="#697784")
    cbar = fig.colorbar(image, ax=ax, shrink=0.72, pad=0.02, ticks=[-1.5, 0, 1.5])
    cbar.ax.set_yticklabels(["lower", "", "higher"], fontsize=7.5)
    cbar.set_label("Relative level (row-standardised)", fontsize=7.5, color="#697784")
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUTPUT_DIR / "M2_cluster_profile.png", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    observed_file = DP2_DIR / "T1_RQ1_clean_observed.csv"
    assignment_file = M1_DIR / "Q1_id_cluster.csv"
    if not observed_file.exists() or not assignment_file.exists():
        raise FileNotFoundError("Run dp2_prepare.py and m1_evaluate.py before m2_profile.py.")

    observed = pd.read_csv(
        observed_file,
        dtype={"listing_id": "string", "host_id": "string"},
        low_memory=False,
    )
    assignments = pd.read_csv(
        assignment_file,
        dtype={"listing_id": "string", "orange_cluster": "string"},
        low_memory=False,
    )
    if observed["listing_id"].duplicated().any() or assignments["listing_id"].duplicated().any():
        raise ValueError("M2 requires unique listing_id values in both inputs.")
    if set(observed["listing_id"].astype(str)) != set(assignments["listing_id"].astype(str)):
        raise ValueError("Observed data and final cluster assignments do not contain the same listing IDs.")

    data = observed.merge(
        assignments,
        on="listing_id",
        how="left",
        validate="one_to_one",
    )
    if data["cluster"].isna().any() or data["listing_id"].duplicated().any():
        raise ValueError("M2 could not match every listing to exactly one final cluster.")

    rows = []
    for cluster_id, group in data.groupby("cluster", sort=True):
        row = {
            "cluster": int(cluster_id),
            "orange_cluster": str(group["orange_cluster"].iloc[0]),
            "cluster_name": str(group["cluster_name"].iloc[0]),
            "listing_count": int(len(group)),
            "listing_share_pct": round(len(group) / len(data) * 100, 2),
            "host_listing_count_interpretation": HOST_SCOPE_NOTE,
        }
        for column in NUMERIC_COLUMNS:
            add_numeric_profile(row, group, column)
        for column in CATEGORICAL_COLUMNS:
            values = group[column].fillna("Missing").astype(str)
            dominant = values.mode().iloc[0]
            row[f"dominant_{column}"] = dominant
            row[f"dominant_{column}_share_pct"] = round(float(values.eq(dominant).mean() * 100), 2)
        rows.append(row)

    profile = pd.DataFrame(rows).sort_values("cluster")
    if int(profile["listing_count"].sum()) != len(data):
        raise AssertionError("Cluster counts do not sum to the modelling sample.")
    if not np.isclose(profile["listing_share_pct"].sum(), 100.0, atol=0.1):
        raise AssertionError("Cluster shares do not sum to 100%.")
    if set(profile["cluster"].astype(int)) != set(range(1, 6)):
        raise AssertionError("M2 expected five price-ranked clusters.")

    profile.to_csv(OUTPUT_DIR / "M2_cluster_profile.csv", index=False, encoding="utf-8-sig")

    plot_heatmap(profile)

    print("M2 completed successfully.")
    print(f"Profiled listings: {len(data):,}; clusters: {len(profile)}.")
    print("Cluster 5 label: Higher-Priced, Large-Capacity Listings.")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

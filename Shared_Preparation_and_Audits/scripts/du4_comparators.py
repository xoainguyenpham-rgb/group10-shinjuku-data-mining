from pathlib import Path
import os
os.chdir(Path(__file__).resolve().parents[1])
def display(value):
    print(value.to_string() if hasattr(value,"to_string") else value)
import numpy as np
# The supplied Q2 raw snapshot is the late-June 2026 Tokyo extract used for this legacy analysis.
from pathlib import Path
import pandas as pd

filename = "Tokyo_listings_2026_Q2.csv"
current_dir = Path.cwd().resolve()
DATA_FILE = None
for candidate in (current_dir, *current_dir.parents):
    options = [
        candidate / "Shared_Preparation_and_Audits" / "raw_data" / filename,
        candidate / "raw_data" / filename,
    ]
    DATA_FILE = next((path for path in options if path.is_file()), None)
    if DATA_FILE is not None:
        break

if DATA_FILE is None:
    raise FileNotFoundError(
        f"Could not locate {filename} in the repository raw_data folder."
    )

df = pd.read_csv(DATA_FILE, low_memory=False, dtype={"id": "string", "host_id": "string"})
print("Loaded source:", filename)
print("Dataset shape:", df.shape)


# Check listing-level scraping dates
df["last_scraped"] = pd.to_datetime(
    df["last_scraped"],
    errors="coerce"
)

print("Earliest scrape date:", df["last_scraped"].min())
print("Latest scrape date:", df["last_scraped"].max())

print("\nListings by scrape date:")
print(
    df["last_scraped"]
    .value_counts()
    .sort_index()
)

wards = [
    "Shinjuku Ku",
    "Shibuya Ku",
    "Taito Ku",
    "Toshima Ku"
]

du4 = df[
    df["neighbourhood_cleansed"].isin(wards)
].copy()

ward_counts = (
    du4["neighbourhood_cleansed"]
    .value_counts()
    .reindex(wards)
)

print("Listings by ward:")
print(ward_counts)

print("\nTotal four-ward observations:", len(du4))

# Duplicate listing IDs
duplicate_ids = du4["id"].duplicated().sum()

print("Duplicate listing IDs:", duplicate_ids)

# Missingness in variables used for structural comparison
structural_vars = [
    "room_type",
    "accommodates",
    "calculated_host_listings_count"
]

missing_structural = du4[structural_vars].isna().sum()

print("\nMissing values in structural variables:")
print(missing_structural)

du4["price_num"] = pd.to_numeric(
    du4["price"]
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False),
    errors="coerce"
)

print(du4[["price", "price_num"]].head())

price_audit = (
    du4.groupby("neighbourhood_cleansed")
       .agg(
           total_listings=("id", "size"),
           valid_price=("price_num", "count")
       )
       .reindex(wards)
)

price_audit["missing_price"] = (
    price_audit["total_listings"]
    - price_audit["valid_price"]
)

price_audit["missing_price_pct"] = (
    price_audit["missing_price"]
    / price_audit["total_listings"]
    * 100
)

print(price_audit.round(2))

non_positive = (
    du4["price_num"].dropna() <= 0
).sum()

print("\nNon-positive numeric prices:", non_positive)

# Capacity groups
du4["capacity_group"] = pd.cut(
    du4["accommodates"],
    bins=[0, 2, 4, np.inf],
    labels=["1-2", "3-4", "5+"],
    include_lowest=True
)

# Host-size groups
# IMPORTANT: use calculated_host_listings_count,
# not host_listings_count
du4["host_size_group"] = pd.cut(
    du4["calculated_host_listings_count"],
    bins=[0, 1, 5, np.inf],
    labels=["1", "2-5", "6+"],
    include_lowest=True
)

print("Capacity groups:")
print(du4["capacity_group"].value_counts(dropna=False))

print("\nHost-size groups:")
print(du4["host_size_group"].value_counts(dropna=False))

# N
N = (
    du4.groupby("neighbourhood_cleansed")
       .size()
       .reindex(wards)
)

# Room-type shares
room_share = (
    pd.crosstab(
        du4["neighbourhood_cleansed"],
        du4["room_type"],
        normalize="index"
    ) * 100
).reindex(wards)

# Median capacity
median_capacity = (
    du4.groupby("neighbourhood_cleansed")
       ["accommodates"]
       .median()
       .reindex(wards)
)

# Listing-weighted host-size shares
host_share = (
    pd.crosstab(
        du4["neighbourhood_cleansed"],
        du4["host_size_group"],
        normalize="index"
    ) * 100
).reindex(wards)

table1 = pd.DataFrame({
    "N": N,
    "Entire home/apt (%)":
        room_share["Entire home/apt"],
    "Private room (%)":
        room_share["Private room"],
    "Median capacity":
        median_capacity,
    "Host size: 1 (%)":
        host_share["1"],
    "Host size: 2-5 (%)":
        host_share["2-5"],
    "Host size: 6+ (%)":
        host_share["6+"]
})

print(table1.round(1))

capacity_table = (
    pd.crosstab(
        du4["neighbourhood_cleansed"],
        du4["capacity_group"],
        normalize="index"
    ) * 100
).reindex(wards)

print(capacity_table.round(1))

price_summary = (
    du4.groupby("neighbourhood_cleansed")
       ["price_num"]
       .agg(
           Valid_N="count",
           Median="median",
           P25=lambda x: x.quantile(0.25),
           P75=lambda x: x.quantile(0.75)
       )
       .reindex(wards)
)

print(price_summary.round(1))

shinjuku_median = price_summary.loc[
    "Shinjuku Ku", "Median"
]

overall_diff = {}

for comparator in [
    "Shibuya Ku",
    "Toshima Ku",
    "Taito Ku"
]:
    comparator_median = price_summary.loc[
        comparator, "Median"
    ]

    overall_diff[comparator] = (
        100
        * (shinjuku_median - comparator_median)
        / comparator_median
    )

print("\nShinjuku median price difference:")
for market, diff in overall_diff.items():
    print(f"vs {market}: {diff:.1f}%")

major_rooms = du4[
    du4["room_type"].isin([
        "Entire home/apt",
        "Private room"
    ])
].copy()

room_price = (
    major_rooms
    .groupby(
        ["neighbourhood_cleansed", "room_type"]
    )["price_num"]
    .agg(
        Valid_N="count",
        Median="median"
    )
)

table4 = room_price.unstack("room_type")
table4 = table4.reindex(wards)

print(table4.round(1))

capacity_price = du4.loc[du4["room_type"].eq("Entire home/apt")].groupby(["capacity_group", "neighbourhood_cleansed"], observed=True)["price_num"].agg(N="count", Median="median")
median_by_capacity = (
    capacity_price["Median"]
    .unstack("neighbourhood_cleansed")
)

n_by_capacity = (
    capacity_price["N"]
    .unstack("neighbourhood_cleansed")
)

median_by_capacity = median_by_capacity[
    wards
]

n_by_capacity = n_by_capacity[
    wards
]

print("Median price:")
print(median_by_capacity.round(1))

print("\nValid N:")
print(n_by_capacity)

comparators = [
    "Shibuya Ku",
    "Toshima Ku",
    "Taito Ku"
]

price_diff = pd.DataFrame(
    index=median_by_capacity.index
)

for comparator in comparators:
    price_diff[f"Relative to {comparator}"] = (
        100
        * (
            median_by_capacity["Shinjuku Ku"]
            - median_by_capacity[comparator]
        )
        / median_by_capacity[comparator]
    )

print(price_diff.round(1))

import matplotlib.pyplot as plt
import pandas as pd

# Calculate room-type shares for visualization
room_plot = (
    pd.crosstab(
        du4["neighbourhood_cleansed"],
        du4["room_type"],
        normalize="index"
    ) * 100
)

# Keep the two major room types
room_plot = room_plot[
    ["Entire home/apt", "Private room"]
].reindex([
    "Shinjuku Ku",
    "Toshima Ku",
    "Shibuya Ku",
    "Taito Ku"
])

# Short labels for display
room_plot.index = [
    "Shinjuku",
    "Toshima",
    "Shibuya",
    "Taito"
]

# Plot
ax = room_plot.plot(
    kind="bar",
    figsize=(9, 5)
)

ax.set_title("Room-Type Composition Across Candidate Wards")
ax.set_xlabel("Ward")
ax.set_ylabel("Share of Listings (%)")
ax.set_ylim(0, 100)

plt.xticks(rotation=0)
plt.legend(title="Room Type")

# Add percentage labels
for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.1f%%",
        padding=3,
        fontsize=9
    )

plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Clean price if price_num has not been created
if "price_num" not in du4.columns:
    du4["price_num"] = pd.to_numeric(
        du4["price"]
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False),
        errors="coerce"
    )

# Create capacity groups if not already created
if "capacity_group" not in du4.columns:
    du4["capacity_group"] = pd.cut(
        du4["accommodates"],
        bins=[0, 2, 4, np.inf],
        labels=["1-2", "3-4", "5+"],
        include_lowest=True
    )

# Keep entire-home listings only
entire = du4[
    du4["room_type"] == "Entire home/apt"
].copy()

# Calculate median price for the chart
price_plot = (
    entire.groupby(
        ["capacity_group", "neighbourhood_cleansed"],
        observed=True
    )["price_num"]
    .median()
    .unstack()
)

# Order wards
price_plot = price_plot.reindex(
    columns=[
        "Shinjuku Ku",
        "Toshima Ku",
        "Shibuya Ku",
        "Taito Ku"
    ]
)

price_plot.columns = [
    "Shinjuku",
    "Toshima",
    "Shibuya",
    "Taito"
]

# Plot
ax = price_plot.plot(
    kind="bar",
    figsize=(10, 5.5)
)

ax.set_title(
    "Median Price of Entire-Home Listings by Capacity Group"
)
ax.set_xlabel("Accommodation Capacity")
ax.set_ylabel("Median Nightly Price (JPY)")

plt.xticks(rotation=0)
plt.legend(title="Ward")

# Add values above bars
for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.0f",
        padding=3,
        fontsize=8
    )

plt.tight_layout()
plt.show()

print("=== FINAL DU4 VALIDATION ===")

print("Full dataset:", df.shape)
print("DU4 subset:", du4.shape)

print(
    "Duplicate IDs:",
    du4["id"].duplicated().sum()
)

print(
    "Missing room_type:",
    du4["room_type"].isna().sum()
)

print(
    "Missing accommodates:",
    du4["accommodates"].isna().sum()
)

print(
    "Missing calculated_host_listings_count:",
    du4["calculated_host_listings_count"]
       .isna().sum()
)

print(
    "Non-positive prices:",
    (du4["price_num"].dropna() <= 0).sum()
)

print("\nDU4 complete.")
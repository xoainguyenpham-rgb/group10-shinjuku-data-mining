from pathlib import Path
import os
os.chdir(Path(__file__).resolve().parents[1])
def display(value):
    print(value.to_string() if hasattr(value,"to_string") else value)
# 1. SETUP - Locate the shared project data folder from common notebook launch locations.
from pathlib import Path
import pandas as pd

current_dir = Path.cwd().resolve()
project_dir = None
for candidate in (current_dir, *current_dir.parents):
    shared_dir = candidate / "Shared_Preparation_and_Audits"
    if (shared_dir / "raw_data").is_dir() and (shared_dir / "outputs").is_dir():
        project_dir = shared_dir
        break
    if (candidate / "raw_data").is_dir() and (candidate / "outputs").is_dir():
        project_dir = candidate
        break

if project_dir is None:
    raise FileNotFoundError(
        "Could not locate Shared_Preparation_and_Audits/raw_data. "
        "Open this notebook from inside the GitHub project or one of its subfolders."
    )

raw_dir = project_dir / "raw_data"

snapshot_files = {
    "2026_Q1": "Tokyo_listings_2026_Q1.csv",
    "2026_Q2": "Tokyo_listings_2026_Q2.csv",
    "2026_Q3": "Tokyo_listings_2026_Q3.csv",
}
missing_files = [name for name in snapshot_files.values() if not (raw_dir / name).is_file()]
if missing_files:
    raise FileNotFoundError("Missing raw snapshot(s): " + ", ".join(missing_files))

snapshots = {
    period: pd.read_csv(
        raw_dir / filename,
        dtype={"id": "string", "host_id": "string", "scrape_id": "string"},
        encoding="utf-8-sig",
        low_memory=False,
    )
    for period, filename in snapshot_files.items()
}

load_summary = pd.DataFrame([
    {"snapshot": period, "source_file": snapshot_files[period], "rows": len(df), "columns": len(df.columns)}
    for period, df in snapshots.items()
])
display(load_summary)


# 2. TARGET — Tạo nhãn RQ2 trước khi làm sạch biến đầu vào
# Mục đích: bảo toàn toàn bộ listing Shinjuku của kỳ gốc.

# 2A. Kiểm tra ID trên toàn Tokyo trước khi nối dữ liệu.
for period, df in snapshots.items():
    ids = df["id"].astype("string").str.strip()

    if not ids.str.fullmatch(r"[0-9]+").fillna(False).all():
        raise ValueError(f"{period}: có ID thiếu hoặc sai định dạng.")

    if ids.duplicated().any():
        raise ValueError(f"{period}: có ID trùng, cần kiểm tra lại.")


# 2B. Tạo nhãn riêng cho từng chuyển kỳ.
transition_specs = [
    ("2026_Q1", "2026_Q2", "train"),
    ("2026_Q2", "2026_Q3", "test")
]

label_tables = []

for baseline, followup, split in transition_specs:
    baseline_df = snapshots[baseline]
    followup_df = snapshots[followup]

    wards = (
        baseline_df["neighbourhood_cleansed"]
        .astype("string")
        .str.strip()
    )

    # Chọn Shinjuku ở kỳ gốc; chưa loại dòng thiếu biến đầu vào.
    baseline_ids = (
        baseline_df.loc[
            wards.eq("Shinjuku Ku").fillna(False), ["id"]
        ]
        .copy()
        .rename(columns={"id": "listing_id"})
    )
    baseline_ids["listing_id"] = (
        baseline_ids["listing_id"].astype("string").str.strip()
    )

    if baseline_ids.empty:
        raise ValueError(f"{baseline}: không có listing Shinjuku.")

    # Bảng đối chiếu lấy từ toàn Tokyo kỳ sau.
    next_lookup = (
        followup_df[["id", "source"]]
        .copy()
        .rename(columns={
            "id": "listing_id",
            "source": "followup_source"
        })
    )
    next_lookup["listing_id"] = (
        next_lookup["listing_id"].astype("string").str.strip()
    )

    # Phân biệt: có dòng nhưng thiếu source và không tìm thấy dòng.
    next_lookup["followup_source"] = (
        next_lookup["followup_source"]
        .astype("string")
        .str.strip()
        .fillna("")
        .replace("", "(missing)")
    )

    labels = baseline_ids.merge(
        next_lookup,
        on="listing_id",
        how="left",
        validate="one_to_one",
        indicator=True
    )

    # Nhãn dựa trên kết quả nối ID, không dựa trên source.
    labels["present_next"] = (
        labels["_merge"].eq("both").astype("int64")
    )
    labels.loc[
        labels["_merge"].eq("left_only"), "followup_source"
    ] = "(not present)"

    labels["baseline_period"] = baseline
    labels["followup_period"] = followup
    labels["split"] = split

    labels = labels[[
        "listing_id",
        "baseline_period",
        "followup_period",
        "split",
        "present_next",
        "followup_source"
    ]]

    if len(labels) != len(baseline_ids):
        raise ValueError("Số dòng thay đổi ngoài dự kiến sau khi nối.")

    label_tables.append(labels)

rq2_labels = pd.concat(label_tables, ignore_index=True)

# Một listing có thể có mặt ở cả hai kỳ gốc.
# Khóa duy nhất là listing_id + baseline_period, không chỉ listing_id.
if rq2_labels.duplicated(["listing_id", "baseline_period"]).any():
    raise ValueError("Trùng khóa listing_id + baseline_period.")


# 2C. Tóm tắt nhãn để đối chiếu với kết quả DU1.
label_summary = (
    rq2_labels
    .groupby(
        ["split", "baseline_period", "followup_period"],
        sort=False
    )
    .agg(
        baseline_n=("listing_id", "size"),
        present_n=("present_next", "sum")
    )
    .reset_index()
)

label_summary["absent_n"] = (
    label_summary["baseline_n"] - label_summary["present_n"]
)
label_summary["present_pct"] = (
    label_summary["present_n"] / label_summary["baseline_n"] * 100
).round(2)

print("TABLE 2A — RQ2 labels before feature cleaning")
display(label_summary)

print("\nTotal listing-period rows:", len(rq2_labels))

# 3. BASIC CLEANING — Tạo bản làm việc và chuẩn hóa biến cơ bản
# Mục đích: dùng cùng quy tắc cho các bộ dữ liệu của nhóm.
# Không thay đổi snapshots hoặc rq2_labels.

study_wards = ["Shinjuku Ku", "Toshima Ku", "Shibuya Ku", "Taito Ku"]

scope_by_period = {
    "2026_Q1": ["Shinjuku Ku"],
    "2026_Q2": study_wards
}

allowed_room_types = {
    "Entire home/apt",
    "Private room",
    "Shared room",
    "Hotel room"
}

price_pattern = (
    r"\$?\s*-?"
    r"(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)"
    r"(?:\.[0-9]+)?"
)

working_snapshots = {}
basic_cleaning_rows = []

for period, selected_wards in scope_by_period.items():
    raw_df = snapshots[period]
    wards = raw_df["neighbourhood_cleansed"].astype("string").str.strip()

    # 3A. Sao chép đúng phạm vi nghiên cứu, giữ tất cả dòng trong phạm vi.
    work = raw_df.loc[wards.isin(selected_wards)].copy()
    rows_before = len(work)

    work = work.rename(columns={"id": "listing_id"})
    work["listing_id"] = work["listing_id"].astype("string").str.strip()
    work["host_id"] = work["host_id"].astype("string").str.strip()
    work["snapshot"] = period
    work["neighbourhood_cleansed"] = (
        work["neighbourhood_cleansed"].astype("string").str.strip()
    )

    # 3B. Chuẩn hóa nhãn loại phòng, ghi nhận nhãn ngoài danh sách.
    work["room_type"] = (
        work["room_type"].astype("string").str.strip().replace("", pd.NA)
    )
    unexpected_room = (
        work["room_type"].notna()
        & ~work["room_type"].isin(allowed_room_types)
    )

    # 3C. Tạo cột giá số; giữ nguyên cột price để đối chiếu.
    raw_price = work["price"].astype("string").str.strip()
    missing_price = raw_price.isna() | raw_price.eq("").fillna(False)
    valid_format = raw_price.str.fullmatch(price_pattern).fillna(False)

    price_numeric = pd.to_numeric(
        raw_price.where(valid_format)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip(),
        errors="coerce"
    )

    parse_failed = (~missing_price) & price_numeric.isna()
    non_finite = price_numeric.isin([float("inf"), float("-inf")])
    non_positive = price_numeric.le(0).fillna(False)

    # Giá không dùng được trở thành thiếu trong cột mới, không xóa listing.
    work["price_numeric"] = price_numeric.mask(non_finite | non_positive)

    # 3D. Tạo biến phân loại Superhost, giữ nguyên cột gốc.
    superhost_raw = (
        work["host_is_superhost"]
        .astype("string").str.strip().replace("", pd.NA)
    )
    unexpected_superhost = (
        superhost_raw.notna() & ~superhost_raw.isin(["t", "f"])
    )
    work["superhost_status"] = (
        superhost_raw.map({"t": "Yes", "f": "No"}).astype("string")
    )

    # 3E. Kiểm tra không mất dòng hoặc trùng ID.
    if len(work) != rows_before or work["listing_id"].duplicated().any():
        raise ValueError(f"{period}: cần kiểm tra số dòng hoặc ID.")

    working_snapshots[period] = work

    basic_cleaning_rows.append({
        "snapshot": period,
        "rows": len(work),
        "price_missing_raw": int(missing_price.sum()),
        "price_parse_failed": int(parse_failed.sum()),
        "price_non_finite": int(non_finite.sum()),
        "price_non_positive": int(non_positive.sum()),
        "price_missing_clean": int(work["price_numeric"].isna().sum()),
        "unexpected_room_type": int(unexpected_room.sum()),
        "unexpected_superhost": int(unexpected_superhost.sum())
    })

basic_cleaning_audit = pd.DataFrame(basic_cleaning_rows)

# Hiển thị dọc để bảng không bị khuất cột.
print("TABLE 3A — Basic standardisation checks")
display(basic_cleaning_audit.set_index("snapshot").T)

print("\nTABLE 3B — Working data by ward")
working_ward_counts = pd.concat([
    work.groupby("neighbourhood_cleansed").size()
    .rename("n").reset_index().assign(snapshot=period)
    for period, work in working_snapshots.items()
], ignore_index=True)

display(
    working_ward_counts[["snapshot", "neighbourhood_cleansed", "n"]]
)

# 4. BATHROOMS — Đọc mô tả phòng tắm và đối chiếu hai nguồn
# Mục đích: tạo biến có quy tắc rõ ràng, không tự ghi đè dữ liệu gốc.

import re

def parse_bathroom_description(value):
    """Trả về: số phòng tắm, loại sử dụng, trạng thái đọc."""
    if pd.isna(value) or not str(value).strip():
        return (None, "Unspecified", "missing")

    # Chuẩn hóa chữ hoa/thường và khoảng trắng để đọc nhất quán.
    text = re.sub(r"\s+", " ", str(value).strip().lower())

    half_bath_values = {
        "half-bath": "Unspecified",
        "private half-bath": "Private",
        "shared half-bath": "Shared"
    }

    if text in half_bath_values:
        return (0.5, half_bath_values[text], "parsed")

    # Chấp nhận: 1 bath, 1.5 baths, 1 private bath, 2 shared baths...
    match = re.fullmatch(
        r"([0-9]+(?:\.[0-9]+)?) (shared |private )?baths?",
        text
    )

    if match:
        count = float(match.group(1))
        access = (
            match.group(2).strip().title()
            if match.group(2)
            else "Unspecified"
        )
        return (count, access, "parsed")

    # Không suy đoán với mô tả ngoài quy tắc.
    return (None, "Unspecified", "unrecognised")


bathroom_audit_rows = []
bathroom_unrecognised_tables = []
bathroom_disagreement_tables = []

for period, work in working_snapshots.items():
    parsed = pd.DataFrame(
        work["bathrooms_text"].map(parse_bathroom_description).tolist(),
        index=work.index,
        columns=[
            "bathrooms_from_text",
            "bathroom_access",
            "bathroom_parse_status"
        ]
    )

    # Gán lại cột mới giúp chạy lại ô này mà không tạo cột trùng.
    work["bathrooms_from_text"] = pd.to_numeric(
        parsed["bathrooms_from_text"], errors="coerce"
    )
    work["bathroom_access"] = parsed["bathroom_access"].astype("string")
    work["bathroom_parse_status"] = (
        parsed["bathroom_parse_status"].astype("string")
    )

    original_numeric = pd.to_numeric(work["bathrooms"], errors="coerce")
    text_numeric = work["bathrooms_from_text"]

    # Chỉ so sánh khi cả hai nguồn có số hợp lệ, hữu hạn và không âm.
    original_valid = (
        original_numeric.notna()
        & ~original_numeric.isin([float("inf"), float("-inf")])
        & original_numeric.ge(0)
    )
    comparable = original_valid & text_numeric.notna()

    disagreement = (
        comparable
        & original_numeric.sub(text_numeric).abs().gt(0.000001)
    ).fillna(False)

    unrecognised = work["bathroom_parse_status"].eq("unrecognised")

    bathroom_audit_rows.append({
        "snapshot": period,
        "rows": len(work),
        "text_parsed": int(work["bathroom_parse_status"].eq("parsed").sum()),
        "text_missing": int(work["bathroom_parse_status"].eq("missing").sum()),
        "text_unrecognised": int(unrecognised.sum()),
        "numeric_missing": int(original_numeric.isna().sum()),
        "both_comparable": int(comparable.sum()),
        "disagreements": int(disagreement.sum()),
        "numeric_missing_text_available": int(
            (original_numeric.isna() & text_numeric.notna()).sum()
        )
    })

    # Lưu đầy đủ các mô tả chưa đọc được, kèm tần suất.
    unknown_table = (
        work.loc[unrecognised, "bathrooms_text"]
        .value_counts()
        .rename_axis("bathrooms_text")
        .reset_index(name="n")
    )
    unknown_table.insert(0, "snapshot", period)
    bathroom_unrecognised_tables.append(unknown_table)

    # Lưu đầy đủ các dòng bất đồng để kiểm tra; chỉ hiển thị mẫu bên dưới.
    differences = work.loc[
        disagreement,
        ["listing_id", "bathrooms", "bathrooms_text", "bathrooms_from_text"]
    ].copy()
    differences.insert(0, "snapshot", period)
    bathroom_disagreement_tables.append(differences)

bathroom_parsing_audit = pd.DataFrame(bathroom_audit_rows)
bathroom_unrecognised = pd.concat(
    bathroom_unrecognised_tables, ignore_index=True
)
bathroom_disagreements = pd.concat(
    bathroom_disagreement_tables, ignore_index=True
)

print("TABLE 4A — Bathroom parsing and consistency")
display(bathroom_parsing_audit.set_index("snapshot").T)

print("\nTABLE 4B — Unrecognised descriptions")
if bathroom_unrecognised.empty:
    print("No unrecognised descriptions.")
else:
    display(bathroom_unrecognised)

print("\nTABLE 4C — Up to 10 disagreement examples per snapshot")
if bathroom_disagreements.empty:
    print("No disagreements among comparable records.")
else:
    display(
        bathroom_disagreements
        .groupby("snapshot", sort=False)
        .head(10)
        .reset_index(drop=True)
    )

# 4D. RECONCILIATION — Tạo biến phòng tắm dùng chung
# Mục đích: tận dụng thông tin có sẵn nhưng không đoán khi hai nguồn mâu thuẫn.

bathroom_resolution_rows = []

for period, work in working_snapshots.items():
    original = pd.to_numeric(work["bathrooms"], errors="coerce")
    from_text = pd.to_numeric(
        work["bathrooms_from_text"], errors="coerce"
    )

    # Chỉ sử dụng số hữu hạn và không âm.
    original_valid = (
        original.notna()
        & ~original.isin([float("inf"), float("-inf")])
        & original.ge(0)
    ).fillna(False)

    text_valid = (
        from_text.notna()
        & ~from_text.isin([float("inf"), float("-inf")])
        & from_text.ge(0)
    ).fillna(False)

    both_valid = original_valid & text_valid
    conflict = (
        both_valid
        & original.sub(from_text).abs().gt(0.000001)
    ).fillna(False)

    agreement = both_valid & ~conflict
    numeric_only = original_valid & ~text_valid
    text_only = text_valid & ~original_valid

    # Khởi tạo thiếu; chỉ gán khi có cơ sở theo quy tắc.
    reconciled = pd.Series(
        float("nan"), index=work.index, dtype="float64"
    )
    reconciled.loc[agreement | numeric_only] = original.loc[
        agreement | numeric_only
    ]
    reconciled.loc[text_only] = from_text.loc[text_only]

    resolution = pd.Series(
        "unavailable", index=work.index, dtype="string"
    )
    resolution.loc[agreement] = "both_agree"
    resolution.loc[numeric_only] = "numeric_only"
    resolution.loc[text_only] = "text_only"
    resolution.loc[conflict] = "conflict"

    work["bathrooms_clean"] = reconciled
    work["bathroom_conflict"] = conflict.astype("int64")
    work["bathroom_resolution"] = resolution

    # Cờ và nguồn xử lý dùng cho audit, không tự đưa vào predictors.
    bathroom_resolution_rows.append({
        "snapshot": period,
        "rows": len(work),
        "both_agree": int(agreement.sum()),
        "numeric_only": int(numeric_only.sum()),
        "text_only": int(text_only.sum()),
        "conflict": int(conflict.sum()),
        "neither_valid": int((~original_valid & ~text_valid).sum()),
        "bathrooms_clean_missing": int(reconciled.isna().sum())
    })

bathroom_resolution_audit = pd.DataFrame(bathroom_resolution_rows)

print("TABLE 4D — Reconciled bathroom counts")
display(bathroom_resolution_audit.set_index("snapshot").T)

# 5. NUMERIC VARIABLES — Chuẩn hóa số và tạo nhóm so sánh
# Mục đích: thống nhất quy tắc cho các thành viên, giữ cột gốc để truy vết.

# Quy tắc: (nhỏ nhất, lớn nhất, yêu cầu số nguyên).
# None nghĩa là không đặt giới hạn trên.
numeric_rules = {
    "accommodates": (1, None, True),
    "bedrooms": (0, None, True),
    "beds": (0, None, True),
    "minimum_nights": (1, None, True),
    "availability_365": (0, 365, True),
    "number_of_reviews": (0, None, True),
    "reviews_per_month": (0, None, False),
    "review_scores_rating": (0, 5, False),
    "calculated_host_listings_count": (1, None, True)
}

numeric_cleaning_rows = []
comparison_group_tables = []

for period, work in working_snapshots.items():

    # 5A. Tạo cột số sạch, chưa điền giá trị thiếu.
    for column, (lower, upper, integer_required) in numeric_rules.items():
        raw = work[column].astype("string").str.strip()
        missing = raw.isna() | raw.eq("").fillna(False)
        values = pd.to_numeric(raw.mask(missing), errors="coerce")

        parse_failed = (~missing) & values.isna()
        non_finite = values.isin([float("inf"), float("-inf")])
        finite_values = values.mask(non_finite)

        out_of_range = finite_values.lt(lower).fillna(False)
        if upper is not None:
            out_of_range |= finite_values.gt(upper).fillna(False)

        non_integer = pd.Series(False, index=work.index)
        if integer_required:
            non_integer = (
                finite_values.notna()
                & finite_values.mod(1).ne(0).fillna(False)
            )

        invalid = (
            parse_failed | non_finite | out_of_range | non_integer
        )
        clean_column = f"{column}_clean"
        work[clean_column] = values.mask(invalid).astype("Float64")

        numeric_cleaning_rows.append({
            "snapshot": period,
            "variable": column,
            "missing_before": int(missing.sum()),
            "invalid_nonmissing": int(invalid.sum()),
            "missing_after": int(work[clean_column].isna().sum())
        })

    # 5B. Tạo nhóm sức chứa từ giá trị hợp lệ.
    capacity = work["accommodates_clean"]
    work["capacity_group"] = pd.Series(
        pd.NA, index=work.index, dtype="string"
    )
    work.loc[capacity.between(1, 2).fillna(False),
             "capacity_group"] = "1-2"
    work.loc[capacity.between(3, 4).fillna(False),
             "capacity_group"] = "3-4"
    work.loc[capacity.ge(5).fillna(False),
             "capacity_group"] = "5+"

    # 5C. Dùng số listing của host do snapshot cung cấp.
    # Không đếm lại host sau khi lọc quận.
    host_count = work["calculated_host_listings_count_clean"]
    work["host_portfolio_group"] = pd.Series(
        pd.NA, index=work.index, dtype="string"
    )
    work.loc[host_count.eq(1).fillna(False),
             "host_portfolio_group"] = "1"
    work.loc[host_count.between(2, 5).fillna(False),
             "host_portfolio_group"] = "2-5"
    work.loc[host_count.ge(6).fillna(False),
             "host_portfolio_group"] = "6+"

    # 5D. Kiểm tra tần suất nhóm theo từng quận.
    for group_column in ["capacity_group", "host_portfolio_group"]:
        counts = (
            work.assign(
                group_value=work[group_column].fillna("(missing)")
            )
            .groupby(["neighbourhood_cleansed", "group_value"])
            .size()
            .reset_index(name="n")
        )
        counts.insert(0, "group_variable", group_column)
        counts.insert(0, "snapshot", period)
        comparison_group_tables.append(counts)

numeric_cleaning_audit = pd.DataFrame(numeric_cleaning_rows)
comparison_group_audit = pd.concat(
    comparison_group_tables, ignore_index=True
)

print("TABLE 5A — Numeric cleaning checks")
display(numeric_cleaning_audit)

print("\nTABLE 5B — Comparison groups")
with pd.option_context("display.max_rows", 100):
    display(comparison_group_audit)

# ============================================================
# BƯỚC 6 — GHÉP CÁC BẢNG PHÂN TÍCH T1, T2 VÀ T3
# Mục đích: tạo đầu vào riêng cho từng RQ và kiểm tra việc ghép.
# Chưa điền dữ liệu thiếu, chuẩn hóa hoặc chọn biến mô hình.
# ============================================================

# 6.1. Chọn các cột cần giữ trong bảng phân tích
# ID và nhóm so sánh được giữ để đối chiếu, chưa phải biến mô hình.
analysis_columns = [
    "listing_id",
    "host_id",
    "snapshot",
    "neighbourhood_cleansed",
    "room_type",
    "superhost_status",
    "price_numeric",
    "accommodates_clean",
    "bedrooms_clean",
    "beds_clean",
    "bathrooms_clean",
    "minimum_nights_clean",
    "availability_365_clean",
    "number_of_reviews_clean",
    "reviews_per_month_clean",
    "review_scores_rating_clean",
    "calculated_host_listings_count_clean",
    "capacity_group",
    "host_portfolio_group",
]

# Kiểm tra đủ cột trước khi ghép để tránh bỏ sót biến.
for period, work in working_snapshots.items():
    missing_columns = sorted(set(analysis_columns) - set(work.columns))
    assert not missing_columns, (
        f"{period}: thiếu các cột {missing_columns}"
    )

# 6.2. T1 — Dữ liệu Shinjuku Q2 dành cho phân cụm RQ1
q2_work = working_snapshots["2026_Q2"]

T1 = q2_work.loc[
    q2_work["neighbourhood_cleansed"].eq("Shinjuku Ku"),
    analysis_columns,
].copy()

# 6.3. T3 — Dữ liệu Q2 của bốn quận dành cho so sánh RQ3
T3 = q2_work.loc[
    q2_work["neighbourhood_cleansed"].isin(study_wards),
    analysis_columns,
].copy()

# 6.4. Chuẩn bị đặc trưng tại thời điểm baseline cho RQ2
baseline_parts = []

for period in ["2026_Q1", "2026_Q2"]:
    work = working_snapshots[period]

    baseline = work.loc[
        work["neighbourhood_cleansed"].eq("Shinjuku Ku"),
        analysis_columns,
    ].copy()

    baseline = baseline.rename(
        columns={"snapshot": "baseline_period"}
    )
    baseline_parts.append(baseline)

baseline_features = pd.concat(baseline_parts, ignore_index=True)

# Một listing có thể xuất hiện ở cả hai quý.
# Khóa duy nhất phải là listing_id + baseline_period.
join_keys = ["listing_id", "baseline_period"]

assert not baseline_features.duplicated(join_keys).any()
assert not rq2_labels.duplicated(join_keys).any()

# 6.5. Ghép baseline với nhãn đã tạo từ toàn bộ snapshot Tokyo
# Không đưa followup_source vào bảng phân tích.
label_columns = [
    "listing_id",
    "baseline_period",
    "followup_period",
    "split",
    "present_next",
]

T2 = baseline_features.merge(
    rq2_labels[label_columns],
    on=join_keys,
    how="outer",
    validate="one_to_one",
    indicator=True,
)

# Outer join giúp phát hiện cả baseline hoặc nhãn bị thiếu.
assert T2["_merge"].eq("both").all(), (
    "Có baseline hoặc nhãn không ghép được; cần kiểm tra trước khi tiếp tục."
)

T2 = T2.drop(columns="_merge")

T2_train = T2.loc[T2["split"].eq("train")].copy()
T2_test = T2.loc[T2["split"].eq("test")].copy()

# 6.6. Kiểm tra quy mô và nhãn theo kết quả DU1 đã xác nhận
assert len(T1) == 5689
assert len(T3) == 14935
assert len(T2_train) == 4166
assert len(T2_test) == 5689

assert T1["listing_id"].is_unique
assert T3["listing_id"].is_unique
assert T2_train["listing_id"].is_unique
assert T2_test["listing_id"].is_unique

assert T2_train["present_next"].value_counts().to_dict() == {
    1: 3900, 0: 266
}
assert T2_test["present_next"].value_counts().to_dict() == {
    1: 5372, 0: 317
}

assert T2_train["baseline_period"].eq("2026_Q1").all()
assert T2_train["followup_period"].eq("2026_Q2").all()
assert T2_test["baseline_period"].eq("2026_Q2").all()
assert T2_test["followup_period"].eq("2026_Q3").all()

# 6.7. Hiển thị tổng quan để kiểm tra và đưa vào report
dataset_summary = pd.DataFrame([
    {
        "dataset": name,
        "rows": len(table),
        "columns": table.shape[1],
        "unique_listings": table["listing_id"].nunique(),
    }
    for name, table in [
        ("T1_RQ1_Q2_Shinjuku", T1),
        ("T2_RQ2_train_Q1_to_Q2", T2_train),
        ("T2_RQ2_test_Q2_to_Q3", T2_test),
        ("T3_RQ3_Q2_four_wards", T3),
    ]
])

target_summary = (
    T2.groupby(
        ["split", "baseline_period", "followup_period", "present_next"],
        dropna=False,
    )
    .size()
    .reset_index(name="n_listings")
)

display(dataset_summary)
display(target_summary)

print("Đã ghép T1–T2–T3 và vượt qua các kiểm tra quy mô, khóa và nhãn.")

# ============================================================
# BƯỚC 7 — QUY ĐỊNH VAI TRÒ BIẾN
# Mục đích: thống nhất đầu vào phân tích khi bàn giao cho nhóm.
# Không thay đổi giá trị trong T1, T2 hoặc T3.
# ============================================================

# 7.1. Bộ biến phân cụm chính đề xuất cho Toàn
# Các biến về phòng ngủ/giường/phòng tắm dùng để mô tả cụm,
# tránh đưa quá nhiều đại diện cho quy mô chỗ ở vào khoảng cách.
rq1_features = [
    "price_numeric",
    "accommodates_clean",
    "minimum_nights_clean",
    "availability_365_clean",
    "reviews_per_month_clean",
    "calculated_host_listings_count_clean",
]

# 7.2. Bộ biến dự báo baseline cho Hoàng
# Chỉ dùng thông tin có tại quý baseline.
rq2_numeric_features = [
    "price_numeric",
    "accommodates_clean",
    "bedrooms_clean",
    "beds_clean",
    "bathrooms_clean",
    "minimum_nights_clean",
    "availability_365_clean",
    "number_of_reviews_clean",
    "reviews_per_month_clean",
    "review_scores_rating_clean",
    "calculated_host_listings_count_clean",
]

rq2_categorical_features = [
    "room_type",
    "superhost_status",
]

rq2_features = rq2_numeric_features + rq2_categorical_features
rq2_target = "present_next"

# 7.3. Các cột chỉ dùng đối chiếu hoặc quản lý dữ liệu
metadata_columns = {
    "listing_id",
    "host_id",
    "snapshot",
    "baseline_period",
    "followup_period",
    "split",
    "neighbourhood_cleansed",
}

comparison_groups = {
    "capacity_group",
    "host_portfolio_group",
}

# Kiểm tra không vô tình đưa ID, thời gian hoặc target vào mô hình.
assert set(rq1_features).issubset(T1.columns)
assert set(rq2_features).issubset(T2_train.columns)
assert set(rq2_features).issubset(T2_test.columns)

excluded_predictors = metadata_columns | comparison_groups | {
    rq2_target, "followup_source"
}

assert not set(rq1_features) & excluded_predictors
assert not set(rq2_features) & excluded_predictors
assert len(rq2_features) == len(set(rq2_features))

# 7.4. Lập bảng vai trò cho mọi cột trong các bảng bàn giao
all_columns = list(dict.fromkeys(
    list(T1.columns) + list(T2_train.columns) + list(T3.columns)
))

role_records = []

for column in all_columns:
    # Vai trò trong RQ1
    if column not in T1.columns:
        rq1_role = "Not included"
    elif column in metadata_columns:
        rq1_role = "Metadata - exclude from clustering"
    elif column in rq1_features:
        rq1_role = "Primary clustering input"
    else:
        rq1_role = "Cluster profiling only"

    # Vai trò trong RQ2
    if column not in T2_train.columns:
        rq2_role = "Not included"
    elif column == rq2_target:
        rq2_role = "Target"
    elif column in rq2_numeric_features:
        rq2_role = "Numeric predictor"
    elif column in rq2_categorical_features:
        rq2_role = "Categorical predictor"
    else:
        rq2_role = "Metadata/group - exclude from predictors"

    # Vai trò trong RQ3
    if column not in T3.columns:
        rq3_role = "Not included"
    elif column in {
        "neighbourhood_cleansed",
        "room_type",
        "superhost_status",
    } | comparison_groups:
        rq3_role = "Comparison grouping/category"
    elif column in metadata_columns:
        rq3_role = "Metadata - not a numeric measure"
    else:
        rq3_role = "Observed numeric measure"

    role_records.append({
        "variable": column,
        "RQ1_role": rq1_role,
        "RQ2_role": rq2_role,
        "RQ3_role": rq3_role,
    })

variable_roles = pd.DataFrame(role_records)

display(variable_roles)

print(f"RQ1: {len(rq1_features)} biến phân cụm chính đề xuất.")
print(
    f"RQ2: {len(rq2_numeric_features)} biến số + "
    f"{len(rq2_categorical_features)} biến phân loại."
)
print("Đã kiểm tra: ID, thời gian và target không nằm trong predictors.")

# ============================================================
# BƯỚC 8 — XUẤT FILE VÀ KIỂM TRA BỘ DỮ LIỆU BÀN GIAO
# Mục đích: lưu dữ liệu, tài liệu và bằng chứng làm sạch.
# Chỉ ghi vào outputs/DP1_prepared; không sửa dữ liệu raw.
# Chạy lại ô này sẽ cập nhật các file cùng tên.
# ============================================================

# 8.1. Tạo thư mục đầu ra
dp1_output_dir = project_dir / "outputs" / "DP1_prepared"
dp1_output_dir.mkdir(parents=True, exist_ok=True)

# 8.2. Thống kê dữ liệu thiếu trong từng bảng bàn giao
analysis_tables = {
    "T1_RQ1": T1,
    "T2_train": T2_train,
    "T2_test": T2_test,
    "T3_RQ3": T3,
}

missingness_records = []

for dataset_name, table in analysis_tables.items():
    for column in table.columns:
        n_missing = int(table[column].isna().sum())

        missingness_records.append({
            "dataset": dataset_name,
            "variable": column,
            "n_rows": len(table),
            "n_missing": n_missing,
            "missing_pct": round(n_missing / len(table) * 100, 2),
        })

prepared_missingness = pd.DataFrame(missingness_records)

# 8.3. Khai báo các bảng cần xuất
# Nhãn kèm followup_source chỉ nằm trong file audit riêng.
export_tables = {
    "01_T1_RQ1_Q2_Shinjuku.csv": T1,
    "02_T2_RQ2_train_Q1_to_Q2.csv": T2_train,
    "03_T2_RQ2_test_Q2_to_Q3.csv": T2_test,
    "04_T3_RQ3_Q2_four_wards.csv": T3,
    "05_variable_roles.csv": variable_roles,
    "06_dataset_summary.csv": dataset_summary,
    "07_target_summary.csv": target_summary,
    "08_prepared_missingness.csv": prepared_missingness,
    "09_basic_cleaning_audit.csv": basic_cleaning_audit,
    "10_bathroom_parsing_audit.csv": bathroom_parsing_audit,
    "11_bathroom_disagreements.csv": bathroom_disagreements,
    "12_bathroom_resolution_audit.csv": bathroom_resolution_audit,
    "13_numeric_cleaning_audit.csv": numeric_cleaning_audit,
    "14_comparison_group_audit.csv": comparison_group_audit,
    "15_RQ2_label_audit_NOT_predictors.csv": rq2_labels,
}

# 8.4. Xuất CSV và đọc lại dưới dạng chuỗi
# Đọc chuỗi trước giúp bảo toàn ID dài, không chuyển ID thành float.
verification_records = []

for filename, table in export_tables.items():
    file_path = dp1_output_dir / filename

    table.to_csv(
        file_path,
        index=False,
        encoding="utf-8-sig",
        na_rep="",
    )

    restored = pd.read_csv(
        file_path,
        dtype="string",
        encoding="utf-8-sig",
        keep_default_na=False,
    )

    assert list(restored.columns) == list(table.columns), (
        f"{filename}: tên hoặc thứ tự cột không khớp."
    )
    assert len(restored) == len(table), (
        f"{filename}: số dòng không khớp."
    )

    # Kiểm tra từng cột, bao gồm giá trị thiếu.
    for column in table.columns:
        expected = table[column].reset_index(drop=True)
        actual_text = restored[column].reset_index(drop=True)

        if pd.api.types.is_numeric_dtype(expected.dtype):
            actual_numeric = pd.to_numeric(
                actual_text.replace("", pd.NA),
                errors="raise",
            ).astype("Float64")

            pd.testing.assert_series_equal(
                expected.astype("Float64"),
                actual_numeric,
                check_names=False,
                check_exact=False,
                rtol=1e-12,
                atol=1e-12,
            )
        else:
            pd.testing.assert_series_equal(
                expected.astype("string").fillna(""),
                actual_text,
                check_names=False,
                check_dtype=False,
            )

    verification_records.append({
        "file": filename,
        "rows": len(restored),
        "columns": restored.shape[1],
        "verified": True,
    })

# 8.5. Lưu hướng dẫn bàn giao đi cùng dữ liệu
readme_text = """# Group 10 - DP1 handover

## Files and ownership
- 01_T1_RQ1_Q2_Shinjuku.csv: Toan - RQ1 segmentation.
- 02_T2_RQ2_train_Q1_to_Q2.csv: Hoang - RQ2 model development.
- 03_T2_RQ2_test_Q2_to_Q3.csv: Hoang - final temporal evaluation.
- 04_T3_RQ3_Q2_four_wards.csv: Mai Anh / An - RQ3 comparisons.
- 05_variable_roles.csv: column roles for each research question.
- 06-14: summaries and preparation audits.
- 15_RQ2_label_audit_NOT_predictors.csv: label provenance only.

## Data status
These are cleaned analysis tables, not fully preprocessed model inputs.
Missing values are preserved. No imputation, scaling or winsorisation
has been applied. Blank CSV cells represent missing values.

## Import rules
Import listing_id and host_id as text, not numeric variables.
Import capacity_group and host_portfolio_group as categorical text.
Do not open and re-save these CSVs in Excel with automatic type detection:
long IDs and group labels may be changed.
In Orange, explicitly set present_next as a categorical target.
Use 05_variable_roles.csv to select features and metadata.

## RQ1
Use the six proposed clustering inputs documented in variable_roles.
Handle missingness and skewness, then standardise before K-means.
Record preprocessing choices and assess cluster stability.
Profile clusters using observed values in the original units.
Keep listing_id for matching cluster assignments back to T1.

## RQ2
Target: 1 = present in the next Tokyo snapshot; 0 = absent.
Match labels against the full next Tokyo snapshot, not a ward subset.
Fit learned preprocessing within training folds during model selection.
Use the temporal test set only after model and tuning decisions are fixed.
Do not combine train and test for random splitting or preprocessing.
Exclude identifiers, period fields, split, comparison groups and
follow-up source information from predictors.
Evaluate against a majority-class baseline and report minority-class
performance, not accuracy alone.
Listings can recur across baseline periods; this is a temporal evaluation,
not an unseen-listing evaluation.

## RQ3
Use Q2 observed values for all four wards.
Primary comparators: Shibuya and Toshima. Sensitivity comparator: Taito.
Do not impute or winsorise prices for primary descriptive comparisons.
Retain price-missing listings for summaries of other observed variables.
Report valid counts and missingness for every measure.
Portfolio groups use the supplied Tokyo-scoped host listing count.
Group counts describe listings, not distinct hosts.

## Cleaning decisions and limits
Bathroom conflicts are left missing; use audit files 11 and 12 for details.
Missing review frequency or ratings must not automatically become zero.
Availability is not occupancy; review frequency is not booking volume.
Snapshot presence does not establish continued operation or compliance.
Comparisons do not identify causal effects of regulatory changes.
"""

readme_path = dp1_output_dir / "README_DP1.txt"
readme_path.write_text(readme_text, encoding="utf-8")

assert readme_path.read_text(encoding="utf-8") == readme_text

# 8.6. Hiển thị kết quả; chỉ in đường dẫn tương đối
export_verification = pd.DataFrame(verification_records)
display(export_verification)

print(f"Đã xuất và kiểm tra {len(export_tables)} file CSV.")
print("Đã lưu README_DP1.txt.")
print("Thư mục: outputs/DP1_prepared")
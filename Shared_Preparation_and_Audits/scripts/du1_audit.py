from pathlib import Path
import os
os.chdir(Path(__file__).resolve().parents[1])
def display(value):
    print(value.to_string() if hasattr(value,"to_string") else value)
# 1. SETUP - Locate the shared data folder and record the runtime environment.
import sys
from pathlib import Path
import pandas as pd

print("Python version:", sys.version.split()[0])
print("Pandas version:", pd.__version__)

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
output_dir = project_dir / "outputs"

if not raw_dir.is_dir():
    raise FileNotFoundError("The shared raw_data folder is missing.")

print("\nData folder:", raw_dir.relative_to(project_dir))
print("\nAvailable CSV files:")
for file in sorted(raw_dir.glob("*.csv")):
    print("-", file.name)


# 2. LOAD DATA — Đọc ba snapshot chính
# Mục đích: kiểm tra kích thước ban đầu, chưa lọc hoặc làm sạch.

# 2.1. Khai báo file tương ứng với từng kỳ
snapshot_files = {
    "2026_Q1": "Tokyo_listings_2026_Q1.csv",
    "2026_Q2": "Tokyo_listings_2026_Q2.csv",
    "2026_Q3": "Tokyo_listings_2026_Q3.csv",
}

# 2.2. Kiểm tra đủ file trước khi đọc
missing_files = [
    filename
    for filename in snapshot_files.values()
    if not (raw_dir / filename).is_file()
]

if missing_files:
    raise FileNotFoundError(
        "Thiếu file trong raw_data: " + ", ".join(missing_files)
    )

# 2.3. Đọc dữ liệu; giữ các mã định danh ở dạng chuỗi
# Không chuyển ID sang số thực vì có thể làm mất độ chính xác.
snapshots = {}

for period, filename in snapshot_files.items():
    snapshots[period] = pd.read_csv(
        raw_dir / filename,
        dtype={
            "id": "string",
            "host_id": "string",
            "scrape_id": "string",
        },
        encoding="utf-8-sig",
        low_memory=False,
    )

# 2.4. Tạo bảng tổng hợp kích thước dữ liệu gốc
snapshot_summary = pd.DataFrame([
    {
        "snapshot": period,
        "source_file": snapshot_files[period],
        "rows": df.shape[0],
        "columns": df.shape[1],
    }
    for period, df in snapshots.items()
])

display(snapshot_summary)

# 3. AUDIT — Kiểm tra ngày thu thập và tính toàn vẹn của ID
# Mục đích: phát hiện vấn đề trước khi ghép dữ liệu giữa các kỳ.

audit_rows = []

for period, df in snapshots.items():

    # 3.1. Kiểm tra các cột bắt buộc
    required_columns = {"id", "last_scraped"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"{period}: thiếu cột {sorted(missing_columns)}"
        )

    # 3.2. Kiểm tra ID trên bản tạm; không sửa bảng dữ liệu gốc
    ids = df["id"].astype("string")
    stripped_ids = ids.str.strip()

    missing_id = stripped_ids.isna() | stripped_ids.eq("").fillna(False)
    nonmissing_ids = stripped_ids[~missing_id]

    # ID listing cần là chuỗi chữ số; phát hiện dạng lỗi như 123.0 hoặc 1e+18
    invalid_id_format = ~nonmissing_ids.str.fullmatch(r"[0-9]+")

    # Đếm ID có khoảng trắng ở đầu/cuối để ghi nhận việc cần chuẩn hóa
    id_whitespace = ids.ne(stripped_ids).fillna(False)

    # 3.3. Kiểm tra ngày: phân biệt giá trị thiếu và giá trị không đọc được
    raw_dates = df["last_scraped"].astype("string").str.strip()
    missing_date = raw_dates.isna() | raw_dates.eq("").fillna(False)

    parsed_dates = pd.to_datetime(
        raw_dates.mask(missing_date),
        format="%Y-%m-%d",
        errors="coerce",
    )

    invalid_date = (~missing_date) & parsed_dates.isna()

    # 3.4. Ghi kết quả kiểm toán cho từng snapshot
    audit_rows.append({
        "snapshot": period,
        "rows": len(df),
        "first_scraped": (
            parsed_dates.min().date()
            if parsed_dates.notna().any() else None
        ),
        "last_scraped": (
            parsed_dates.max().date()
            if parsed_dates.notna().any() else None
        ),
        "missing_dates": int(missing_date.sum()),
        "invalid_dates": int(invalid_date.sum()),
        "missing_ids": int(missing_id.sum()),
        "invalid_id_format": int(invalid_id_format.sum()),
        "ids_with_whitespace": int(id_whitespace.sum()),
        "unique_nonmissing_ids": nonmissing_ids.nunique(),
        "duplicate_id_rows": int(
            nonmissing_ids.duplicated(keep="first").sum()
        ),
    })

id_date_audit = pd.DataFrame(audit_rows)

# 3.5. Hiển thị thành hai bảng để dễ đọc trên màn hình
print("TABLE 3A — Collection dates")
display(id_date_audit[[
    "snapshot", "rows", "first_scraped", "last_scraped",
    "missing_dates", "invalid_dates",
]])

print("TABLE 3B — Listing ID integrity")
display(id_date_audit[[
    "snapshot", "missing_ids", "invalid_id_format",
    "ids_with_whitespace", "unique_nonmissing_ids",
    "duplicate_id_rows",
]])

# 4. COVERAGE — Kiểm tra khu vực và nguồn thu thập
# Mục đích: phát hiện khác biệt giữa các kỳ trước khi tính persistence.

# 4.1. Khai báo các quận đã thống nhất trong DU4
study_wards = [
    "Shinjuku Ku",
    "Toshima Ku",
    "Shibuya Ku",
    "Taito Ku",
]

ward_rows = []
source_tables = []

for period, df in snapshots.items():

    # 4.2. Kiểm tra các cột cần dùng
    required_columns = {"neighbourhood_cleansed", "source"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"{period}: thiếu cột {sorted(missing_columns)}"
        )

    # 4.3. Đếm listing theo quận trên toàn bộ dữ liệu gốc
    # Chuẩn hóa khoảng trắng trên biến tạm, không sửa snapshots.
    wards = df["neighbourhood_cleansed"].astype("string").str.strip()

    row = {
        "snapshot": period,
        "Tokyo_total": len(df),
        "missing_ward": int(
            (wards.isna() | wards.eq("").fillna(False)).sum()
        ),
    }

    for ward in study_wards:
        row[ward] = int(wards.eq(ward).sum())

    ward_rows.append(row)

    # 4.4. Thống kê source cho Tokyo và riêng Shinjuku
    # Giữ nhóm thiếu thông tin; không tự gán thành city scrape.
    sources = df["source"].astype("string").str.strip()
    sources = sources.fillna("").replace("", "(missing)")

    scopes = {
        "Tokyo": sources,
        "Shinjuku": sources[wards.eq("Shinjuku Ku").fillna(False)],
    }

    for scope, values in scopes.items():
        counts = values.value_counts(dropna=False)

        table = counts.rename_axis("source").reset_index(name="n")
        table.insert(0, "scope", scope)
        table.insert(0, "snapshot", period)

        # Mẫu số là tổng listing của phạm vi tương ứng trong kỳ đó
        table["share_pct"] = (
            table["n"] / len(values) * 100
        ).round(2)

        source_tables.append(table)

# 4.5. Tổng hợp và hiển thị kết quả
ward_coverage = pd.DataFrame(ward_rows)
source_audit = pd.concat(source_tables, ignore_index=True)

print("TABLE 4A — Geographic coverage")
display(ward_coverage)

print("TABLE 4B — Collection sources: Tokyo")
display(
    source_audit[source_audit["scope"].eq("Tokyo")]
    .reset_index(drop=True)
)

print("TABLE 4C — Collection sources: Shinjuku")
display(
    source_audit[source_audit["scope"].eq("Shinjuku")]
    .reset_index(drop=True)
)

# 5. SCHEMA & MISSINGNESS — Kiểm tra cấu trúc và giá trị thiếu
# Mục đích: xác định vấn đề cần xử lý ở DP1, chưa sửa dữ liệu gốc.

# 5A. So sánh tên cột của từng kỳ với Q1
reference_columns = set(snapshots["2026_Q1"].columns)
schema_rows = []

for period, df in snapshots.items():
    current_columns = set(df.columns)

    schema_rows.append({
        "snapshot": period,
        "n_columns": len(df.columns),
        "same_columns_as_Q1": current_columns == reference_columns,
        "missing_vs_Q1": ", ".join(
            sorted(reference_columns - current_columns)
        ) or "(none)",
        "extra_vs_Q1": ", ".join(
            sorted(current_columns - reference_columns)
        ) or "(none)"
    })

schema_audit = pd.DataFrame(schema_rows)

print("TABLE 5A — Column consistency")
display(schema_audit)


# 5B. Kiểm tra các biến quan trọng trong phạm vi Shinjuku
# Danh sách này phục vụ audit, chưa phải bộ biến đầu vào cuối cùng.
audit_columns = [
    "id",
    "host_id",
    "room_type",
    "accommodates",
    "bedrooms",
    "beds",
    "bathrooms",
    "bathrooms_text",
    "price",
    "minimum_nights",
    "availability_365",
    "number_of_reviews",
    "reviews_per_month",
    "review_scores_rating",
    "host_is_superhost",
    "calculated_host_listings_count",
    "latitude",
    "longitude",
    "license",
    "source"
]

missingness_rows = []

for period, df in snapshots.items():
    wards = df["neighbourhood_cleansed"].astype("string").str.strip()
    shinjuku = df.loc[wards.eq("Shinjuku Ku").fillna(False)]

    if shinjuku.empty:
        raise ValueError(f"{period}: không tìm thấy listing Shinjuku.")

    for column in audit_columns:
        column_present = column in shinjuku.columns

        if column_present:
            values = shinjuku[column]

            # Tính cả NA và chuỗi rỗng/chỉ chứa khoảng trắng là thiếu.
            blank_values = (
                values.astype("string")
                .str.strip()
                .eq("")
                .fillna(False)
            )
            missing_values = values.isna() | blank_values

            missing_n = int(missing_values.sum())
            missing_pct = round(missing_n / len(shinjuku) * 100, 2)
            imported_dtype = str(values.dtype)

        else:
            # Phân biệt "không có cột" với "có cột nhưng thiếu dữ liệu".
            missing_n = pd.NA
            missing_pct = pd.NA
            imported_dtype = "(column absent)"

        missingness_rows.append({
            "snapshot": period,
            "variable": column,
            "column_present": column_present,
            "imported_dtype": imported_dtype,
            "n_listings": len(shinjuku),
            "missing_n": missing_n,
            "missing_pct": missing_pct
        })

missingness_audit = pd.DataFrame(missingness_rows)

# Hiển thị đủ số dòng trong phạm vi các bảng của bước này.
with pd.option_context("display.max_rows", 100):
    print("\nTABLE 5B — Missing values in Shinjuku (%)")
    display(
        missingness_audit.pivot(
            index="variable",
            columns="snapshot",
            values="missing_pct"
        ).reindex(audit_columns)
    )

    print("\nTABLE 5C — Imported data types")
    display(
        missingness_audit.pivot(
            index="variable",
            columns="snapshot",
            values="imported_dtype"
        ).reindex(audit_columns)
    )

# 6. PRICE AUDIT — Kiểm tra định dạng và giá trị giá
# Mục đích: phân biệt giá bị thiếu với giá có nội dung nhưng không đọc được.

price_rows = []
price_example_rows = []
price_issue_rows = []

# Chấp nhận số thông thường hoặc số có dấu phẩy phân tách hàng nghìn.
# Cho phép ký hiệu $ ở đầu; không tự xóa các ký tự lạ khác.
price_pattern = (
    r"\$?\s*-?"
    r"(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)"
    r"(?:\.[0-9]+)?"
)

for period, df in snapshots.items():
    wards = df["neighbourhood_cleansed"].astype("string").str.strip()
    shinjuku = df.loc[wards.eq("Shinjuku Ku").fillna(False)]

    raw_price = shinjuku["price"].astype("string").str.strip()
    missing = raw_price.isna() | raw_price.eq("").fillna(False)

    # Chỉ chuyển các chuỗi đúng định dạng đã quy định.
    format_ok = raw_price.str.fullmatch(price_pattern).fillna(False)

    cleaned_price = (
        raw_price.where(format_ok)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )

    numeric_price = pd.to_numeric(cleaned_price, errors="coerce")

    # Giá có nội dung nhưng không chuyển được phải được ghi nhận riêng.
    parse_failed = (~missing) & numeric_price.isna()
    non_positive = numeric_price.le(0).fillna(False)
    positive_prices = numeric_price.loc[
        numeric_price.gt(0).fillna(False)
    ]

    price_rows.append({
        "snapshot": period,
        "n_listings": len(shinjuku),
        "missing_price": int(missing.sum()),
        "parse_failed": int(parse_failed.sum()),
        "non_positive": int(non_positive.sum()),
        "positive_price_n": len(positive_prices),
        "min": positive_prices.min(),
        "p25": positive_prices.quantile(0.25),
        "median": positive_prices.median(),
        "p75": positive_prices.quantile(0.75),
        "max": positive_prices.max()
    })

    # Lưu một số ví dụ để nhìn rõ định dạng gốc và kết quả chuyển đổi.
    example_indices = raw_price.loc[~missing].head(3).index

    for idx in example_indices:
        price_example_rows.append({
            "snapshot": period,
            "raw_price": raw_price.loc[idx],
            "parsed_price": numeric_price.loc[idx]
        })

    # Hiển thị tối đa 5 ví dụ lỗi cho mỗi kỳ, nếu có.
    issue_indices = shinjuku.index[
        parse_failed | non_positive
    ][:5]

    for idx in issue_indices:
        price_issue_rows.append({
            "snapshot": period,
            "listing_id": shinjuku.loc[idx, "id"],
            "raw_price": raw_price.loc[idx],
            "issue": (
                "parse_failed"
                if parse_failed.loc[idx]
                else "non_positive"
            )
        })

price_audit = pd.DataFrame(price_rows)
price_examples = pd.DataFrame(price_example_rows)
price_issues = pd.DataFrame(
    price_issue_rows,
    columns=["snapshot", "listing_id", "raw_price", "issue"]
)

print("TABLE 6A — Price format examples")
display(price_examples)

print("\nTABLE 6B — Price validity and positive-price summary")
display(price_audit.round(2))

print("\nTABLE 6C — Examples requiring review")
if price_issues.empty:
    print("No parsing failures or non-positive prices detected.")
else:
    display(price_issues)

# 7. ID MATCHING — Kiểm tra listing xuất hiện ở kỳ tiếp theo
# Mục đích: xác định nhãn RQ2 trước khi xử lý thiếu hoặc loại biến.

# 7A. Lưu tập ID toàn Tokyo và tập ID Shinjuku của từng kỳ.
tokyo_id_sets = {}
shinjuku_id_sets = {}

for period, df in snapshots.items():
    ids = df["id"].astype("string").str.strip()
    wards = df["neighbourhood_cleansed"].astype("string").str.strip()

    # Dừng nếu ID không hợp lệ để tránh tạo nhãn sai.
    valid_ids = ids.str.fullmatch(r"[0-9]+").fillna(False)
    if not valid_ids.all() or ids.duplicated().any():
        raise ValueError(f"{period}: cần kiểm tra lại ID trước khi nối.")

    tokyo_id_sets[period] = set(ids)
    shinjuku_id_sets[period] = set(
        ids.loc[wards.eq("Shinjuku Ku").fillna(False)]
    )

# 7B. So ID Shinjuku kỳ đầu với toàn Tokyo kỳ sau.
transition_rows = []

for baseline, followup in [
    ("2026_Q1", "2026_Q2"),
    ("2026_Q2", "2026_Q3")
]:
    baseline_ids = shinjuku_id_sets[baseline]
    next_ids = tokyo_id_sets[followup]

    present_ids = baseline_ids & next_ids
    absent_ids = baseline_ids - next_ids

    transition_rows.append({
        "transition": f"{baseline} → {followup}",
        "baseline_Shinjuku": len(baseline_ids),
        "present_next": len(present_ids),
        "absent_next": len(absent_ids),
        "present_pct": round(
            len(present_ids) / len(baseline_ids) * 100, 2
        ),
        "absent_pct": round(
            len(absent_ids) / len(baseline_ids) * 100, 2
        )
    })

persistence_audit = pd.DataFrame(transition_rows)

print("TABLE 7A — Next-snapshot ID presence")
display(persistence_audit)

# 7C. Kiểm tra trường hợp vắng ở Q2 nhưng xuất hiện lại ở Q3.
# Đây là lý do không gọi nhãn vắng mặt là "rời thị trường vĩnh viễn".
q1_absent_q2 = (
    shinjuku_id_sets["2026_Q1"] - tokyo_id_sets["2026_Q2"]
)
q1_reappeared_q3 = q1_absent_q2 & tokyo_id_sets["2026_Q3"]

reappearance_audit = pd.DataFrame([{
    "Q1_Shinjuku_absent_Q2": len(q1_absent_q2),
    "present_again_Q3": len(q1_reappeared_q3),
    "still_absent_Q3": len(q1_absent_q2 - tokyo_id_sets["2026_Q3"])
}])

print("\nTABLE 7B — Reappearance check")
display(reappearance_audit)

# 8. MATCHED SOURCES — Kiểm tra nguồn của listing nối được
# Mục đích: hiểu nhãn "có xuất hiện" đang dựa trên các nguồn nào.
# Source của kỳ sau chỉ phục vụ audit, không dùng làm biến dự đoán.

matched_source_tables = []

for baseline, followup in [
    ("2026_Q1", "2026_Q2"),
    ("2026_Q2", "2026_Q3")
]:
    baseline_ids = shinjuku_id_sets[baseline]
    next_df = snapshots[followup]

    next_ids = next_df["id"].astype("string").str.strip()

    # Tìm trong toàn Tokyo, không lọc lại quận ở kỳ sau.
    matched = next_df.loc[next_ids.isin(baseline_ids)]
    expected_matches = len(
        baseline_ids & tokyo_id_sets[followup]
    )

    # Kiểm tra số dòng nối được khớp với bước 7.
    if len(matched) != expected_matches:
        raise ValueError(
            f"{baseline} → {followup}: số dòng nối không khớp bước 7."
        )

    sources = (
        matched["source"]
        .astype("string")
        .str.strip()
        .fillna("")
        .replace("", "(missing)")
    )

    table = (
        sources.value_counts(dropna=False)
        .rename_axis("followup_source")
        .reset_index(name="matched_n")
    )

    table.insert(0, "transition", f"{baseline} → {followup}")
    table["share_of_matched_pct"] = (
        table["matched_n"] / len(matched) * 100
    ).round(2)

    matched_source_tables.append(table)

matched_source_audit = pd.concat(
    matched_source_tables,
    ignore_index=True
)

print("TABLE 8A — Follow-up sources among matched listings")
display(matched_source_audit)

# 9. VALUE CHECKS — Kiểm tra phạm vi số và nhãn phân loại
# Mục đích: phát hiện giá trị cần xem lại trước khi chuẩn bị dữ liệu.

# Quy tắc: (giá trị nhỏ nhất, lớn nhất, có yêu cầu số nguyên không).
# None nghĩa là chưa đặt giới hạn trên.
numeric_rules = {
    "accommodates": (1, None, True),
    "bedrooms": (0, None, True),
    "beds": (0, None, True),
    "bathrooms": (0, None, False),
    "minimum_nights": (1, None, True),
    "availability_365": (0, 365, True),
    "number_of_reviews": (0, None, True),
    "reviews_per_month": (0, None, False),
    "review_scores_rating": (0, 5, False),
    "calculated_host_listings_count": (1, None, True)
}

numeric_rows = []
category_tables = []

for period, df in snapshots.items():
    wards = df["neighbourhood_cleansed"].astype("string").str.strip()
    shinjuku = df.loc[wards.eq("Shinjuku Ku").fillna(False)]

    # 9A. Kiểm tra dữ liệu số; giữ riêng lỗi đọc số và giá trị thiếu.
    for column, (lower, upper, integer_required) in numeric_rules.items():
        raw = shinjuku[column].astype("string").str.strip()
        missing = raw.isna() | raw.eq("").fillna(False)
        values = pd.to_numeric(raw.mask(missing), errors="coerce")

        parse_failed = (~missing) & values.isna()
        non_finite = values.isin([float("inf"), float("-inf")])
        finite_values = values.mask(non_finite)

        out_of_range = finite_values.lt(lower).fillna(False)
        if upper is not None:
            out_of_range |= finite_values.gt(upper).fillna(False)

        non_integer = pd.Series(False, index=shinjuku.index)
        if integer_required:
            non_integer = (
                finite_values.notna()
                & finite_values.mod(1).ne(0).fillna(False)
            )

        numeric_rows.append({
            "snapshot": period,
            "variable": column,
            "missing_n": int(missing.sum()),
            "parse_failed": int(parse_failed.sum()),
            "non_finite": int(non_finite.sum()),
            "out_of_range": int(out_of_range.sum()),
            "non_integer": int(non_integer.sum()),
            "min": finite_values.min(),
            "max": finite_values.max()
        })

    # 9B. Đếm nhãn thực tế, chưa mã hóa hoặc suy diễn ý nghĩa.
    for column in ["room_type", "host_is_superhost", "bathrooms_text"]:
        labels = (
            shinjuku[column].astype("string").str.strip()
            .fillna("").replace("", "(missing)")
        )
        table = (
            labels.value_counts(dropna=False)
            .rename_axis("value")
            .reset_index(name="n")
        )
        table.insert(0, "variable", column)
        table.insert(0, "snapshot", period)
        category_tables.append(table)

numeric_value_audit = pd.DataFrame(numeric_rows)
categorical_value_audit = pd.concat(category_tables, ignore_index=True)

# Chia bảng theo kỳ để dễ đọc và chụp kết quả.
with pd.option_context("display.max_rows", 100):
    for period in snapshots:
        print(f"\nTABLE 9A — Numeric checks: {period}")
        display(
            numeric_value_audit.loc[
                numeric_value_audit["snapshot"].eq(period)
            ].drop(columns="snapshot").reset_index(drop=True)
        )

    print("\nTABLE 9B — Room type and superhost values")
    display(
        categorical_value_audit.loc[
            categorical_value_audit["variable"].isin(
                ["room_type", "host_is_superhost"]
            )
        ].reset_index(drop=True)
    )

    print("\nTABLE 9C — Ten most frequent bathroom descriptions per snapshot")
    display(
        categorical_value_audit.loc[
            categorical_value_audit["variable"].eq("bathrooms_text")
        ].groupby("snapshot", sort=False).head(10).reset_index(drop=True)
    )

# 10. EXPORT — Lưu bảng audit để nhóm dùng trong report và DP1
# Chạy lại ô này sẽ cập nhật các CSV audit cùng tên.
# Không ghi đè bất kỳ file dữ liệu gốc nào.

audit_dir = output_dir / "DU1_audit"
audit_dir.mkdir(parents=True, exist_ok=True)

audit_tables = {
    "01_snapshot_summary.csv": snapshot_summary,
    "02_id_date_audit.csv": id_date_audit,
    "03_ward_coverage.csv": ward_coverage,
    "04_source_audit.csv": source_audit,
    "05_schema_audit.csv": schema_audit,
    "06_missingness_audit.csv": missingness_audit,
    "07_price_audit.csv": price_audit,
    "08_price_examples.csv": price_examples,
    "09_price_issues.csv": price_issues,
    "10_persistence_audit.csv": persistence_audit,
    "11_reappearance_audit.csv": reappearance_audit,
    "12_matched_source_audit.csv": matched_source_audit,
    "13_numeric_value_audit.csv": numeric_value_audit,
    "14_categorical_value_audit.csv": categorical_value_audit
}

export_rows = []

for filename, table in audit_tables.items():
    destination = audit_dir / filename

    # UTF-8 BOM giúp Excel đọc tiếng Việt và ký tự đặc biệt.
    table.to_csv(destination, index=False, encoding="utf-8-sig")

    # Kiểm tra file xuất có đúng tên cột và số dòng.
    check = pd.read_csv(destination, encoding="utf-8-sig")
    if list(check.columns) != list(table.columns) or len(check) != len(table):
        raise ValueError(f"File xuất chưa khớp bảng gốc: {filename}")

    export_rows.append({
        "file": filename,
        "rows": len(table),
        "columns": len(table.columns),
        "verified": True
    })

export_manifest = pd.DataFrame(export_rows)

# Chỉ in đường dẫn tương đối, không hiển thị tên tài khoản máy.
print("Saved folder:", audit_dir.relative_to(project_dir))
display(export_manifest)
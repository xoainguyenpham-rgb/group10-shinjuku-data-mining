# DP4 — Chuẩn bị dữ liệu so sánh RQ3

**Người phụ trách: Mai Anh · Group 10 · Shinjuku Airbnb**

Bộ này hoàn thành phần chuẩn bị DP4 theo plan cập nhật: so sánh Shinjuku với Shibuya và Toshima, dùng Taito để kiểm tra độ nhạy. Đầu vào, code, notebook, bảng xuất và kiểm tra đều có trong folder. Không cần chạy lại DP1–DP3 để dùng bộ này.

## 1. Bắt đầu trong VS Code

1. Giải nén ZIP; mở đúng folder `DP4_MaiAnh` bằng **File → Open Folder**.
2. Mở `notebooks/04_DP4_Comparison_Preparation.ipynb`.
3. Bấm **Select Kernel** và chọn môi trường Python đã dùng cho DP1. Nếu VS Code yêu cầu, dùng extensions Python/Jupyter và cài ipykernel cho môi trường được chọn.
4. Chạy lần lượt từ đầu; khi đã hiểu có thể **Run All**. Notebook tự tìm root trong folder hiện tại hoặc folder cha; không in đường dẫn máy hay username.
5. Xem phần QA cuối notebook: `status = PASS`, 14.935 dòng, không tăng dòng, 0 nhãn Shinjuku khác DP3.
6. Mở `outputs/DP4_ward_summary.csv` để xem tổng hợp; file bàn giao chính là `outputs/T3_compare.csv`.

Nếu chạy bằng Terminal, đứng tại folder `DP4_MaiAnh`:

```bash
python3 scripts/dp4_prepare.py
python3 -m unittest discover -s tests -v
```

Nếu môi trường thiếu thư viện: `python3 -m pip install -r requirements.txt`. Chọn đúng môi trường tương ứng ở notebook. Đây là bước cài trên máy cậu khi cần; bộ file không tự cài gì.

## 2. Đọc file nào, dùng để làm gì?

| File trong outputs | Mục đích |
|---|---|
| T3_compare.csv | Bảng từng listing: giữ 19 biến DP1, thêm metadata, missing flags, nhãn quan sát và cột chờ kết quả mô hình |
| DP4_ward_summary.csv | Cỡ mẫu từng quận, số giá quan sát, median/IQR và mẫu số presence |
| DP4_group_sample_sizes.csv | Đủ 48 ô: 4 quận × 4 room type × 3 nhóm sức chứa, kể cả ô rỗng |
| DP4_pairwise_support.csv | 36 ô đối chiếu Shinjuku với từng comparator; kiểm tra hai bên có đủ mẫu |
| DP4_structure_counts.csv | Số listing và tỷ trọng theo room type, sức chứa, quy mô portfolio; không phải tỷ trọng số host |
| DP4_missingness.csv | Số lượng/tỷ lệ thiếu của 11 biến số trong từng quận |
| DP4_join_keys.csv | ID, snapshot, kỳ follow-up và phạm vi có thể nhận kết quả mô hình |
| DP4_join_audit.csv | Tổng số dòng theo trạng thái ghép; kết quả chưa gửi được phân biệt với giá trị 0 |
| DP4_data_dictionary.csv | Định nghĩa từng biến trong T3_compare, đơn vị và vai trò |
| DP4_QA.json | Kết quả kiểm tra của lần chạy; trạng thái pending mô hình vẫn hiển thị |
| DP4_manifest.csv | SHA-256 của code, đầu vào và output để đối chiếu phiên bản |

`docs/DP4_Methods_Report_EN.md` là phần phương pháp tiếng Anh để biên tập vào report. `docs/DP4_Handover_VI.md` hướng dẫn An/Toàn dùng dữ liệu làm M4. `inputs/source_provenance.json` ghi nguồn và ngày scrape thực tế.

## 3. Những quy tắc cần giữ nguyên

- Đơn vị quan sát: **một listing tại Q2**, khóa `listing_id + snapshot`. Giữ ID là text khi đọc CSV; không mở rồi lưu ID dài bằng Excel theo kiểu number.
- Không loại listing thiếu giá khỏi bảng chung. Giá median/IQR chỉ tính trên giá quan sát, không impute/winsorize/log/scale. Giá là JPY/night theo dữ liệu DP1, không phải doanh thu.
- Cơ cấu listing và nhãn quan sát dùng toàn bộ mẫu của quận/nhóm, kể cả dòng thiếu giá.
- `room_type × capacity_group` tạo các nhóm tương đồng thô; không bảo đảm listing giống nhau về vị trí, chất lượng hay tiện ích.
- `calculated_host_listings_count_clean` giữ nguyên phạm vi Tokyo của DP1. Không tính lại theo riêng quận rồi gọi là cùng một biến.
- `present_next_observed`: ID có mặt trong toàn bộ Tokyo Q3 = 1; không có = 0. Không diễn giải thành hoạt động, sống sót, đóng cửa hoặc tuân thủ luật. Đây là outcome quan sát, **không đưa vào predictor**.
- Không dùng các bảng outcome Q3 để lựa chọn/chỉnh mô hình RQ2. Người làm RQ2 phải chốt mô hình bằng train Q1→Q2 trước khi đánh giá test Q2→Q3.
- Ngưỡng `min_n=30` chỉ là cờ lưu ý mô tả. Không phải bảo đảm ý nghĩa thống kê; không tự động loại dòng. Với giá, kiểm tra **n_price của cả hai quận**.
- Phần trăm trong CSV mang thang 0–100 (94.4 nghĩa là 94,4%). Khi so sánh hai tỷ lệ, dùng chênh lệch điểm phần trăm. Không lấy trung bình các median nhóm để gọi là median tổng.
- `n_unique_hosts` không cộng giữa nhóm/quận vì cùng host có thể xuất hiện nhiều nơi.

## 4. Ghép kết quả mô hình khi Toàn/Hoàng bàn giao

**Chưa cần kết quả này để hoàn thành DP4 chuẩn bị.** Cột kết quả hiện trống có chủ đích. Không gọi toàn bộ kết nối RQ1–RQ2–RQ3 là hoàn thành trước khi ghép kết quả cuối.

Toàn xuất `rq1_final_clusters.csv` với đúng ba cột:

```text
listing_id,snapshot,rq1_cluster
```

Mỗi ID một dòng, `snapshot=2026_Q2`; chỉ cohort Shinjuku có giá được dùng trong DP2. Dự kiến đủ 5.328 nhãn. Nhãn cụm có thể bắt đầu từ 0; **0 là nhãn thật nếu có, không phải ký hiệu thiếu**.

Hoàng xuất `rq2_final_test_predictions.csv` với đúng bốn cột:

```text
listing_id,baseline_period,followup_period,rq2_pred_present_probability
```

`baseline_period=2026_Q2`, `followup_period=2026_Q3`; xác suất thuộc lớp `present_next=1`, trong [0,1]. Dự kiến 5.689 dòng Shinjuku. File này phải từ mô hình đã khóa; không dùng fitted probabilities trên training data.

Đặt hai file vào `inputs/`, sau đó chạy:

```bash
python3 scripts/dp4_prepare.py --clusters inputs/rq1_final_clusters.csv --predictions inputs/rq2_final_test_predictions.csv --output-dir outputs_integrated
```

Có thể chỉ truyền một trong hai file. Code chặn ID trùng, ID ngoài phạm vi, sai kỳ, kết quả thiếu và xác suất ngoài [0,1]. File bàn giao thiếu một phần ID vẫn ghép được nhưng trạng thái là `not_in_supplied_results`; kiểm tra các số unresolved trong QA trước khi công bố kết quả.

- 361 listing Shinjuku thiếu giá: `excluded_missing_price`, không có nhãn cụm.
- Các comparator: `outside_rq1_scope` và `outside_rq2_scope`; không tự gán nhãn hoặc xác suất cho họ.
- Không áp dụng mô hình Shinjuku sang comparator nếu chưa có thiết kế/kiểm định bổ sung.
- Ghép prediction là bước kỹ thuật; code không thể tự xác minh mô hình có thực sự được khóa hay chưa. Người bàn giao chịu trách nhiệm về provenance của model.

## 5. Kiểm tra độ nhạy của cờ nhóm nhỏ

Không thay đổi output chính khi thử ngưỡng khác:

```bash
python3 scripts/dp4_prepare.py --min-n 20 --output-dir outputs_n20
python3 scripts/dp4_prepare.py --min-n 50 --output-dir outputs_n50
```

Các lệnh chỉ đổi cờ cỡ mẫu, không thay đổi dòng hoặc giá trị quan sát. Kiểm tra độ nhạy thực tế của phát hiện Q3 thuộc M4 do An/Toàn thực hiện.

## 6. Tình trạng bàn giao

Đã tạo và chạy script; thực thi tuần tự các code cell của notebook bằng Python; qua 9 kiểm tra tự động, gồm các ca ghép sai. Notebook chưa được chạy trên kernel VS Code/Mac của cậu. `docs/DP4_Validation.md` ghi cụ thể phạm vi kiểm tra.

Gửi cả folder hoặc ZIP cho nhóm, không gửi riêng notebook vì notebook dùng script và inputs đi kèm. Không đổi dữ liệu đã gửi bằng cách sửa CSV tay; sửa code, chạy lại, rồi dùng bộ outputs cùng một lần chạy.

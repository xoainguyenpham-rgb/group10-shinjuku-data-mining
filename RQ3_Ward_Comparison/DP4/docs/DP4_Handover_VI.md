# Bàn giao DP4 cho An và Toàn

DP4 chuẩn bị dữ liệu cho câu hỏi: **How does Shinjuku's market structure differ from other Tokyo areas under the announced regulatory changes?**

Dữ liệu cùng snapshot Q2, cùng định nghĩa biến DP1. Shibuya/Toshima là comparator chính; Taito là comparator kiểm tra độ nhạy. Quy định được dùng làm bối cảnh diễn giải; DP4 không tạo biến “bị ảnh hưởng pháp lý” hoặc “vi phạm”. Phần nguồn pháp lý của An cần được xác minh riêng trước khi viết kết luận.

## Thứ tự làm M4

1. Đọc `DP4_ward_summary`: n_listings, n_price và tỷ lệ thiếu giá. Không gọi median thô là premium đã kiểm soát khác biệt listing.
2. Đọc `DP4_structure_counts`: so sánh cơ cấu room type, sức chứa và portfolio. Mẫu số là toàn bộ listing của từng quận, không phải tổng bốn quận.
3. Dùng `DP4_group_sample_sizes` để so sánh trong cùng room_type và capacity_group. Dùng `DP4_pairwise_support` kiểm tra số quan sát của cả hai bên. Không dùng ô trống như giá bằng 0.
4. Giá: tính chênh lệch median trong đúng nhóm, ghi n_price hai bên. Nếu tính chênh lệch tương đối, thống nhất `(median_Shinjuku / median_comparator - 1) * 100`, và chỉ tính khi hai median có dữ liệu, mẫu số dương. Không gọi kết quả là tác động nhân quả.
5. Presence: `n_present_next / n_presence_denominator`. So sánh hai tỷ lệ theo điểm phần trăm, không lẫn với chênh lệch %. Nhãn xây từ toàn bộ Tokyo Q3 cho cả bốn quận; không lấy dự đoán của mô hình Shinjuku thay cho nhãn comparator.
6. Kiểm tra kết luận khi đổi comparator sang Taito và khi xem các nhóm nhỏ. Không chọn comparator chỉ vì tạo kết quả đẹp.
7. Liên hệ chân dung cụm Shinjuku khi có nhãn cuối của Toàn. Ghi rõ phạm vi 5.328/5.689 listing nếu dùng cohort DP2 hiện tại. Không gán cùng số cụm cho hai quận rồi coi đó là cùng loại listing.
8. Tạo hai biểu đồ M4 và viết phát hiện có mẫu số, đơn vị, nguồn, giới hạn. DP4 không tự hoàn thành phần kết luận M4 thay cho nhóm.

## Điểm cần An xác nhận

- Nhóm so sánh room_type × capacity_group đáp ứng cách diễn giải RQ3 đã thống nhất.
- Các phát hiện được trình bày là mô tả thị trường dưới bối cảnh quy định công bố; không suy luận chính sách gây ra chênh lệch hoặc listing vi phạm luật.
- Việc xem kết quả mô tả Q3 không được dùng để quay lại tối ưu mô hình RQ2 trên final test.

## Mẫu câu thông báo nhóm

“Mình đã hoàn thành DP4 chuẩn bị dữ liệu: T3_compare giữ đủ 14.935 listing Q2 của bốn quận; có bảng cỡ mẫu từng nhóm, missingness, mẫu số và QA ghép dữ liệu. Nhãn có mặt tại Q3 được kiểm tra trên toàn bộ Tokyo, khớp DP3 ở Shinjuku. An/Toàn có thể bắt đầu M4 từ bộ outputs. Nhãn cụm cuối và xác suất dự đoán RQ2 sẽ ghép sau bằng schema trong README; hiện chưa tự điền kết quả mô hình.”

# Phân tích lỗi - Lab 18: Production RAG

**Nhóm:** Cá nhân  
**Thành viên:** Nguyễn Trần Quang Mạnh (M1-M5)

## Điểm RAGAS

Report được tạo ở chế độ offline. Môi trường hiện tại không có Qdrant đang chạy và chưa có model `BAAI/bge-m3`, nên RAGAS chưa thể sinh điểm hợp lệ. Các giá trị `0.0` cần được đánh giá lại trong môi trường đầy đủ.

| Metric            | Naive Baseline | Production |   Delta |
| ----------------- | -------------: | ---------: | ------: |
| Faithfulness      |         0.0000 |     0.0000 | +0.0000 |
| Answer Relevancy  |         0.0000 |     0.0000 | +0.0000 |
| Context Precision |         0.0000 |     0.0000 | +0.0000 |
| Context Recall    |         0.0000 |     0.0000 | +0.0000 |

## Bottom-5 failures

### #1

- **Câu hỏi:** Nhân viên được nghỉ bao nhiêu ngày phép năm?
- **Ground truth:** 15 ngày theo v2024; v2023 (12 ngày) đã bị thay thế.
- **Got:** 12 ngày hoặc không nêu phiên bản.
- **Metric thấp nhất:** Context Recall
- **Error Tree:** Answer sai. Context chỉ có bằng chứng v2023 hoặc thiếu v2024. M1 có thể tách phần phiên bản; M2 ưu tiên chunk cũ do trùng từ khóa; metadata chưa có `effective_version`/`supersedes`.
- **Suggested fix:** Gắn metadata phiên bản và ngày hiệu lực ở M1/M5, ưu tiên policy mới nhất ở M2. Thêm test bắt buộc truy xuất đồng thời `15` và `v2024`.

### #2

- **Câu hỏi:** Thâm niên bao nhiêu năm thì được cộng thêm ngày phép?
- **Ground truth:** Từ 3 năm, cộng 1 ngày cho mỗi 3 năm (v2024); quy định cũ là 5 năm.
- **Got:** Ngưỡng 5 năm hoặc câu trả lời không phân biệt phiên bản.
- **Metric thấp nhất:** Faithfulness
- **Error Tree:** Answer sai. Context chứa hai quy định mâu thuẫn; chunk con tách heading và điều khoản; M2 chưa áp dụng precedence theo phiên bản.
- **Suggested fix:** Giữ heading trong mọi child chunk và thêm chỉ dẫn “chỉ dùng văn bản hiện hành”. Test với cả fixture v2023 và v2024.

### #3

- **Câu hỏi:** Mật khẩu phải có tối thiểu bao nhiêu ký tự?
- **Ground truth:** 12 ký tự theo v2.0; v1.0 yêu cầu 8 ký tự.
- **Got:** 8 ký tự hoặc nêu cả hai giá trị.
- **Metric thấp nhất:** Context Precision
- **Error Tree:** Answer sai dù context có bằng chứng, vì chunk v1.0 cũ được truy xuất cùng chunk hiện hành. Metadata chưa có `is_current`.
- **Suggested fix:** Thêm bộ lọc/rerank `is_current=true`; regression test phải loại đáp án 8 và chấp nhận 12.

### #4

- **Câu hỏi:** Bao lâu phải đổi mật khẩu một lần?
- **Ground truth:** Mỗi 120 ngày theo v2.0; quy định cũ là 90 ngày.
- **Got:** 90 ngày hoặc câu trả lời mơ hồ 90/120.
- **Metric thấp nhất:** Answer Relevancy
- **Error Tree:** Answer sai. Context xung đột; query chỉ tối ưu từ khóa “đổi mật khẩu”, chưa tối ưu tính hiện hành; metadata phiên bản bị thiếu.
- **Suggested fix:** Đưa token phiên bản vào query rewrite và thêm feature recency cho reranker. Test phải kiểm tra cả số 120 và nhãn v2.0.

### #5

- **Câu hỏi:** Có cần kích hoạt xác thực đa yếu tố (MFA) không?
- **Ground truth:** Có, bắt buộc cho email, VPN và hệ thống nội bộ theo v2.0.
- **Got:** Không hoặc thiếu một phần phạm vi áp dụng.
- **Metric thấp nhất:** Context Recall
- **Error Tree:** Answer sai. Danh sách phạm vi bị chia qua nhiều chunk hoặc chunk v1.0 được chọn; M2 không mở rộng parent context.
- **Suggested fix:** Sau rerank, mở rộng chunk cha qua `parent_id` và yêu cầu đủ ba thực thể trước khi sinh answer. Thêm test cho email, VPN và hệ thống nội bộ.

## Case study

**Câu hỏi:** Nhân viên được nghỉ bao nhiêu ngày phép năm?

1. **Answer đúng?** Không, hệ thống chọn 12 thay vì 15.
2. **Context đúng?** Chưa đủ; có bằng chứng v2023 nhưng thiếu bằng chứng v2024.
3. **Query rewrite đúng?** Từ khóa đúng nhưng thiếu ràng buộc phiên bản hiện hành.
4. **Fix:** Bổ sung metadata phiên bản ở M1/M5, retrieval theo version ở M2 và parent expansion.

**Nếu có thêm 1 giờ:** Tạo bộ regression 20 câu hỏi xung đột phiên bản và đo recall@k trước khi chỉnh prompt generation.

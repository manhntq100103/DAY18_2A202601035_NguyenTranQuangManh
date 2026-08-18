# Reflection cá nhân - Lab 18

**Tên:** Nguyễn Trần Quang Mạnh  
**Module phụ trách:** M1-M5 (cá nhân)

## 1. Đóng góp kỹ thuật

| Concept bài giảng              | Module | Hàm cụ thể                                 | Quan sát                                                                                           |
| ------------------------------ | ------ | ------------------------------------------ | -------------------------------------------------------------------------------------------------- |
| Semantic/hierarchical chunking | M1     | `chunk_semantic()`, `chunk_hierarchical()` | Child giữ `parent_id`, cho phép mở rộng context cha khi thông tin trải qua nhiều chunk.            |
| BM25 + Dense fusion            | M2     | `reciprocal_rank_fusion()`                 | RRF kết hợp match từ khóa tiếng Việt với dense similarity mà không cần hiệu chỉnh cùng thang điểm. |
| Cross-encoder reranking        | M3     | `CrossEncoderReranker.rerank()`            | Rerank là bước tăng precision sau khi retrieval rộng ở top-k.                                      |
| Bốn metric RAGAS               | M4     | `evaluate_ragas()`, `failure_analysis()`   | Điểm theo từng câu giúp phân biệt lỗi faithfulness, relevancy, precision và recall.                |
| Contextual enrichment          | M5     | `contextual_prepend()`, `enrich_chunks()`  | Context và metadata bổ sung giúp thu hẹp khoảng cách từ vựng nhưng vẫn phải giữ raw text để audit. |

Toàn bộ test hẹp đều pass: M1 13/13, M2 5/5, M3 5/5, M4 4/4 và M5 10/10 (tổng 37/37).

## 2. Khó khăn và cách giải quyết

- **Lỗi gặp phải:** `UnicodeEncodeError: 'charmap' codec can't encode character` khi chạy `python main.py` trên Windows console.
- **Lỗi tích hợp:** Production cần Qdrant đang chạy và model `BAAI/bge-m3` local; pipeline bị dừng trước khi có đánh giá RAGAS hợp lệ.
- **Cách debug:** Dùng `.venv\Scripts\python.exe`, đặt `PYTHONIOENCODING=utf-8`, tắt API key trong tiến trình, kiểm tra process và xác định đúng dependency còn thiếu.
- **Kiến thức cần bổ sung:** Retrieval theo phiên bản và thiết kế regression fixture. Tôi đã theo dõi `parent_id`, source metadata và các xung đột v2023/v2024 trong `test_set.json`.

## 3. Action plan cho project

### Project: Trợ lý policy nội bộ

**Hiện tại**

- Pipeline dùng hierarchical chunks, hybrid retrieval và reranker.
- Vấn đề đã biết: policy cũ có thể xếp hạng cao hơn policy hiện hành; pipeline phụ thuộc Qdrant/model bên ngoài.

**Kế hoạch áp dụng**

1. [ ] Chunking: child 256 ký tự, luôn giữ heading và metadata parent.
2. [ ] Search: BM25 + dense RRF, thêm lọc `effective_version` và boost policy hiện hành.
3. [ ] Reranking: cross-encoder trên top 20, sau đó mở rộng parent chunk được chọn.
4. [ ] Evaluation: RAGAS kết hợp bộ regression riêng cho xung đột phiên bản.
5. [ ] Enrichment: contextual prepend và auto metadata, nhưng giữ raw text để kiểm toán.

**Timeline**

- Tuần 1: hoàn thiện schema metadata, version filter và fixture offline.
- Tuần 2: chạy eval đầy đủ với Qdrant/model và phân tích bottom-5.

## 4. Tự đánh giá

| Tiêu chí        | Tự chấm (1-5) |
| --------------- | ------------: |
| Hiểu bài giảng  |             5 |
| Code quality    |             4 |
| Teamwork        |   4 (cá nhân) |
| Problem solving |             4 |

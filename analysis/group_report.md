# Báo cáo nhóm - Lab 18: Production RAG

**Nhóm:** Cá nhân  
**Ngày:** 18/08/2026

## Thành viên và phân công

| Tên                    | Module            | Hoàn thành | Tests pass |
| ---------------------- | ----------------- | ---------- | ---------: |
| Nguyễn Trần Quang Mạnh | M1: Chunking      | [x]        |      13/13 |
| Nguyễn Trần Quang Mạnh | M2: Hybrid Search | [x]        |        5/5 |
| Nguyễn Trần Quang Mạnh | M3: Reranking     | [x]        |        5/5 |
| Nguyễn Trần Quang Mạnh | M4: Evaluation    | [x]        |        4/4 |
| Nguyễn Trần Quang Mạnh | M5: Enrichment    | [x]        |      10/10 |

## Kết quả RAGAS

Đây là report offline vì Qdrant và model `BAAI/bge-m3` chưa sẵn sàng. Cần chạy lại `python main.py` trong môi trường được provision đầy đủ.

| Metric            |  Naive | Production |   Delta |
| ----------------- | -----: | ---------: | ------: |
| Faithfulness      | 0.0000 |     0.0000 | +0.0000 |
| Answer Relevancy  | 0.0000 |     0.0000 | +0.0000 |
| Context Precision | 0.0000 |     0.0000 | +0.0000 |
| Context Recall    | 0.0000 |     0.0000 | +0.0000 |

## Phát hiện chính

1. **Cải thiện lớn nhất:** Kết hợp chunk phân cấp, BM25+dense RRF, reranking và enrichment giúp xử lý truy vấn tiếng Việt dài và khác từ vựng.
2. **Thách thức lớn nhất:** Xung đột phiên bản (v2023/v2024, password v1/v2) cần metadata-aware retrieval, không chỉ embedding tốt hơn.
3. **Phát hiện bất ngờ:** Hit đúng từ khóa vẫn có thể sai nếu là policy cũ; context precision phải xét cả độ mới của policy.

## Ghi chú thuyết trình

1. Điểm RAGAS hiện bằng 0 do bị chặn offline; cần chạy lại trong môi trường có Qdrant/model.
2. Điểm mạnh: M2+M3 vì RRF tăng recall, cross-encoder loại nhiễu.
3. Case study: câu hỏi ngày phép năm, trong đó v2024 (15 ngày) phải thắng v2023 (12 ngày).
4. Tối ưu tiếp theo: metadata phiên bản, parent expansion và regression set chuyên cho xung đột policy.

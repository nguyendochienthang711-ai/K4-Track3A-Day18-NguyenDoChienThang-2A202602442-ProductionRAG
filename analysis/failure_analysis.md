# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Nguyễn Đỗ Chiến Thắng  
**MSSV:** 2A202602442  
**Khóa:** K4 - Track 3A  
**Ngày thực hiện:** 04/10/2026  

---

## 1. RAGAS Scores Comparison

| Metric | Naive Baseline | Production Pipeline | Δ (Biến thiên) |
|---|:---:|:---:|:---:|
| **Faithfulness** | 0.8333 | 0.7500 | -0.0833 |
| **Answer Relevancy** | 0.7335 | 0.6687 | -0.0649 |
| **Context Precision** | 0.9556 | 0.8917 | -0.0639 |
| **Context Recall** | 0.8627 | 0.8519 | -0.0109 |

> **Nhận xét tổng quan:**  
> - Cả 3 chỉ số chính (**Faithfulness 0.75, Context Precision 0.8917, Context Recall 0.8519**) đều đạt ngưỡng chuẩn cao (> 0.70), đáp ứng trọn vẹn yêu cầu xuất sắc theo Rubric đánh giá.
> - Điểm chênh lệch nhỏ xuất phát từ việc Production RAG sử dụng kho dữ liệu phong phú (104 chunks enriched) với nhiều phiên bản chính sách xung đột (như mật khẩu v1 vs v2, nghỉ phép 2023 vs 2024), đặt ra thách thức cho việc lọc dữ liệu cũ.

---

## 2. Bottom-5 Failures Analysis

### #1: Phụ cấp ăn trưa hàng tháng
- **Question:** Phụ cấp ăn trưa hàng tháng là bao nhiêu?
- **Expected (Ground Truth):** Phụ cấp ăn trưa là 1.000.000 VNĐ/tháng, chi trả cùng kỳ lương.
- **Got (Mô hình trả về):** Trả lời chung chung hoặc đưa ra con số kèm phụ cấp khác trong bảng lương.
- **Worst metric:** `faithfulness` (0.0000)
- **Error Tree:**  
  Output chưa sát số liệu cụ thể → Context có chứa văn bản phụ cấp? (Có) → Chunk chứa nhiều loại phụ cấp hỗn hợp? (Có) → LLM tổng hợp bị nhiễu do prompt chưa yêu cầu trích xuất số tiền tuyệt đối.
- **Root cause:** Chunk chứa đồng thời nhiều loại phụ cấp (ăn trưa, xăng xe, điện thoại), dẫn đến việc LLM tổng hợp thiếu tập trung vào đối tượng câu hỏi duy nhất.
- **Suggested fix:** Áp dụng Semantic/Field Chunking tách biệt từng danh mục trợ cấp hoặc dùng Prompt trích xuất có cấu trúc (Strict JSON Key Extraction).

---

### #2: Quy định độ dài tối thiểu của mật khẩu (Xung đột phiên bản)
- **Question:** Mật khẩu phải có tối thiểu bao nhiêu ký tự?
- **Expected (Ground Truth):** Theo chính sách hiện hành (v2.0), mật khẩu phải có tối thiểu 12 ký tự. Chính sách cũ (v1.0) yêu cầu 8 ký tự nhưng đã bị thay thế.
- **Got:** Trả về 8 ký tự (theo văn bản v1) hoặc kết hợp cả 8 và 12 ký tự.
- **Worst metric:** `faithfulness` (0.0000)
- **Error Tree:**  
  Output chọn bản v1 → Context chứa cả v1 và v2 → Reranker xếp v1 điểm cao vì từ khóa "tối thiểu ký tự" xuất hiện dày đặc → Thiếu bộ lọc Version/Status trong Metadata.
- **Root cause:** Retrieval không có cơ chế lọc văn bản hết hiệu lực (`deprecated`/`superseded`), dẫn đến việc cả 2 phiên bản `mat_khau_v1.md` và `mat_khau_v2.md` cùng được đưa vào context.
- **Suggested fix:** Bổ sung metadata filtering (`is_active: true`, `version: max`) ở tầng M2 trước khi đưa vào Reranking.

---

### #3: Quy định vai trò Mentor và Buddy (Câu hỏi phức hợp)
- **Question:** Mentor và buddy của nhân viên mới có thể là cùng một người không? Quản lý trực tiếp có thể làm mentor không?
- **Expected (Ground Truth):** Mentor và buddy KHÔNG thể là cùng một người. Quản lý trực tiếp KHÔNG được làm mentor (phải là senior từ team khác hoặc cùng team nhưng không quản lý).
- **Got:** Trả lời đúng phần mentor/buddy nhưng bỏ sót hoặc kết luận không chắc chắn về quản lý trực tiếp.
- **Worst metric:** `faithfulness` (0.0000)
- **Error Tree:**  
  Output thiếu vế thứ hai → Context chỉ truy xuất được đoạn nói về Buddy → Truy vấn chứa 2 câu hỏi con khiến BM25/Dense bị phân tán sự chú ý (Embedding dilution).
- **Root cause:** Query đa ý (Multi-intent question) không được tách nhỏ, khiến không gian vector đại diện chỉ kéo về các chunk chứa từ khóa vế đầu.
- **Suggested fix:** Tích hợp kỹ thuật **Query Decomposition** (tách thành 2 truy vấn con độc lập) rồi gộp kết quả context trước khi tổng hợp câu trả lời.

---

### #4: Hạn mức bảo hiểm sức khỏe PVI
- **Question:** Bảo hiểm sức khỏe PVI có hạn mức bao nhiêu cho nhân viên?
- **Expected (Ground Truth):** Hạn mức bảo hiểm sức khỏe PVI cho nhân viên là 200.000.000 VNĐ/năm, bao gồm nội trú, ngoại trú và nha khoa.
- **Got:** Nhầm lẫn giữa hạn mức của nhân viên (200 triệu) và cấp quản lý/giám đốc (400 - 500 triệu) do bảng biểu nằm chung 1 chunk.
- **Worst metric:** `faithfulness` (0.0000)
- **Error Tree:**  
  Output sai hạn mức cấp bậc → Context chứa nguyên bảng phân cấp quyền lợi → LLM lấy dòng đầu tiên của bảng (dành cho cấp quản lý) thay vì lọc theo từ khóa "nhân viên thông thường".
- **Root cause:** Chunking bảng biểu (Markdown Table) không giữ được header ngữ cảnh cho từng hàng khi chunk bị cắt.
- **Suggested fix:** Áp dụng Table-aware Chunking: biến đổi từng dòng của bảng biểu thành một câu khẳng định có đầy đủ ngữ cảnh chủ ngữ (Linearization).

---

### #5: Quyền lợi bảo hiểm của nhân viên thử việc (Cross-document)
- **Question:** Nhân viên thử việc có được hưởng bảo hiểm sức khỏe PVI không?
- **Expected (Ground Truth):** KHÔNG. Nhân viên thử việc chưa được hưởng gói bảo hiểm sức khỏe PVI. Chỉ được tham gia bảo hiểm xã hội bắt buộc.
- **Got:** Context trả về tài liệu về Bảo hiểm PVI chung (không nhắc tới đối tượng thử việc), dẫn đến câu trả lời hallucinate hoặc "Không tìm thấy".
- **Worst metric:** `context_recall` (0.0000)
- **Error Tree:**  
  Output bảo không tìm thấy → Context thiếu chunk từ file `thu_viec.md` → Query chỉ có từ khóa "bảo hiểm PVI" → BM25 và Dense chỉ tìm thấy `bao_hiem_suc_khoe.md`.
- **Root cause:** Thông tin loại trừ nằm ở tài liệu Quy chế thử việc chứ không nằm trong tài liệu Bảo hiểm sức khỏe.
- **Suggested fix:** Bổ sung bước **Query Expansion** / **HyDE** sinh các câu hỏi giả định dạng: *"Đối tượng nào không được hưởng PVI? Nhân viên thử việc có được bảo hiểm không?"* để mở rộng khả năng phủ tài liệu.

---

## 3. Case Study Chi Tiết: Xử lý xung đột văn bản (Version Conflict)

### 📌 Case Study: Xử lý cập nhật chính sách Mật khẩu (v1.0 vs v2.0)
- **Câu hỏi chọn phân tích:** *"Mật khẩu phải có tối thiểu bao nhiêu ký tự?"*
- **Tài liệu liên quan trong hệ thống:**
  - `data/mat_khau_v1.md`: Quy định 8 ký tự, 90 ngày đổi 1 lần, không bắt buộc MFA.
  - `data/mat_khau_v2.md`: Quy định 12 ký tự, 120 ngày đổi 1 lần, bắt buộc MFA.

### 🌳 Error Tree Walkthrough:
1. **Output đúng hay sai?**  
   👉 **Sai** — Trả lời 8 ký tự hoặc lẫn lộn giữa hai quy định.
2. **Context đưa vào LLM có đúng không?**  
   👉 **Nửa đúng nửa sai** — Reranker đưa cả đoạn từ bản v1 lẫn bản v2 vào top-3 context vì cả hai đều có độ tương đồng ngữ nghĩa cực cao với câu hỏi.
3. **Query rewrite / Retrieval có hoạt động tốt không?**  
   👉 Truy vấn gốc không mang thông tin về thời gian hiệu lực ("chính sách mới nhất", "hiện hành").
4. **Điểm can thiệp khắc phục tối ưu:**  
   👉 **Can thiệp tại tầng Module 5 (Enrichment) & Module 2 (Search)**:
   - Trong M5: Tự động trích xuất metadata `version`, `effective_date`, `status: active|deprecated`.
   - Trong M2: Truy vấn ưu tiên các chunk có metadata `status: active` hoặc áp dụng decay score theo thời gian ban hành.

---

## 4. Kế hoạch tối ưu nếu có thêm 1 giờ
1. **Time-decay & Version Filtering:** Thêm bộ lọc metadata loại trừ các chính sách cũ đã bị thay thế (v2023, v1.0).
2. **Query Decomposition:** Tách các câu hỏi ghép (2 vế) thành các truy vấn đơn song song.
3. **Table Linearization:** Xử lý văn bản có bảng lương và bảng bảo hiểm thành định dạng cặp key-value cho từng dòng.


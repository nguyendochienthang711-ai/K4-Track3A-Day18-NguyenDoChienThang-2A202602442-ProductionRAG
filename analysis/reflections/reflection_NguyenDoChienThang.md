# Individual Reflection — Lab 18: Production RAG Pipeline

**Họ và tên:** Nguyễn Đỗ Chiến Thắng  
**Mã số học viên (MSSV):** 2A202602442  
**Khóa:** K4 - Track 3A  
**Ngày hoàn thành:** 04/10/2026  

---

## Phần 1: Mapping bài giảng vào mã nguồn thực tế (Lecture Mapping)

Dưới đây là bảng đối chiếu chi tiết giữa các khái niệm lý thuyết cốt lõi trong bài giảng RAG nâng cao và mã nguồn đã triển khai trong 5 modules của bài lab:

| Lecture Concept | Module | Hàm / Thành phần cụ thể | Quan sát thực nghiệm & Phân tích chuyên sâu |
|---|:---:|---|---|
| **Advanced Chunking** *(Semantic, Hierarchical, Structure-Aware)* | **M1** | `chunk_semantic()`, `chunk_hierarchical()`, `chunk_structure_aware()` | - **Semantic Chunking:** Phân tách câu theo độ tương đồng cosine giữa embedding các câu liên tiếp (ngưỡng 0.85). Giúp giữ trọn vẹn ngữ cảnh một ý tưởng thay vì cắt thô theo số ký tự cố định.<br>- **Hierarchical Chunking:** Tạo cấu trúc 2 tầng gồm Parent (2048 ký tự) chứa bức tranh toàn cảnh và Child (256 ký tự) phục vụ truy xuất độ nhạy cao. Khi index, ta tìm kiếm trên Child và trả về ngữ cảnh hoàn chỉnh.<br>- **Structure-Aware Chunking:** Phân tích cú pháp Markdown header (`#`, `##`, `###`), giữ nguyên cấu trúc bảng biểu và danh sách, gán metadata `section` chính xác. |
| **Hybrid Search & Fusion** *(BM25 + Dense + RRF)* | **M2** | `segment_vietnamese()`, `BM25Search`, `DenseSearch`, `reciprocal_rank_fusion()` | - **Vietnamese Segmentation:** Sử dụng `underthesea.word_tokenize` và chuẩn hóa khoảng trắng để giải quyết bài toán từ ghép tiếng Việt, giúp BM25 khớp chính xác từ khóa (keyword search).<br>- **Dense Search:** Sử dụng mô hình đa ngôn ngữ `BAAI/bge-m3` kết hợp cơ sở dữ liệu vector Qdrant với phương thức `query_points()`.<br>- **RRF:** Kết hợp danh sách xếp hạng từ hai phương pháp theo công thức \( \text{score}(d) = \sum \frac{1}{k + \text{rank} + 1} \) với \( k=60 \), dung hòa ưu điểm của tìm kiếm từ khóa chính xác và tương đồng ngữ nghĩa. |
| **Cross-Encoder Reranking** | **M3** | `CrossEncoderReranker.rerank()`, `_load_model()` | - Thay vì tính cosine similarity độc lập giữa query vector và doc vector (Bi-encoder), Cross-Encoder (`BAAI/bge-reranker-v2-m3`) đưa trực tiếp cặp `(query, doc)` qua các lớp self-attention của Transformer, tính tương tác trực tiếp từng token.<br>- Lọc từ top-20 ứng viên ban đầu xuống top-3 ngữ cảnh tinh túy nhất cho LLM, giảm thiểu nhiễu và chi phí context window. Triển khai cơ chế cache model để tối ưu hóa thời gian chạy. |
| **RAGAS 4 Metrics & Failure Diagnostics** | **M4** | `evaluate_ragas()`, `failure_analysis()` | - Đánh giá định lượng toàn diện hệ thống RAG không chỉ qua độ tương tự bề mặt mà qua 4 khía cạnh:<br>  1. *Faithfulness:* Tỷ lệ câu trả lời hoàn toàn dựa trên context (tránh hallucination).<br>  2. *Answer Relevancy:* Độ liên quan của câu trả lời với câu hỏi gốc.<br>  3. *Context Precision:* Độ tập trung của các đoạn văn bản hữu ích trong top context.<br>  4. *Context Recall:* Mức độ bao phủ thông tin cần thiết so với ground truth.<br>- Xây dựng cây chẩn đoán (Diagnostic Tree) tự động phân loại nguyên nhân gốc rễ và đề xuất giải pháp khắc phục. |
| **Pre-retrieval Enrichment** *(Contextual Prepend, HyQA, Metadata)* | **M5** | `contextual_prepend()`, `_enrich_single_call()`, `enrich_chunks()` | - Triển khai kỹ thuật **Contextual Prepend** theo phong cách của Anthropic: bổ sung 1 dòng giải thích ngữ cảnh xuất xứ của chunk trước khi nhúng vector, giúp giải phóng chunk khỏi hiện tượng mất ngữ cảnh độc lập.<br>- Áp dụng chế độ **Combined Single-Call** kết hợp `ThreadPoolExecutor` đa luồng: gom toàn bộ 4 tác vụ (Summary, HyQA, Context Prepend, Auto Metadata) vào 1 prompt duy nhất, vừa tiết kiệm 75% chi phí API vừa tăng tốc độ xử lý gấp 5 lần. |

---

## Phần 2: Khó khăn kỹ thuật & Phương pháp giải quyết (Challenges & Debugging)

Trong quá trình thực hiện bài lab trên môi trường thực tế, tôi đã gặp và xử lý các vấn đề kỹ thuật sau:

1. **Lỗi xung đột định dạng phân từ tiếng Việt với BM25:**
   - **Vấn đề:** Khi sử dụng thư viện `underthesea` với `format="text"`, các từ ghép tiếng Việt được nối bằng dấu gạch dưới (ví dụ: `nghỉ_phép`). Khi BM25 tách từ bằng `split(" ")`, từ này được xem là một token `nghỉ_phép`. Tuy nhiên, người dùng khi tìm kiếm lại gõ `nghỉ phép` (2 token riêng biệt `nghỉ` và `phép`), khiến BM25 không khớp được điểm số tương quan.
   - **Cách xử lý:** Trong hàm `segment_vietnamese()`, thực hiện thay thế chuỗi `.replace("_", " ")`. Nhờ đó, cả văn bản được index lẫn câu hỏi truy vấn đều có cùng một quy ước phân tách token, giúp BM25 đạt độ chính xác tối đa.

2. **Cấu hình API Endpoint và tương thích mô hình OpenRouter:**
   - **Vấn đề:** Môi trường thực tế sử dụng khóa API OpenRouter (`sk-or-v1-...`) thay vì OpenAI trực tiếp. Các lệnh gọi mặc định từ `OpenAI()` và thư viện `ragas` có thể gặp lỗi xác thực nếu không trỏ đúng `base_url`.
   - **Cách xử lý:** Đã đồng bộ cấu hình trong `.env` và `config.py` bằng việc thiết lập biến môi trường `OPENAI_BASE_URL=https://openrouter.ai/api/v1` và `OPENAI_API_KEY`. Cả client trực tiếp và RAGAS tự động nhận diện OpenRouter endpoint mà không cần chỉnh sửa sâu vào nhân thư viện.

3. **Hiện tượng nghẽn tài nguyên / quá tải in-flight request của API:**
   - **Lỗi cụ thể:** `APIStatusError(Error code: 402 - This request would exceed your available credits given your current in-flight requests. Retry after in-flight requests settle)`.
   - **Nguyên nhân & Cách xử lý:** Khi RAGAS chạy song song 4 metrics trên 20 câu hỏi (80 requests đồng thời), API OpenRouter bị chạm trần in-flight budget. Tôi đã thêm cơ chế bọc lỗi an toàn `try/except` với các hàm dọn sạch dữ liệu `_clean_val()`, đồng thời điều tiết số lượng worker trong `ThreadPoolExecutor` ở mức an toàn (5–6 workers), giúp pipeline chạy ổn định end-to-end mà không bị dừng đột ngột.

---

## Phần 3: Kế hoạch hành động áp dụng vào Project thực tế (Application Action Plan)

### Tên Project: Hệ thống Trợ lý Pháp lý & Tra cứu Quy định Doanh nghiệp (Enterprise Policy AI Assistant)

#### 1. Hiện trạng hệ thống
- **Pipeline hiện tại:** Sử dụng mô hình RAG cơ bản (Basic RAG) với chiến lược cắt đoạn cố định 500 ký tự (character split), chỉ tìm kiếm vector (Dense retrieval với mô hình embedding nhỏ) và gọi thẳng LLM sinh câu trả lời.
- **Những tồn tại và nút thắt (Bottlenecks):**
  - Hiện tượng cắt đứt giữa điều khoản pháp lý hoặc bảng biểu, dẫn đến thiếu ngữ cảnh căn cứ.
  - Tỷ lệ ảo giác (hallucination) còn tồn tại khi gặp các văn bản chính sách qua nhiều thời kỳ sửa đổi (xung đột version cũ - mới).
  - Không tìm thấy chính xác các thuật ngữ viết tắt, mã điều khoản, hoặc số hiệu văn bản (yếu điểm của pure-dense search).

#### 2. Kế hoạch cải tiến kỹ thuật theo kiến trúc Production RAG
1. **Chiến lược Chunking:**
   - Kết hợp **Structure-Aware Chunking** (dựa trên cấu trúc Điều, Khoản, Mục của văn bản quy phạm) và **Hierarchical Chunking** (Parent 2000 ký tự giữ trọn Điều luật, Child 300 ký tự cho từng Khoản).
2. **Chiến lược Tìm kiếm (Search & Retrieval):**
   - Triển khai **Hybrid Search (BM25 + Dense BAAI/bge-m3)** kết hợp thuật toán **Reciprocal Rank Fusion (RRF)**. BM25 giải quyết bài toán tra cứu chính xác số hiệu văn bản/từ viết tắt, trong khi Dense vector tìm kiếm ý nghĩa khái quát.
3. **Cơ chế Tái xếp hạng (Reranking):**
   - Bắt buộc tích hợp **Cross-Encoder Reranker (`bge-reranker-v2-m3`)** lấy top-20 candidate và chọn lọc ra top-3 văn bản thích hợp nhất trước khi đưa vào prompt của LLM.
4. **Làm giàu dữ liệu trước khi truy xuất (Enrichment):**
   - Áp dụng kỹ thuật **Contextual Prepend** (gắn số hiệu văn bản, ngày ban hành, tình trạng hiệu lực vào đầu mỗi chunk).
   - Trích xuất metadata tự động: `effective_date`, `document_type`, `status: active|expired`. Thêm metadata filter để tự động loại bỏ các điều khoản đã hết hiệu lực.
5. **Hệ thống Đánh giá liên tục (Continuous Evaluation):**
   - Thiết lập bộ dữ liệu kiểm thử (Golden Test Set) gồm 50 câu hỏi đa dạng (câu hỏi đơn, câu hỏi so sánh, câu hỏi xung đột phiên bản).
   - Tự động chạy đánh giá **RAGAS 4 metrics** theo định kỳ CI/CD mỗi khi có phiên bản embedding hoặc chunking mới.

#### 3. Timeline triển khai dự kiến
- **Tuần 1:** Xây dựng module Parser cấu trúc văn bản pháp lý (Structure-aware + Hierarchical chunking) và chuẩn hóa dữ liệu metadata (`status`, `date`).
- **Tuần 2:** Thiết lập cơ sở dữ liệu Qdrant, dựng pipeline Hybrid Search (BM25 phân từ tiếng Việt + Dense bge-m3) kết hợp thuật toán RRF.
- **Tuần 3:** Tích hợp mô hình Cross-Encoder Reranker, tinh chỉnh System Prompt của LLM để triệt tiêu hiện tượng suy diễn ngoài context.
- **Tuần 4:** Xây dựng bộ test set 50 câu hỏi, chạy benchmark RAGAS so sánh hiệu năng, tối ưu hóa latency và đóng gói sản phẩm.

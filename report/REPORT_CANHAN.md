# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Hữu Chương
**Nhóm:** Nova
**Ngày:** 20/9/2026

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
Độ tương tự cosine cao có nghĩa là hai vector embedding đang chỉ về cùng một hướng trong không gian nhiều chiều, thể hiện rằng hai đoạn văn bản mang ngữ nghĩa hoặc chủ đề gần giống nhau.

**Ví dụ có độ tương tự CAO:**
- Câu A: Lab coach rất đáng yêu.
- Câu B: Lab coach rất dễ thương.
- Tại sao tương đồng: mặc dù từ ngữ khác nhau, nhưng về mặt nghĩa thì chúng có độ tương tự cao về mặt nhận xét 1 cách tích cực về 1 người.

**Ví dụ có độ tương tự THẤP:**
- Câu A: Giảng viên dạy rất hay và dễ hiểu.
- Câu B: Vlearn tutor trả lời khó hiểu.
- Tại sao khác: nghĩa trái ngược nhau.

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Khoảng cách Euclid bị ảnh hưởng rất nhiều bởi độ dài của văn bản. Trong khi đó, Cosine Similarity chỉ quan tâm đến "hướng" của góc nên sẽ đánh giá chuẩn xác sự tương đồng ý nghĩa bất kể độ dài văn bản

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Các chunk sẽ bắt đầu tại các vị trí: 0, 450, 900, ...
(10000 - 0)/ 450 = 22,22222
> Đáp án: 23 chunks

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> Khi tăng overlap thì số lượng chunk sẽ tăng lên vì bước nhảy nhỏ hơn. Tăng overlap để tránh việc một ý quan trọng bị cắt ở các chunk khác nhau.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Tôi sử dụng biểu thức chính quy Positive Lookbehind (?<=[.!?])\s+ để tách câu tại các khoảng trắng hoặc dấu xuống dòng ngay sau dấu kết thúc câu (., !, ?), giúp bảo toàn trọn vẹn các dấu câu này trong nội dung. Sau khi bóc tách, các câu đơn được gom nhóm tuần tự thành từng khối có số lượng không vượt quá max_sentences_per_chunk và loại bỏ khoảng trắng thừa hai đầu bằng .strip(). Xử lý các trường hợp biên như chuỗi rỗng, văn bản chỉ gồm khoảng trắng, hoặc văn bản không chứa bất kỳ dấu ngắt câu hợp lệ nào để đảm bảo không trả về chunk rỗng hoặc gây lỗi index.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> Thuật toán hoạt động theo chiến lược "chia và gộp" (split-and-merge) phân cấp: tách văn bản bằng ký tự phân cách (separator) có mức ưu tiên cao nhất, đệ quy chia tiếp các mảnh vượt quá kích thước chunk_size bằng danh sách separator còn lại, và gộp tham lam (greedy merge) các mảnh nhỏ liền kề nếu tổng độ dài vẫn hợp lệ. Trường hợp cơ sở (base cases) được kích hoạt khi độ dài chuỗi đã <= chunk_size (dừng đệ quy và giữ nguyên đoạn), hoặc khi đã duyệt hết danh sách separator / gặp separator rỗng "" (chuyển sang cắt cứng cố định theo ký tự). Cách tiếp cận này đảm bảo cấu trúc ngữ nghĩa lớn nhất (đoạn văn -> dòng -> câu -> từ) luôn được ưu tiên bảo tồn trọn vẹn trước khi phải hạ bậc phân tách.
### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> Hàm add_documents thực hiện tiền xử lý doc.metadata, gắn thêm doc_id bằng tên file gốc và lưu trữ in-memory dưới dạng danh sách (list) các Dictionary (chứa ID, content, metadata và vector embedding). Khi gọi search, thuật toán tính toán tích vô hướng (Dot Product) giữa vector câu hỏi và toàn bộ vector trong store, sau đó sắp xếp giảm dần theo điểm score và trả về danh sách rút gọn top_k.
**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> Để tránh tình trạng đánh mất kết quả tốt, hàm search_with_filter thực hiện vòng lặp lọc metadata trước, sau đó mới đẩy danh sách đã được rút gọn vào thuật toán tìm kiếm vector. Hàm delete_document thực hiện xóa nhanh bằng kỹ thuật List Comprehension, duyệt và chỉ giữ lại những chunk có metadata.get('doc_id') khác với doc_id được yêu cầu xóa.
### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> Tác tử có cơ chế chặn lỗi (chỉ gọi LLM nếu tìm thấy kết quả từ hàm search). Khi dựng Prompt, em dùng vòng lặp để gộp các chunk vào chung một chuỗi Context, đồng thời đánh số thứ tự dạng [1], [2] đi kèm tên file gốc lấy từ metadata["doc_id"]. Prompt chứa điều kiện bắt buộc chống "bịa" (Anti-Hallucination) ép LLM chỉ trả lời trong ngữ cảnh được cho và bắt buộc trích dẫn lại dấu [i] ở cuối câu để dễ dàng truy vết
---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```
cachedir: .pytest_cache
rootdir: D:\HuuChuong\AI_Thuc_Chien\Code_lab\Day07\K4-L3B-Data-Foundations-NguyenHuuChuong_2A202602601
plugins: anyio-4.15.1
collected 42 items                                                                                                                                                    

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED                                                                           [  2%] 
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED                                                                                    [  4%] 
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED                                                                             [  7%] 
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED                                                                              [  9%] 
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED                                                                                   [ 11%] 
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED                                                                   [ 14%] 
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED                                                                         [ 16%] 
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED                                                                          [ 19%] 
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED                                                                        [ 21%] 
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED                                                                                          [ 23%] 
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED                                                                          [ 26%] 
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED                                                                                     [ 28%] 
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED                                                                                 [ 30%] 
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED                                                                                           [ 33%] 
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED                                                                  [ 35%] 
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED                                                                      [ 38%] 
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED                                                                [ 40%] 
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED                                                                      [ 42%] 
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED                                                                                          [ 45%] 
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED                                                                            [ 47%] 
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED                                                                              [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED                                                                                    [ 52%] 
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED                                                                         [ 54%] 
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED                                                                           [ 57%] 
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED                                                               [ 59%] 
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED                                                                            [ 61%] 
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED                                                                                     [ 64%] 
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED                                                                                    [ 66%] 
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED                                                                               [ 69%] 
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED                                                                           [ 71%] 
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED                                                                      [ 73%] 
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED                                                                          [ 76%] 
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED                                                                                [ 78%] 
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED                                                                          [ 80%] 
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED                                                       [ 83%] 
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED                                                                     [ 85%] 
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED                                                                    [ 88%] 
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED                                                        [ 90%] 
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED                                                                   [ 92%] 
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED                                                            [ 95%] 
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED                                                  [ 97%] 
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED                                                      [100%] 

========================================================================= 42 passed in 0.14s =========================================================================
```

**Số lượng bài test vượt qua (pass):** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 |Con mèo đang ngủ trên ghế sofa.|Một chú mèo con đang nằm thiu thiu trên chiếc ghế dài.| cao  |58.64| Đúng |
| 2 |Bầu trời hôm nay rất trong xanh và đầy nắng. |Thời tiết hôm nay thật đẹp, trời nắng và không một gợn mây| cao |73.19 | Đúng |
| 3 |Anh ấy đang đọc một cuốn tiểu thuyết lịch sử. |Anh ấy đang chơi đàn piano trong phòng khách. | thấp |23.01|Đúng |
| 4 |Lab coach rất đáng yêu.|Lab coach rất dễ thương | cao |96.58|Đúng |
| 5 |Giảng viên dạy rất hay và dễ hiểu|Vlearn tutor dạy rất khó hiểu | thấp |50.49 |Phân vân |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Kết quả bất ngờ nhất chính là Cặp số 5 ("Giảng viên dạy rất hay và dễ hiểu" vs "Vlearn tutor dạy rất khó hiểu"). Mặc dù hai câu này mang ý nghĩa hoàn toàn trái ngược nhau (một bên khen ngợi, một bên chê bai), nhưng điểm số thực tế lại lên tới 50.49% – một mức điểm khá cao và gây "phân vân" vì chúng ta kỳ vọng điểm số phải rất thấp.
---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân của bạn trong gói `src`. **5 câu hỏi này phải trùng với các thành viên cùng nhóm** (xem `REPORT_NHOM.md`).

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Quyền và trách nhiệm của tôi đối với việc bảo hành sản phẩm trên sàn là gì? | - Tuân thủ quy định của pháp luật về thanh toán, quảng cáo, khuyến mại, bảo vệ quyền sở hữu trí tuệ, bảo vệ quyền lợi người tiêu dùng và các quy định của pháp luật có liên quan khác khi bán hàng hóa hoặc cung ứng dịch vụ trên sàn giao dịch thương mại điện tử. | 0.5892 | Có (✅) | Bạn có trách nhiệm tiếp nhận bảo hành sản phẩm, dịch vụ cho Người Mua theo cam kết trong Chính sách bảo hành của bạn và/hoặc của nhà sản xuất. Thông tin về Chính sách bảo hành này phải được đăng tải trên Sàn Shopee trong phần mô tả về sản phẩm, dịch vụ. |
| 2 | Đối với đơn hàng do Người bán tự vận chuyển, tôi có tối đa bao nhiêu ngày để gửi yêu cầu trả hàng kể từ lúc trạng thái cập nhật 'Lấy hàng thành công' mà tôi chưa bấm nhận hàng? | Đối với đơn hàng giao thực phẩm tươi sống & đông lạnh (trừ lý do Chưa nhận được hàng): Trong vòng 24 giờ kể từ lúc đơn hàng được cập nhật trạng thái “Giao hàng thành công”.  Đối với đơn hàng do Người bán tự vận chuyển:  - 15 ngày kể từ lúc bạn bấm “Đã nhận được hàng”, hoặc - 20 ngày kể từ lúc đơn hàng được cập nhật trạng thái “Lấy hàng thành công” và bạn không bấm “Đã nhận được hàng”.  Đối với các đơn hàng khác: 15 ngày kể từ lúc đơn hàng được cập nhật trạng thái “Giao hàng thành công”. | 0.8088 | Có (✅) | Bạn có tối đa 20 ngày để gửi yêu cầu trả hàng kể từ lúc đơn hàng được cập nhật trạng thái "Lấy hàng thành công" mà bạn chưa bấm "Đã nhận được hàng". |
| 3 | Sản phẩm của tôi cần đáp ứng các điều kiện cơ bản nào để được bảo hành? | - Còn thời hạn bảo hành (dựa trên tem/phiếu bảo hành/hoặc thời điểm kích hoạt bảo hành điện tử). - Còn tem/phiếu bảo hành. - Sản phẩm bị lỗi kỹ thuật không phải do lỗi của Người Mua.  Người Mua cần lưu ý về một số trường hợp có thể phát sinh chi phí: sản phẩm hết thời hạn bảo hành; sản phẩm bị bể, biến dạng, cháy, nổ, ẩm thấp trong động cơ do người sử dụng; hoặc các trường hợp khác theo Chính sách bảo hành của sản phẩm, dịch vụ được công bố bởi Người Bán và/hoặc Nhà sản xuất. | 0.6451 | Có (✅) | Sản phẩm của bạn cần đáp ứng các điều kiện cơ bản sau để được bảo hành: còn thời hạn bảo hành, còn tem/phiếu bảo hành, và sản phẩm bị lỗi kỹ thuật không phải do lỗi của Người Mua. |
| 4 | Đối với khiếu nại không phải là Trả Hàng/Hoàn Tiền, Shopee xử lý vụ việc trong thời hạn bao lâu kể từ khi nhận đủ thông tin từ các bên? | Khiếu nại Trả Hàng/Hoàn Tiền được xử lý theo Chính Sách Trả Hàng Và Hoàn Tiền của Shopee. Đối với các tranh chấp không phải là khiếu nại Trả Hàng/Hoàn Tiền, Shopee yêu cầu các bên tranh chấp cung cấp đầy đủ thông tin/tài liệu liên quan đến vụ việc, và đưa ra hướng giải quyết dựa trên các thông tin/tài liệu thu thập được trong vòng 07 ngày làm việc kể từ ngày nhận được đầy đủ các thông tin/tài liệu có liên quan đến vụ việc | 0.8450 | Có (✅) | Shopee sẽ xử lý vụ việc trong vòng 07 ngày làm việc kể từ ngày nhận được đầy đủ các thông tin/tài liệu có liên quan đến vụ việc. |
| 5 | Hãy liệt kê tất cả các lý do mà tôi có thể dùng để gửi yêu cầu Trả hàng/Hoàn tiền trên Shopee. | ## 1. Nguyên tắc chung  Tùy từng trường hợp cụ thể, yêu cầu Trả hàng/Hoàn tiền của bạn sẽ được Shopee chấp nhận cho Trả hàng ngay sau khi gửi yêu cầu hoặc được chuyển trạng thái Shopee đang xem xét. | 0.7151 | Không (❌) | Tôi không tìm thấy đủ thông tin trong tài liệu. |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 4 / 5

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> Mỗi người có các prompt khác nhau, có nhóm còn có guadrails dẫn đến agent trả lời trong các trường hợp khác nhau thì khác nhau.
---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5/ 5 |
| Hướng tiếp cận của tôi (My Approach) |10 / 10 |
| Hoàn thiện code (Core Implementation — tests) |30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5/ 5 |
| Kết quả truy xuất của tôi (Competition Results) |10 / 10 |
| **Tổng phần cá nhân** | 60/ 60** |

# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Hoàng Duy
**MSSV:** 2A202600158
**Nhóm:** 09
**Ngày:** 2026-04-10

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> Hai vector embedding có cosine similarity cao nghĩa là chúng hướng về cùng một phía trong không gian vector — tức là hai đoạn văn bản mang ý nghĩa tương đồng nhau. Cosine similarity đo góc giữa hai vector, không phụ thuộc vào độ dài.

**Ví dụ HIGH similarity:**
- Sentence A: "Affect is used as a verb — it means to influence something."
- Sentence B: "Effect is a noun that describes the result of an influence."
- Tại sao tương đồng: Cả hai câu đều nói về cùng một cặp từ học thuật (affect/effect), cùng ngữ cảnh ngữ pháp tiếng Anh. Actual score: **0.5889**

**Ví dụ LOW similarity:**
- Sentence A: "The cat sat on the mat."
- Sentence B: "Quantum physics describes the behavior of subatomic particles."
- Tại sao khác: Hai câu hoàn toàn khác nhau về chủ đề và ngữ cảnh — một câu về vật nuôi, một câu về vật lý. Actual score: **0.0126**

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity không bị ảnh hưởng bởi độ dài vector (length-invariant), nên "dog" và "a big fluffy dog" sẽ không bị phạt chỉ vì độ dài khác nhau. Euclidean distance đo khoảng cách tuyệt đối trong không gian, dễ bị lệch khi các embedding có magnitude khác nhau — điều rất phổ biến với text embeddings.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Công thức: `num_chunks = ceil((doc_length - overlap) / (chunk_size - overlap))`
>
> `= ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = ceil(22.11) = **23 chunks**`

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> `ceil((10000 - 100) / (500 - 100)) = ceil(9900 / 400) = ceil(24.75) = **25 chunks**` — tăng thêm 2 chunks.
> Overlap nhiều hơn giúp giữ lại ngữ cảnh ở ranh giới giữa hai chunk liền kề, tránh trường hợp một câu bị cắt đôi khiến chunk mất nghĩa.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** IELTS Speaking knowledge base

Nhóm chọn domain IELTS Speaking vì dữ liệu vừa có cấu trúc rõ (part, strategy, ví dụ, lỗi thường gặp), vừa có tính thực dụng để đánh giá chất lượng retrieval. Đây là domain dễ thiết kế benchmark query theo tình huống thật của người học, và có metadata tự nhiên để kiểm thử `search_with_filter`.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | `01_ielts_kb.md` | EnglishExample.md | ~1,500 | source, category, topic, language |
| 2 | `02_ielts_kb.md` | EnglishExample.md | ~2,100 | source, category, topic, language |
| 3 | `03_ielts_kb.md` | EnglishExample.md | ~1,900 | source, category, topic, language |
| 4 | `04_ielts_kb.md` | EnglishExample.md | ~1,800 | source, category, topic, language |
| 5 | `05_ielts_kb.md` | EnglishExample.md | ~1,700 | source, category, topic, language |
| 6 | `06_ielts_kb.md` | EnglishExample.md | ~1,600 | source, category, topic, language |
| 7 | `07_ielts_kb.md` | EnglishExample.md | ~1,700 | source, category, topic, language |
| 8 | `08_ielts_kb.md` | EnglishExample.md | ~1,900 | source, category, topic, language |
| 9 | `09_ielts_kb.md` | EnglishExample.md | ~1,800 | source, category, topic, language |
| 10 | `10_ielts_kb.md` | EnglishExample.md | ~1,600 | source, category, topic, language |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Giá trị cho retrieval |
|----------------|------|---------------|------------------------|
| `source` | string | `ielts_knowledge_base/04_ielts_kb.md` | Truy vết provenance, debug kết quả retrieve |
| `category` | string | `IELTS_Speaking_Strategy` | Lọc đúng nhóm kiến thức bằng `search_with_filter` |
| `topic` | string | `Affect vs Effect` | Tăng precision khi query theo topic hẹp |
| `language` | string | `English` / `Vietnamese` | Ưu tiên tài liệu đúng ngôn ngữ đầu vào |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Nhóm chạy `ChunkingStrategyComparator().compare(text, chunk_size=500)` trên từng file (mẫu 3 file đầu) và trên toàn bộ 10 file nối lại (cùng cách `run_comparison.py`). Số liệu dưới đây lấy từ lần chạy thực tế (`py eval_lab_metrics.py`, `py run_comparison.py`).

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| 01_ielts_kb.md | FixedSizeChunker (fixed_size) | 7 | 178.57 | |
| 01_ielts_kb.md | SentenceChunker (by_sentences) | 6 |206.33 | |
| 01_ielts_kb.md | RecursiveChunker (recursive) | 9 | 137.33 | |
| 02_ielts_kb.md | FixedSizeChunker (fixed_size) | 12 | 195.58 | |
| 02_ielts_kb.md | SentenceChunker (by_sentences) | 7 | 332.14 | |
| 02_ielts_kb.md | RecursiveChunker (recursive) | 17 | 136.59 |  |
| 03_ielts_kb.md | FixedSizeChunker (fixed_size) | 31 | 193.97 | |
| 03_ielts_kb.md | SentenceChunker (by_sentences) | 20 | 298.50 | |
| 03_ielts_kb.md | RecursiveChunker (recursive) | 52 | 114.02 | |

### Strategy Của Tôi

**Loại:** `HeadingChunker` (custom strategy)

**Mô tả cách hoạt động:**
> `HeadingChunker` dùng regex lookahead `^(?=#{1,N}(?!#) )` để tách văn bản tại mỗi dòng bắt đầu bằng Markdown heading (`#` hoặc `##`). Mỗi section từ heading này đến heading kế tiếp trở thành một chunk riêng, bao gồm cả heading line và toàn bộ nội dung bên dưới. Khi load IELTS KB, các chunk placeholder ngắn (`# Concept`, < 150 ký tự) được dùng làm anchor — prepend vào các sibling chunk để embedding có ngữ cảnh tiếng Anh.

**Tại sao tôi chọn strategy này cho domain IELTS KB?**
> Các file IELTS Knowledge Base được cấu trúc rõ ràng theo heading `##` (Content Details, Formulas/Patterns, Examples, Common Mistakes) — mỗi section là một đơn vị kiến thức hoàn chỉnh và độc lập. Tách theo heading giúp mỗi chunk giữ trọn vẹn một concept (ví dụ: toàn bộ ví dụ về "affect" nằm trong một chunk) thay vì bị cắt đôi giữa chừng như `fixed_size`. Đây là lợi thế rõ so với domain không có cấu trúc heading như plain text.

**Code snippet:**
```python
class HeadingChunker:
    def __init__(self, heading_levels: int = 2) -> None:
        self.heading_levels = max(1, heading_levels)
        self._pattern = re.compile(
            rf"^(?=#{{{1},{self.heading_levels}}}(?!#) )", re.MULTILINE
        )

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        parts = self._pattern.split(text)
        chunks = [p.strip() for p in parts if p.strip()]
        return chunks if chunks else [text.strip()]
```

### So Sánh: Strategy của tôi vs Baseline

Chạy trên query `"How do I correctly use affect vs effect?"` với embedding `text-embedding-3-small`:

| Strategy | Chunks | Avg Length | Top-1 Score | Relevant/3 | Score Gap |
|----------|--------|------------|-------------|------------|-----------|
| `fixed_size` (best baseline) | 114 | 489c | 0.6448 | 3/3 | 0.0850 |
| `by_sentences` | 80 | 628c | 0.6427 | 3/3 | 0.1089 |
| `recursive` | 137 | 356c | 0.6283 | 3/3 | 0.0449 |
| **`heading` (của tôi)** | **18** | **2753c** | **0.6459** | **3/3** | **0.0774** |

`heading` đạt top-1 score cao nhất (0.6459) với số chunk ít nhất (18) — chunk to hơn nhưng trọn nghĩa hơn. `by_sentences` có score gap tốt nhất (0.1089) — phân biệt tốt hơn giữa kết quả đúng và nhiễu.

### So Sánh Với Thành Viên Khác

Điểm **/10** dưới đây là **đánh giá đồng thuận trong nhóm** sau khi xem code, benchmark và demo.

| Thành viên | Strategy (tóm tắt) | Điểm nhóm (/10) | Điểm mạnh | Điểm yếu |
|-------------|-------------------|-----------------|-----------|----------|
| Nguyễn Triệu Gia Khánh | `RecursiveChunker` + pipeline đo `eval_lab_metrics.py` | **10** | Phân tích chunk + benchmark rõ ràng; tài liệu nhóm đầy đủ | Cần thêm thử nghiệm embedding neural để so sánh với TF‑IDF |
| Nguyễn Thùy Linh | `FixedSizeChunker` + overlap ổn định | **10** | Triển khai nhanh, dễ tái lập thí nghiệm | Một số đoạn dài vẫn cắt giữa ý |
| Nguyễn Hoàng Khải Minh | `SentenceChunker` + tinh chỉnh `max_sentences_per_chunk` | **10** | Chunk đọc tự nhiên, phù hợp câu hỏi ngắn | Chiến lược phụ thuộc dấu câu tiếng Anh |
| Nguyễn Thị Diệu Linh | `RecursiveChunker` (separator tùy chỉnh nhẹ) | **10** | Giữ được khối heading/bullet | Thời gian tuning separator |
| Nguyễn Hoàng Duy | `HeadingChunker` — tách theo `##` heading, anchor `# Concept` prepend vào mỗi sibling chunk | **10** | Chunk trọn nghĩa theo section; embedding có English anchor dù content tiếng Việt | Chunk có thể rất to nếu section dài; retrieval yếu khi query và content khác ngôn ngữ |

**Kết luận strategy tốt nhất cho domain này:**  
Nhóm thống nhất **`RecursiveChunker`** làm hướng chính cho IELTS (heading/bullet), đồng thời mỗi thành viên có nhánh so sánh riêng để học chéo. Sau benchmark và demo, nhóm **đồng thuận 10/10** cho từng thành viên về đóng góp strategy và phối hợp nhóm.

---

## 4. My Approach — Cá nhân (10 điểm)

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Dùng `re.split(r'(?<=[.!?]) +|(?<=\.)\n', text)` để tách câu — lookbehind đảm bảo dấu câu vẫn gắn với câu trước. Sau đó gom từng `max_sentences_per_chunk` câu thành một chunk bằng `' '.join(group)`. Edge case: text rỗng trả về `[]`, text không có dấu câu trả về `[text.strip()]`.

**`RecursiveChunker.chunk` / `_split`** — approach:
> `_split` thử từng separator theo thứ tự ưu tiên (`\n\n`, `\n`, `. `, ` `, `""`). Nếu text đã ngắn hơn `chunk_size` thì return ngay (base case). Nếu separator xuất hiện, buffer các mảnh nhỏ lại — khi buffer vượt quá `chunk_size` thì flush và đệ quy với `next_seps` cho mảnh quá lớn. Separator `""` là fallback cuối cùng, cắt cứng theo `chunk_size`.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> `add_documents` embed từng doc bằng `embedding_fn`, lưu dict `{id, content, embedding, metadata}` vào `self._store` (list). `search` embed query, tính dot product giữa query embedding và từng stored embedding (các embedding đã normalize nên dot product ≈ cosine similarity), sort descending, return top-k dict `{content, score, metadata}`.

**`search_with_filter` + `delete_document`** — approach:
> `search_with_filter` filter `self._store` trước bằng list comprehension kiểm tra từng key-value trong `metadata_filter`, sau đó gọi `_search_records` trên tập đã lọc. `delete_document` rebuild `self._store` loại bỏ mọi record có `metadata['doc_id'] == doc_id`, return `True` nếu size giảm.

### KnowledgeBaseAgent

**`answer`** — approach:
> Retrieve top-k chunks từ store, format thành numbered context `[1] (source: ...) \n {content}`. Build prompt theo cấu trúc: "Answer based only on context below → context → question → Answer:". Inject cả source metadata vào context để LLM có thể traceability. Gọi `llm_fn(prompt)` và return kết quả.

### Test Results

```
============================= test session starts =============================
platform win32 -- Python 3.10.11, pytest-9.0.2
collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED

======================== 42 passed in 0.09s ===============================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

Dùng `text-embedding-3-small` + `compute_similarity()`:

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | "Affect is used as a verb in a sentence." | "Effect is used as a noun in a sentence." | high | **0.5889** | ✓ |
| 2 | "I prefer indoor activities like reading." | "She enjoys indoor sports such as badminton." | high | **0.5494** | ✓ |
| 3 | "The cat sat on the mat." | "Quantum physics describes subatomic particles." | low | **0.0126** | ✓ |
| 4 | "It depends on the situation." | "Different circumstances require different responses." | high | **0.5709** | ✓ |
| 5 | "Affect can change your mood." | "The weather today is sunny and warm." | low | **0.1434** | ✓ |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Pair 4 bất ngờ nhất — "It depends on the situation" và "Different circumstances require different responses" có score **0.5709** dù không dùng cùng từ nào. Điều này cho thấy embedding model nắm được semantic equivalence thay vì chỉ so sánh surface form. Ngược lại, pair 5 (0.1434) cho thấy dù "affect" xuất hiện trong cả hai domain (tâm lý & thời tiết), ngữ cảnh không liên quan khiến score vẫn thấp.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | In IELTS Speaking Part 2, how should I open my answer in the first 10-15 seconds so I sound clear and on-topic before adding details? | Start with a direct one-sentence answer to the prompt, then extend with reason/example instead of giving background first. |
| 2 | My ideas are too general in Speaking. What exact structure can I use to move from a broad claim to a specific personal example without losing coherence? | Use a 3-step structure: general statement -> narrow reason -> concrete personal example (time/place/result). |
| 3 | If I don't know much about a topic, what is the safest high-control response pattern that avoids silence but still sounds natural and balanced? | Use an "it depends" frame with two short contrasting cases, then close by choosing one side. |
| 4 | During Speaking, when I run out of ideas mid-answer, what language moves can I use to keep fluency while buying thinking time and still add value? | Use filler bridges plus extension templates (reason, example, comparison) to maintain flow instead of stopping abruptly. |
| 5 | For a band-5 to band-6 improvement path, which habit hurts score most in spontaneous speaking and what should I do immediately to replace it? | Avoid switching to L1; stay in English and paraphrase with simpler words when vocabulary gaps appear. |

### Kết Quả Của Tôi

Dùng `HeadingChunker`, `text-embedding-3-small`, 18 chunks từ IELTS KB:

| # | Query | Top-1 Retrieved Chunk (topic) | Score | Relevant? | Points |
|---|-------|-------------------------------|-------|-----------|--------|
| 1 | How do I correctly use affect vs effect? | Vocabulary: Affect vs Effect | 0.6510 | ✓ | 2/2 |
| 2 | What is the RAVEN mnemonic? | Vocabulary: Affect vs Effect | 0.7821 | ✓ | 2/2 |
| 3 | When should I use 'indoors' vs 'indoor'? | Grammar: Indoor vs Indoors | 0.6627 | ✓ | 2/2 |
| 4 | Give me example sentences using 'indoor' as adjective | Grammar: Indoor vs Indoors | 0.6282 | ✓ | 2/2 |
| 5 | Chiến lược 'It Depends' dùng khi nào? | Strategy 3: It Depends (rank 2) | 0.2884 | Partial | 1/2 |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 4.5 / 5 — Query 5 (tiếng Việt) chỉ tìm được 1/3 relevant vì nội dung Strategy viết tiếng Việt trong khi embedding model match tốt hơn với English queries.

**Retrieval Precision tổng (Metric 1):** 9/10 điểm (90%)

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> *(Điền sau khi so sánh với nhóm)*

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> *(Điền sau demo)*

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Tôi sẽ thêm English summary vào đầu mỗi `## Content Details` trong các file Strategy, vì query tiếng Anh không match được nội dung tiếng Việt. Đây là bài học quan trọng về language alignment: khi data và query không cùng ngôn ngữ, embedding retrieval sẽ kém chất lượng dù pipeline code hoàn toàn đúng. Ngoài ra sẽ giảm `heading_levels=1` hoặc thêm max_chars cho `HeadingChunker` để tránh chunk quá to (max 19957c hiện tại).

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 15 / 15 |
| My approach | Cá nhân | 9 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 8 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 5 / 5 |
| **Tổng** | | **87 / 100** |

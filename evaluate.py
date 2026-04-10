"""
evaluate.py — Đánh giá retrieval quality theo docs/EVALUATION.md

Chạy:
    python evaluate.py

Các metric được đo:
    1. Retrieval Precision   — top-3 relevant / score distribution
    2. Chunk Coherence       — count, avg_length, mid-sentence cut rate
    3. Metadata Utility      — search vs search_with_filter
    4. Grounding Quality     — agent answer có dùng context không
    5. Data Strategy Impact  — so sánh 4 chunking strategies
"""

from __future__ import annotations

import os
import re
import textwrap
from pathlib import Path

from dotenv import load_dotenv

from src.chunking import (
    ChunkingStrategyComparator,
    FixedSizeChunker,
    HeadingChunker,
    RecursiveChunker,
    SentenceChunker,
)
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore
from src.agent import KnowledgeBaseAgent

IELTS_KB_DIR = Path("data/ielts_knowledge_base")

# ---------------------------------------------------------------------------
# Benchmark queries với expected_topics (ground truth)
# Mỗi query có:
#   - query: câu hỏi
#   - expected_topics: list topic keywords phải xuất hiện trong top-3
#   - gold_answer_keywords: từ khóa phải có trong agent answer
#   - filter: metadata filter để test Metric 3 (None = không filter)
# ---------------------------------------------------------------------------
BENCHMARK = [
    {
        "query": "How do I correctly use affect vs effect in a sentence?",
        "expected_topics": ["Affect vs Effect"],
        "gold_answer_keywords": ["verb", "noun", "affect", "effect"],
        "filter": {"topic": "Vocabulary: Affect vs Effect"},
    },
    {
        "query": "What is the RAVEN mnemonic and what does each letter mean?",
        "expected_topics": ["Affect vs Effect"],
        "gold_answer_keywords": ["RAVEN", "Affect", "Verb", "Effect", "Noun"],
        "filter": None,
    },
    {
        "query": "When should I use 'indoors' vs 'indoor'?",
        "expected_topics": ["Indoor vs Indoors"],
        "gold_answer_keywords": ["adjective", "adverb"],
        "filter": {"topic": "Grammar: Indoor vs Indoors"},
    },
    {
        "query": "Give me example sentences using 'indoor' as an adjective before a noun.",
        "expected_topics": ["Indoor vs Indoors"],
        "gold_answer_keywords": ["indoor", "noun"],
        "filter": None,
    },
    {
        "query": "Chiến lược 'It Depends' dùng khi nào và cách mở bài như thế nào?",
        "expected_topics": ["Strategy 3", "It Depends"],
        "gold_answer_keywords": ["depends", "tình huống"],
        "filter": None,
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pick_embedder():
    load_dotenv(override=False)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    if provider == "local":
        try:
            return LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception:
            pass
    elif provider == "openai":
        try:
            return OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        except Exception:
            pass
    return _mock_embed


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    lines = text.splitlines()
    meta: dict = {}
    body_start = 0
    for i, line in enumerate(lines):
        if ":" in line and not line.startswith("#"):
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
            body_start = i + 1
        else:
            break
    return meta, "\n".join(lines[body_start:]).strip()


def load_ielts_docs(chunker, kb_dir: Path = IELTS_KB_DIR) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(kb_dir.glob("*.md")):
        if path.name == "EnglishExample.md":
            continue
        raw = path.read_text(encoding="utf-8")
        meta, body = _parse_frontmatter(raw)
        meta.setdefault("source", path.name)
        chunks = chunker.chunk(body)
        anchor = ""
        for chunk in chunks:
            if len(chunk.strip()) < 150 and chunk.lstrip().startswith("# "):
                anchor = chunk.strip()
            else:
                if len(chunk.strip()) < 120:
                    continue
                content = (anchor + "\n\n" + chunk) if anchor else chunk
                docs.append(Document(
                    id=f"{path.stem}_c{len(docs)}",
                    content=content,
                    metadata=meta,
                ))
    return docs


def _is_relevant(result: dict, expected_topics: list[str]) -> bool:
    topic = result["metadata"].get("topic", "")
    return any(kw.lower() in topic.lower() for kw in expected_topics)


def _wrap(text: str, width: int = 90, indent: str = "    ") -> str:
    return textwrap.fill(text.replace("\n", " "), width=width, initial_indent=indent,
                         subsequent_indent=indent)


def _section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Metric 1 — Retrieval Precision
# ---------------------------------------------------------------------------

def eval_retrieval_precision(store: EmbeddingStore) -> float:
    _section("METRIC 1: Retrieval Precision")
    total_score = 0
    max_score = len(BENCHMARK) * 2  # 2 pts per query

    for bm in BENCHMARK:
        results = store.search(bm["query"], top_k=3)
        relevant = [r for r in results if _is_relevant(r, bm["expected_topics"])]
        n_relevant = len(relevant)

        # Score: 2 = ≥2 relevant in top-3, 1 = exactly 1, 0 = none
        score = 2 if n_relevant >= 2 else (1 if n_relevant == 1 else 0)
        total_score += score

        scores = [r["score"] for r in results]
        score_gap = (scores[0] - scores[-1]) if len(scores) >= 2 else 0.0

        print(f"\n  Query: {bm['query'][:65]}...")
        print(f"  Expected topic: {bm['expected_topics']}")
        print(f"  Top-3 topics  : {[r['metadata'].get('topic','?') for r in results]}")
        print(f"  Relevant hits : {n_relevant}/3  |  Score gap: {score_gap:.4f}  |  Points: {score}/2")

    pct = total_score / max_score * 100
    print(f"\n  TOTAL: {total_score}/{max_score} ({pct:.0f}%)")
    return total_score / max_score


# ---------------------------------------------------------------------------
# Metric 2 — Chunk Coherence
# ---------------------------------------------------------------------------

def eval_chunk_coherence(docs: list[Document]):
    _section("METRIC 2: Chunk Coherence")

    lengths = [len(d.content) for d in docs]
    avg = sum(lengths) / len(lengths) if lengths else 0
    min_l, max_l = min(lengths), max(lengths)

    # Mid-sentence cut heuristic: chunk ends without sentence-ending punctuation
    cut_count = sum(
        1 for d in docs
        if d.content.strip() and d.content.strip()[-1] not in ".!?»\"'"
    )
    cut_rate = cut_count / len(docs) * 100 if docs else 0

    print(f"\n  Chunks loaded  : {len(docs)}")
    print(f"  Avg length     : {avg:.1f} chars")
    print(f"  Min / Max      : {min_l} / {max_l} chars")
    print(f"  Mid-cut rate   : {cut_count}/{len(docs)} chunks ({cut_rate:.1f}%) end without sentence boundary")
    print(f"  Assessment     : {'✓ Good coherence' if cut_rate < 30 else '⚠ High cut rate — consider larger chunks'}")


# ---------------------------------------------------------------------------
# Metric 3 — Metadata Utility
# ---------------------------------------------------------------------------

def eval_metadata_utility(store: EmbeddingStore):
    _section("METRIC 3: Metadata Utility")

    filtered_bms = [bm for bm in BENCHMARK if bm["filter"]]
    gains = 0

    for bm in filtered_bms:
        unfiltered = store.search(bm["query"], top_k=3)
        filtered = store.search_with_filter(bm["query"], top_k=3,
                                             metadata_filter=bm["filter"])

        rel_unf = sum(1 for r in unfiltered if _is_relevant(r, bm["expected_topics"]))
        rel_fil = sum(1 for r in filtered if _is_relevant(r, bm["expected_topics"]))
        if rel_fil > rel_unf:
            gains += 1

        print(f"\n  Query : {bm['query'][:60]}...")
        print(f"  Filter: {bm['filter']}")
        print(f"  Without filter: {rel_unf}/3 relevant")
        print(f"  With filter   : {rel_fil}/3 relevant  {'✓ Filter helped' if rel_fil > rel_unf else '– No change' if rel_fil == rel_unf else '⚠ Filter hurt recall'}")
        if not filtered:
            print(f"  ⚠  Filter returned 0 results — filter may be too strict")

    print(f"\n  Filter helped in {gains}/{len(filtered_bms)} queries")


# ---------------------------------------------------------------------------
# Metric 4 — Grounding Quality
# ---------------------------------------------------------------------------

def eval_grounding(store: EmbeddingStore):
    _section("METRIC 4: Grounding Quality")

    def mock_llm(prompt: str) -> str:
        # Extract the context section from the prompt for keyword checking
        return prompt  # return full prompt so we can inspect grounding

    agent = KnowledgeBaseAgent(store=store, llm_fn=mock_llm)

    for bm in BENCHMARK[:3]:  # only first 3 to keep output concise
        results = store.search(bm["query"], top_k=3)
        context_text = " ".join(r["content"] for r in results).lower()
        keywords_found = [kw for kw in bm["gold_answer_keywords"]
                          if kw.lower() in context_text]
        coverage = len(keywords_found) / len(bm["gold_answer_keywords"]) * 100

        print(f"\n  Query   : {bm['query'][:60]}...")
        print(f"  Gold KWs: {bm['gold_answer_keywords']}")
        print(f"  Found   : {keywords_found}")
        print(f"  Context coverage: {coverage:.0f}%  {'✓' if coverage >= 50 else '⚠'}")


# ---------------------------------------------------------------------------
# Metric 5 — Data Strategy Impact (compare 4 chunking strategies)
# ---------------------------------------------------------------------------

def eval_strategy_impact(embedder):
    _section("METRIC 5: Data Strategy Impact — Chunking Comparison")

    strategies = {
        "fixed_size":   FixedSizeChunker(chunk_size=500, overlap=50),
        "by_sentences": SentenceChunker(max_sentences_per_chunk=5),
        "recursive":    RecursiveChunker(chunk_size=500),
        "heading":      HeadingChunker(heading_levels=2),
    }

    query = BENCHMARK[0]["query"]  # use first benchmark query
    expected = BENCHMARK[0]["expected_topics"]

    print(f"\n  Query: {query}")
    print(f"  Expected topic: {expected}\n")
    print(f"  {'Strategy':<16} {'Chunks':>6} {'AvgLen':>7} {'Top1 Score':>10} {'Rel/3':>6} {'Score gap':>10}")
    print(f"  {'-'*58}")

    for name, chunker in strategies.items():
        docs = load_ielts_docs(chunker)
        store = EmbeddingStore(collection_name=f"eval_{name}", embedding_fn=embedder)
        store.add_documents(docs)

        results = store.search(query, top_k=3)
        relevant = sum(1 for r in results if _is_relevant(r, expected))
        top1_score = results[0]["score"] if results else 0.0
        gap = (results[0]["score"] - results[-1]["score"]) if len(results) >= 2 else 0.0
        avg_len = sum(len(d.content) for d in docs) / len(docs) if docs else 0

        print(f"  {name:<16} {len(docs):>6} {avg_len:>7.0f} {top1_score:>10.4f} {relevant:>6} {gap:>10.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 60)
    print("  LAB 7 — Retrieval Quality Evaluation")
    print("  Theo docs/EVALUATION.md")
    print("=" * 60)

    embedder = _pick_embedder()
    print(f"\nEmbedding backend: {getattr(embedder, '_backend_name', embedder.__class__.__name__)}")

    # Build store with HeadingChunker (current strategy)
    chunker = HeadingChunker(heading_levels=2)
    docs = load_ielts_docs(chunker)
    store = EmbeddingStore(collection_name="eval_main", embedding_fn=embedder)
    store.add_documents(docs)
    print(f"Loaded {len(docs)} chunks into store\n")

    eval_retrieval_precision(store)
    eval_chunk_coherence(docs)
    eval_metadata_utility(store)
    eval_grounding(store)
    eval_strategy_impact(embedder)

    print("\n" + "=" * 60)
    print("  Done. Ghi kết quả vào report/REPORT.md")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

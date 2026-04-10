from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.agent import KnowledgeBaseAgent
from src.chunking import HeadingChunker
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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

IELTS_KB_DIR = Path("data/ielts_knowledge_base")

SAMPLE_FILES = [
    "data/python_intro.txt",
    "data/vector_store_notes.md",
    "data/rag_system_design.md",
    "data/customer_support_playbook.txt",
    "data/chunking_experiment_report.md",
    "data/vi_retrieval_notes.md",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pick_embedder():
    """Select embedding backend based on env var; fall back to mock."""
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
    """
    Parse the leading key: value frontmatter block (no --- delimiters).
    Returns (metadata_dict, remaining_content).
    """
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
    body = "\n".join(lines[body_start:]).strip()
    return meta, body


def load_ielts_kb(kb_dir: Path, chunker: HeadingChunker) -> list[Document]:
    """
    Load all .md files in kb_dir, parse frontmatter as metadata,
    then chunk by headings. Each chunk becomes one Document.
    """
    docs: list[Document] = []
    for path in sorted(kb_dir.glob("*.md")):
        if path.name == "EnglishExample.md":
            continue
        raw = path.read_text(encoding="utf-8")
        meta, body = _parse_frontmatter(raw)
        meta.setdefault("source", path.name)

        chunks = chunker.chunk(body)
        for idx, chunk in enumerate(chunks):
            doc_id = f"{path.stem}_chunk{idx}"
            docs.append(Document(id=doc_id, content=chunk, metadata=meta))

    return docs


def load_documents_from_files(file_paths: list[str]) -> list[Document]:
    """Load plain .txt / .md files without chunking (original helper)."""
    allowed = {".md", ".txt"}
    docs: list[Document] = []
    for raw_path in file_paths:
        path = Path(raw_path)
        if path.suffix.lower() not in allowed:
            print(f"  Skipping unsupported type: {path}")
            continue
        if not path.exists():
            print(f"  Skipping missing file: {path}")
            continue
        content = path.read_text(encoding="utf-8")
        docs.append(Document(
            id=path.stem,
            content=content,
            metadata={"source": str(path), "extension": path.suffix.lower()},
        ))
    return docs


def demo_llm(prompt: str) -> str:
    """Mock LLM — echoes prompt. Used when no real LLM is available."""
    return f"[DEMO LLM — no real LLM configured]\nPrompt sent:\n{prompt}"


def make_openai_llm(model: str = "gpt-4o-mini"):
    """Return an LLM function backed by OpenAI chat completions."""
    from openai import OpenAI
    client = OpenAI()

    def _call(prompt: str) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content

    return _call


# ---------------------------------------------------------------------------
# Demo modes
# ---------------------------------------------------------------------------

def run_ielts_demo(question: str | None = None) -> int:
    print("=" * 60)
    print("IELTS Knowledge Base — HeadingChunker + RAG Demo")
    print("=" * 60)

    chunker = HeadingChunker(heading_levels=2)
    docs = load_ielts_kb(IELTS_KB_DIR, chunker)

    if not docs:
        print(f"\nNo documents found in {IELTS_KB_DIR}")
        return 1

    print(f"\nLoaded {len(docs)} chunks from {IELTS_KB_DIR.name}/")

    embedder = _pick_embedder()
    print(f"Embedding backend: {getattr(embedder, '_backend_name', embedder.__class__.__name__)}")

    store = EmbeddingStore(collection_name="ielts_kb", embedding_fn=embedder)
    store.add_documents(docs)
    print(f"Stored {store.get_collection_size()} chunks in EmbeddingStore\n")

    # --- Basic search ---
    query = question or "If I don't know much about a topic, what is the safest high-control response pattern that avoids silence but still sounds natural and balanced?"
    print(f"[Search] Query: {query}")
    results = store.search(query, top_k=3)
    for i, r in enumerate(results, 1):
        topic = r["metadata"].get("topic", "?")
        score = r["score"]
        print(f"  {i}. score={score:.4f} | topic={topic}")
        print(f"     {r['content']}")

    # --- Metadata filter search ---
    print(f"\n[Filter Search] category=IELTS_Speaking_Strategy")
    filtered = store.search_with_filter(
        query,
        top_k=3,
        metadata_filter={"category": "IELTS_Speaking_Strategy"},
    )
    for i, r in enumerate(filtered, 1):
        topic = r["metadata"].get("topic", "?")
        print(f"  {i}. score={r['score']:.4f} | topic={topic}")

    # --- Agent RAG answer ---
    print(f"\n[Agent] Question: {query}")
    try:
        llm_fn = make_openai_llm()
    except Exception:
        llm_fn = demo_llm
    agent = KnowledgeBaseAgent(store=store, llm_fn=llm_fn)
    print(agent.answer(query, top_k=3))

    return 0


def run_manual_demo(question: str | None = None, sample_files: list[str] | None = None) -> int:
    files = sample_files or SAMPLE_FILES
    query = question or "Summarize the key information from the loaded files."

    print("=" * 60)
    print("Manual File Demo")
    print("=" * 60)
    docs = load_documents_from_files(files)
    if not docs:
        print("\nNo valid input files loaded.")
        return 1

    print(f"\nLoaded {len(docs)} documents")
    for doc in docs:
        print(f"  - {doc.id}: {doc.metadata['source']}")

    embedder = _pick_embedder()
    print(f"\nEmbedding backend: {getattr(embedder, '_backend_name', embedder.__class__.__name__)}")

    store = EmbeddingStore(collection_name="manual_test_store", embedding_fn=embedder)
    store.add_documents(docs)
    print(f"Stored {store.get_collection_size()} documents in EmbeddingStore")

    print(f"\n[Search] Query: {query}")
    for i, r in enumerate(store.search(query, top_k=3), 1):
        src = r["metadata"].get("source", "?")
        print(f"  {i}. score={r['score']:.3f} | {src}")
        print(f"     {r['content']}")

    print("\n[Agent]")
    agent = KnowledgeBaseAgent(store=store, llm_fn=demo_llm)
    print(agent.answer(query, top_k=3))
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    args = sys.argv[1:]
    if args and args[0] == "--manual":
        question = " ".join(args[1:]).strip() or None
        return run_manual_demo(question=question)

    question = " ".join(args).strip() or None
    return run_ielts_demo(question=question)


if __name__ == "__main__":
    raise SystemExit(main())

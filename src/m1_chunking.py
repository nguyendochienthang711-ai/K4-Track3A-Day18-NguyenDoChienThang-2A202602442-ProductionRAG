from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


# Cache embedding model for semantic chunking
_SEMANTIC_MODEL = None


def _get_semantic_model():
    global _SEMANTIC_MODEL
    if _SEMANTIC_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _SEMANTIC_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _SEMANTIC_MODEL


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    metadata = metadata or {}
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n\n+', text) if s.strip()]
    if not sentences:
        return []
    if len(sentences) == 1:
        return [Chunk(text=sentences[0], metadata={**metadata, "strategy": "semantic"})]

    from numpy import dot
    from numpy.linalg import norm

    model = _get_semantic_model()
    embeddings = model.encode(sentences)

    def cosine_sim(a, b):
        return float(dot(a, b) / (norm(a) * norm(b) + 1e-9))

    chunks = []
    current_sentences = [sentences[0]]
    for i in range(1, len(sentences)):
        sim = cosine_sim(embeddings[i - 1], embeddings[i])
        if sim < threshold:
            chunks.append(Chunk(text=" ".join(current_sentences), metadata={**metadata, "strategy": "semantic"}))
            current_sentences = [sentences[i]]
        else:
            current_sentences.append(sentences[i])
    if current_sentences:
        chunks.append(Chunk(text=" ".join(current_sentences), metadata={**metadata, "strategy": "semantic"}))
    return chunks


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def _split_text_into_chunks(text: str, max_size: int) -> list[str]:
    """Split text into chunks of at most max_size characters."""
    if len(text) <= max_size:
        return [text] if text.strip() else []

    parts = [p.strip() for p in re.split(r'(?<=[.!?])\s+|\n+', text) if p.strip()]
    chunks = []
    curr = ""
    for part in parts:
        if len(curr) + len(part) + 1 > max_size and curr:
            chunks.append(curr.strip())
            curr = ""
        if len(part) > max_size:
            words = part.split()
            w_curr = ""
            for w in words:
                if len(w_curr) + len(w) + 1 > max_size and w_curr:
                    chunks.append(w_curr.strip())
                    w_curr = ""
                w_curr += (" " if w_curr else "") + w
            if w_curr:
                curr = w_curr
        else:
            curr += (" " if curr else "") + part
    if curr.strip():
        chunks.append(curr.strip())
    return chunks


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    raw_paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not raw_paras:
        return ([], [])

    # Group paragraphs into parent chunks <= parent_size
    parent_texts = []
    current_parent = ""
    for p in raw_paras:
        if len(current_parent) + len(p) + 2 > parent_size and current_parent:
            parent_texts.append(current_parent.strip())
            current_parent = ""
        if len(p) > parent_size:
            sub_parents = _split_text_into_chunks(p, parent_size)
            parent_texts.extend(sub_parents)
        else:
            current_parent += (("\n\n" if current_parent else "") + p)
    if current_parent.strip():
        parent_texts.append(current_parent.strip())

    parents = []
    children = []
    for idx, p_text in enumerate(parent_texts):
        pid = f"parent_{idx}"
        parents.append(Chunk(
            text=p_text,
            metadata={**metadata, "chunk_type": "parent", "parent_id": pid}
        ))
        child_texts = _split_text_into_chunks(p_text, child_size)
        for c_text in child_texts:
            children.append(Chunk(
                text=c_text,
                metadata={**metadata, "chunk_type": "child"},
                parent_id=pid
            ))

    return (parents, children)


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    metadata = metadata or {}
    tokens = re.split(r'(^#{1,3}\s+[^\n]+)', text, flags=re.MULTILINE)

    chunks = []
    current_header = ""
    current_body = ""

    for token in tokens:
        stripped = token.strip()
        if not stripped:
            continue
        if re.match(r'^#{1,3}\s+', stripped):
            if current_body.strip():
                content = f"{current_header}\n\n{current_body.strip()}".strip() if current_header else current_body.strip()
                chunks.append(Chunk(
                    text=content,
                    metadata={**metadata, "section": current_header if current_header else "General", "strategy": "structure"}
                ))
                current_header = stripped
                current_body = ""
            else:
                if current_header:
                    current_header = f"{current_header}\n{stripped}"
                else:
                    current_header = stripped
        else:
            current_body += "\n" + stripped

    if current_header or current_body.strip():
        content = f"{current_header}\n\n{current_body.strip()}".strip() if current_header else current_body.strip()
        if content:
            chunks.append(Chunk(
                text=content,
                metadata={**metadata, "section": current_header if current_header else "General", "strategy": "structure"}
            ))

    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")

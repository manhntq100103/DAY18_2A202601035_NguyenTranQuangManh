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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_DIR,
    HIERARCHICAL_PARENT_SIZE,
    HIERARCHICAL_CHILD_SIZE,
    SEMANTIC_THRESHOLD,
)

embedding_model_cache_folder = "./my_models/"
_semantic_model = None


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
            docs.append(
                {"text": f.read(), "metadata": {"source": os.path.basename(fp)}}
            )

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(
                f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR)."
            )

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(
    text: str, chunk_size: int = 500, metadata: dict | None = None
) -> list[Chunk]:
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
            chunks.append(
                Chunk(
                    text=current.strip(),
                    metadata={**metadata, "chunk_index": len(chunks)},
                )
            )
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(
            Chunk(
                text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}
            )
        )
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(
    text: str, threshold: float = SEMANTIC_THRESHOLD, metadata: dict | None = None
) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    global _semantic_model
    metadata = metadata or {}
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n\n", text) if s.strip()]
    if not sentences:
        return []
    if len(sentences) == 1:
        return [
            Chunk(sentences[0], {**metadata, "strategy": "semantic", "chunk_index": 0})
        ]

    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        if _semantic_model is None:
            _semantic_model = SentenceTransformer(
                "all-MiniLM-L6-v2", cache_folder=embedding_model_cache_folder
            )
        embeddings = np.asarray(_semantic_model.encode(sentences))
        embeddings = embeddings / np.maximum(
            np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-9
        )
        similarities = (embeddings[:-1] * embeddings[1:]).sum(axis=1)
    except Exception:
        tokenized = [re.findall(r"\w+", s.lower()) for s in sentences]
        vocabulary = {word for words in tokenized for word in words}
        index = {word: i for i, word in enumerate(vocabulary)}
        vectors = []
        for words in tokenized:
            vector = [0.0] * len(index)
            for word in words:
                vector[index[word]] += 1.0
            length = sum(value * value for value in vector) ** 0.5 or 1.0
            vectors.append([value / length for value in vector])
        similarities = [
            sum(a * b for a, b in zip(vectors[i], vectors[i + 1]))
            for i in range(len(vectors) - 1)
        ]

    groups, current = [], [sentences[0]]
    for sentence, similarity in zip(sentences[1:], similarities):
        if similarity < threshold:
            groups.append(current)
            current = [sentence]
        else:
            current.append(sentence)
    groups.append(current)
    return [
        Chunk(
            "\n\n".join(group), {**metadata, "strategy": "semantic", "chunk_index": i}
        )
        for i, group in enumerate(groups)
    ]


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(
    text: str,
    parent_size: int = HIERARCHICAL_PARENT_SIZE,
    child_size: int = HIERARCHICAL_CHILD_SIZE,
    metadata: dict | None = None,
) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    if parent_size <= 0 or child_size <= 0:
        raise ValueError("parent_size and child_size must be positive")

    base_metadata = dict(metadata or {})
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return [], []

    # Split oversized paragraphs before grouping so every parent respects the
    # configured limit. Normal paragraphs remain intact where possible.
    units = []
    for paragraph in paragraphs:
        if len(paragraph) <= parent_size:
            units.append(paragraph)
        else:
            units.extend(
                paragraph[i : i + parent_size]
                for i in range(0, len(paragraph), parent_size)
            )

    parent_texts, current, current_len = [], [], 0
    for unit in units:
        added = len(unit) if not current else 2 + len(unit)
        if current and current_len + added > parent_size:
            parent_texts.append("\n\n".join(current))
            current, current_len = [], 0
        current.append(unit)
        current_len += len(unit) if len(current) == 1 else 2 + len(unit)
    if current:
        parent_texts.append("\n\n".join(current))

    parents, children = [], []
    for parent_index, parent_text in enumerate(parent_texts):
        parent_id = f"parent_{parent_index}"
        parents.append(
            Chunk(
                parent_text,
                {
                    **base_metadata,
                    "chunk_type": "parent",
                    "parent_id": parent_id,
                    "chunk_index": parent_index,
                },
                parent_id=parent_id,
            )
        )
        for child_index, start in enumerate(range(0, len(parent_text), child_size)):
            children.append(
                Chunk(
                    parent_text[start : start + child_size],
                    {
                        **base_metadata,
                        "chunk_type": "child",
                        "parent_id": parent_id,
                        "chunk_index": child_index,
                    },
                    parent_id=parent_id,
                )
            )
    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    # 3. Duyệt sections:
    #      - Nếu match header (^#{1,3}\s+): lưu header hiện tại, tạo chunk cho content trước đó
    #      - Else: gộp vào content hiện tại
    # 4. Return [Chunk(text=header+content, metadata={..., "section": header, "strategy": "structure"})]
    base_metadata = dict(metadata or {})
    if not text or not text.strip():
        return []

    heading_re = re.compile(r"^#{1,3}\s+.+$")
    fence_re = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
    sections: list[tuple[str, list[str]]] = []
    current_header = ""
    current_lines: list[str] = []
    fence_char: str | None = None

    for line in text.splitlines():
        fence = fence_re.match(line)
        if fence:
            marker = fence.group(1)
            if fence_char is None:
                fence_char = marker[0]
            elif marker[0] == fence_char:
                fence_char = None
            current_lines.append(line)
            continue

        if fence_char is None and heading_re.match(line):
            if current_header or any(item.strip() for item in current_lines):
                sections.append((current_header, current_lines))
            current_header = line.strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_header or any(item.strip() for item in current_lines):
        sections.append((current_header, current_lines))

    chunks: list[Chunk] = []
    for header, lines in sections:
        chunk_text = "\n".join(lines).strip()
        if not chunk_text:
            continue
        chunks.append(
            Chunk(
                chunk_text,
                {
                    **base_metadata,
                    "section": header,
                    "strategy": "structure",
                    "chunk_index": len(chunks),
                },
            )
        )
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
        print(
            f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}"
        )

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")

from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""

    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    text = text.strip()
    if not text:
        return ""

    if OPENAI_API_KEY:
        try:
            from openai import OpenAI

            response = OpenAI(
                api_key=OPENAI_API_KEY, base_url="https://api.groq.com/openai/v1"
            ).chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {
                        "role": "system",
                        "content": "Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt.",
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
            )
            summary = response.choices[0].message.content
            if summary:
                return summary.strip()
        except Exception as exc:
            print(f"  ⚠️ OpenAI summarize failed: {exc}")

    # Fast extractive fallback for offline runs: retain the first two sentences.
    import re

    sentences = [
        part.strip() for part in re.split(r"(?<=[.!?])\s+|\r?\n+", text) if part.strip()
    ]
    if not sentences:
        return text
    summary = " ".join(sentences[:2])
    if len(sentences) > 1 and summary[-1] not in ".!?":
        summary += "."
    return summary


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    text = text.strip()
    if not text or n_questions <= 0:
        return []

    if OPENAI_API_KEY:
        try:
            from openai import OpenAI

            response = OpenAI(
                api_key=OPENAI_API_KEY, base_url="https://api.groq.com/openai/v1"
            ).chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi mà đoạn văn "
                            "có thể trả lời. Trả về mỗi câu hỏi trên một dòng."
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=max(80, n_questions * 50),
            )
            content = response.choices[0].message.content or ""
            questions = []
            for line in content.splitlines():
                question = line.strip().lstrip("0123456789.-) ")
                if question:
                    questions.append(
                        question if question.endswith("?") else f"{question}?"
                    )
                if len(questions) >= n_questions:
                    break
            if questions:
                return questions
        except Exception as exc:
            print(f"  ⚠️ OpenAI HyQA failed: {exc}")

    import re

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\r?\n+", text)
        if len(sentence.strip()) > 10
    ]
    return [f"{sentence.rstrip('.!?')}?" for sentence in sentences[:n_questions]]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI

            response = OpenAI(
                api_key=OPENAI_API_KEY, base_url="https://api.groq.com/openai/v1"
            ).chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Viết một câu ngắn mô tả vị trí và chủ đề của đoạn văn "
                            "trong tài liệu. Chỉ trả về một câu."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}",
                    },
                ],
                max_tokens=80,
            )
            context = response.choices[0].message.content
            if context := context.strip() if context else "":
                return f"{context}\n\n{text}"
        except Exception as exc:
            print(f"  ⚠️ OpenAI contextual failed: {exc}")

    prefix = f"Trích từ {document_title}. " if document_title else ""
    return f"{prefix}{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    import re

    text = text.strip()
    default = {
        "topic": "general",
        "entities": [],
        "category": "policy",
        "language": "vi" if re.search(r"[\u00c0-\u1ef9]", text) else "en",
    }
    if not text:
        return default

    if OPENAI_API_KEY:
        try:
            import json
            from openai import OpenAI

            response = OpenAI(
                api_key=OPENAI_API_KEY, base_url="https://api.groq.com/openai/v1"
            ).chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Trích xuất metadata và chỉ trả về JSON hợp lệ với các khóa "
                            "topic, entities, category, language. category phải là một trong "
                            "policy, hr, it, finance; language là vi hoặc en."
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
                response_format={"type": "json_object"},
            )
            metadata = json.loads(response.choices[0].message.content or "{}")
            if isinstance(metadata, dict):
                return {
                    "topic": str(metadata.get("topic") or default["topic"]),
                    "entities": (
                        metadata.get("entities", [])
                        if isinstance(metadata.get("entities", []), list)
                        else []
                    ),
                    "category": (
                        metadata.get("category")
                        if metadata.get("category") in {"policy", "hr", "it", "finance"}
                        else default["category"]
                    ),
                    "language": (
                        metadata.get("language")
                        if metadata.get("language") in {"vi", "en"}
                        else default["language"]
                    ),
                }
        except Exception as exc:
            print(f"  ⚠️ OpenAI metadata failed: {exc}")

    return default


# ─── Combined Single-Call Mode ───────────────────────────


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    import json
    import re

    text = (text or "").strip()
    default_meta = {
        "topic": "general",
        "entities": [],
        "category": "policy",
        "language": "vi" if re.search(r"[\u00c0-\u1ef9]", text) else "en",
    }

    def fallback() -> dict:
        sentences = [
            part.strip()
            for part in re.split(r"(?<=[.!?])\s+|\r?\n+", text)
            if part.strip()
        ]
        return {
            "summary": " ".join(sentences[:2]) or text,
            "questions": [
                f"{sentence.rstrip('.!?')}?"
                for sentence in sentences[:3]
                if len(sentence) > 10
            ],
            "context": f"Trích từ {source}." if source else "Nội dung của đoạn văn.",
            "metadata": default_meta,
        }

    if not OPENAI_API_KEY or not text:
        return fallback()

    try:
        from openai import OpenAI

        response = OpenAI(
            api_key=OPENAI_API_KEY, base_url="https://api.groq.com/openai/v1"
        ).chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Phân tích đoạn văn và chỉ trả về JSON hợp lệ với các khóa "
                        "summary (2-3 câu), questions (tối đa 3 câu hỏi), context "
                        "(một câu), metadata (topic, entities, category, language). "
                        "category chỉ được policy, hr, it, finance; language chỉ vi hoặc en."
                    ),
                },
                {"role": "user", "content": f"Tài liệu: {source}\n\nĐoạn văn:\n{text}"},
            ],
            max_tokens=400,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content or "{}")
        if not isinstance(result, dict):
            return fallback()
        metadata = result.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        normalized_meta = {
            "topic": str(metadata.get("topic") or default_meta["topic"]),
            "entities": (
                metadata.get("entities", [])
                if isinstance(metadata.get("entities", []), list)
                else []
            ),
            "category": (
                metadata.get("category")
                if metadata.get("category") in {"policy", "hr", "it", "finance"}
                else default_meta["category"]
            ),
            "language": (
                metadata.get("language")
                if metadata.get("language") in {"vi", "en"}
                else default_meta["language"]
            ),
        }
        questions = result.get("questions", [])
        questions = questions if isinstance(questions, list) else []
        return {
            "summary": str(result.get("summary") or "").strip(),
            "questions": [str(q).strip() for q in questions if str(q).strip()][:3],
            "context": str(result.get("context") or "").strip(),
            "metadata": normalized_meta,
        }
    except Exception as exc:
        print(f"  ⚠️ Combined enrichment failed: {exc}")
        return fallback()


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = (
                contextual_prepend(text, source) if "contextual" in methods else text
            )
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(
            EnrichedChunk(
                original_text=text,
                enriched_text=enriched_text,
                summary=summary,
                hypothesis_questions=questions,
                auto_metadata={**chunk.get("metadata", {}), **auto_meta},
                method="+".join(methods),
            )
        )

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")

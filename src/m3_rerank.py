from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            cache_folder = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "my_models",
            )
            self._model = CrossEncoder(self.model_name, cache_folder=cache_folder)
        return self._model

    def rerank(
        self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K
    ) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if not documents:
            return []

        model = self._load_model()
        pairs = [(query, doc["text"]) for doc in documents]
        scores = model.predict(pairs)
        if isinstance(scores, (int, float)):
            scores = [scores]

        scored = sorted(
            zip(scores, documents), key=lambda item: item[0], reverse=True
        )
        return [
            RerankResult(
                text=doc["text"],
                original_score=doc.get("score", 0.0),
                rerank_score=float(score),
                metadata=doc.get("metadata", {}),
                rank=rank,
            )
            for rank, (score, doc) in enumerate(scored[:top_k])
        ]


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""

    def __init__(self):
        self._model = None

    def rerank(
        self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K
    ) -> list[RerankResult]:
        if not documents or top_k <= 0:
            return []

        if self._model is None:
            from flashrank import Ranker

            self._model = Ranker()

        from flashrank import RerankRequest

        passages = [
            {"id": index, "text": document.get("text", "")}
            for index, document in enumerate(documents)
        ]
        results = self._model.rerank(
            RerankRequest(query=query, passages=passages)
        )

        reranked: list[RerankResult] = []
        for rank, result in enumerate(results[:top_k]):
            if isinstance(result, dict):
                index = result.get("id", rank)
                text = result.get("text", "")
                score = result.get("score", result.get("rerank_score", 0.0))
            else:
                index = getattr(result, "id", rank)
                text = getattr(result, "text", "")
                score = getattr(result, "score", getattr(result, "rerank_score", 0.0))
            document = documents[int(index)] if 0 <= int(index) < len(documents) else {}
            reranked.append(
                RerankResult(
                    text=text or document.get("text", ""),
                    original_score=float(document.get("score", 0.0)),
                    rerank_score=float(score),
                    metadata=document.get("metadata", {}),
                    rank=rank,
                )
            )
        return reranked


def benchmark_reranker(
    reranker, query: str, documents: list[dict], n_runs: int = 5
) -> dict:
    """Benchmark latency over n_runs. (Đã implement sẵn)"""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {
        "avg_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")

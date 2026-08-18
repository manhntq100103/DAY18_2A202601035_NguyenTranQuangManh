from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import heapq
import os, sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    QDRANT_HOST,
    QDRANT_PORT,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    BM25_TOP_K,
    DENSE_TOP_K,
    HYBRID_TOP_K,
)

embedding_model_cache_folder = "./my_models/"


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


_WORD_TOKENIZE = None
_TOKENIZER_UNAVAILABLE = False


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    if not text:
        return text

    global _WORD_TOKENIZE, _TOKENIZER_UNAVAILABLE
    if _WORD_TOKENIZE is None and not _TOKENIZER_UNAVAILABLE:
        try:
            from underthesea import word_tokenize
        except ImportError:
            _TOKENIZER_UNAVAILABLE = True
        else:
            _WORD_TOKENIZE = word_tokenize

    if _WORD_TOKENIZE is None:
        return text.replace("_", " ")
    return _WORD_TOKENIZE(text, format="text").replace("_", " ")


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        from rank_bm25 import BM25Okapi

        self.documents = chunks
        self.corpus_tokens = [
            segment_vietnamese(chunk["text"]).split() for chunk in chunks
        ]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None or top_k <= 0:
            return []

        tokenized_query = segment_vietnamese(query).split()
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
        ranked = heapq.nlargest(
            min(top_k, len(scores)),
            ((float(score), index) for index, score in enumerate(scores) if score > 0),
            key=lambda item: item[0],
        )
        return [
            SearchResult(
                text=self.documents[index]["text"],
                score=score,
                metadata=self.documents[index].get("metadata", {}),
                method="bm25",
            )
            for score, index in ranked
        ]


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient

        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(
                EMBEDDING_MODEL, cache_folder=embedding_model_cache_folder
            )
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        from qdrant_client.models import Distance, PointStruct, VectorParams

        self.client.recreate_collection(
            collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        if not chunks:
            return

        texts = [chunk["text"] for chunk in chunks]
        vectors = self._get_encoder().encode(texts, show_progress_bar=True)
        points = [
            PointStruct(
                id=index,
                vector=vector.tolist(),
                payload={**chunk.get("metadata", {}), "text": chunk["text"]},
            )
            for index, (chunk, vector) in enumerate(zip(chunks, vectors))
        ]
        self.client.upsert(collection, points)

    def search(
        self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME
    ) -> list[SearchResult]:
        """Search using dense vectors."""
        if top_k <= 0:
            return []

        query_vector = self._get_encoder().encode(query).tolist()
        response = self.client.query_points(collection, query=query_vector, limit=top_k)
        return [
            SearchResult(
                text=point.payload["text"],
                score=float(point.score),
                metadata=point.payload,
                method="dense",
            )
            for point in response.points
        ]


def reciprocal_rank_fusion(
    results_list: list[list[SearchResult]], k: int = 60, top_k: int = HYBRID_TOP_K
) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    if top_k <= 0:
        return []

    scores: dict[str, tuple[float, SearchResult]] = {}
    for result_list in results_list:
        for rank, result in enumerate(result_list, start=1):
            previous_score, first_result = scores.get(result.text, (0.0, result))
            scores[result.text] = (previous_score + 1.0 / (k + rank), first_result)

    ranked = heapq.nlargest(top_k, scores.values(), key=lambda item: item[0])
    return [
        SearchResult(
            text=result.text,
            score=rrf_score,
            metadata=result.metadata,
            method="hybrid",
        )
        for rrf_score, result in ranked
    ]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""

    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")

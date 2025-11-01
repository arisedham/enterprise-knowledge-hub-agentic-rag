"""
utils/reranker.py
-----------------
Re-ranks retrieved documents based on semantic relevance to the query.
Uses cross-encoder models from HuggingFace or basic cosine similarity fallback.
"""

import numpy as np
from typing import List
from langchain.schema import Document

try:
    from sentence_transformers import CrossEncoder
    _has_cross_encoder = True
except ImportError:
    _has_cross_encoder = False


class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize reranker with a cross-encoder model.
        If unavailable, falls back to cosine similarity reranking.
        """
        self.model_name = model_name
        if _has_cross_encoder:
            self.model = CrossEncoder(model_name)
        else:
            self.model = None

    def _cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def rerank(self, query: str, retrieved_docs: List[Document], query_embedding=None, doc_embeddings=None):
        """
        Re-rank documents using semantic relevance.
        - If CrossEncoder is available: use deep relevance scoring.
        - Else: use cosine similarity (fallback).
        """
        if not retrieved_docs:
            return []

        if self.model is not None:
            # Use CrossEncoder scoring
            pairs = [[query, d.page_content] for d in retrieved_docs]
            scores = self.model.predict(pairs)
        elif query_embedding is not None and doc_embeddings is not None:
            scores = [self._cosine_similarity(query_embedding, e) for e in doc_embeddings]
        else:
            # Fallback equal score
            scores = [1.0] * len(retrieved_docs)

        ranked = sorted(zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True)
        reranked_docs = [doc for doc, _ in ranked]
        return reranked_docs

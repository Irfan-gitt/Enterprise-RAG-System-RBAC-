"""Hybrid retrieval: vector search + BM25, merged with Reciprocal Rank Fusion.

Pipeline:
    1. vector_search()  -> top-k from Chroma (semantic)
    2. bm25_search()    -> top-k from BM25 over the same documents (lexical)
    3. reciprocal_rank_fusion() -> merges the two ranked lists into one
    4. (optional) rerank() -> Jina's rerank API re-scores the merged candidates

Steps 1-3 need only `rank_bm25`. Step 4 is optional and calls Jina's
rerank API (same provider/API key you already use for embeddings) — no
extra local model, no extra dependency.

Requires: pip install rank_bm25 requests
Needs JINA_API_KEY set in your environment / .env for rerank().
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import requests
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import JinaEmbeddings
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

load_dotenv()
EMBEDDING = JinaEmbeddings(model_name="jina-embeddings-v3")
DB_DIR = Path("chroma_db")
JINA_API_KEY = os.environ.get("JINA_API_KEY")
JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"

DEPARTMENT = "financial"                  # <- change this to switch department
QUERY = "renewal pricing policy"   # <- change this to test a different query

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _store(department: str) -> Chroma:
    return Chroma(
        persist_directory=str(DB_DIR / department),
        embedding_function=EMBEDDING,
        collection_name=f"dept_{department}",
    )


# --------------------------------------------------------------------------
# Two independent retrievers, each returning a plain ranked list of Documents
# --------------------------------------------------------------------------

def vector_search(query: str, department: str, k: int = 10) -> list[Document]:
    """Semantic search, ranked best-first by relevance score."""
    store = _store(department)
    results = store.similarity_search_with_relevance_scores(query, k=k)
    results.sort(key=lambda pair: pair[1], reverse=True)
    return [doc for doc, _score in results]


def bm25_search(query: str, department: str, k: int = 10) -> list[Document]:
    """Lexical search, ranked best-first by BM25 score."""
    store = _store(department)
    data = store.get(include=["documents", "metadatas"])
    documents = list(zip(data["documents"], data["metadatas"]))
    if not documents:
        return []

    corpus_tokens = [tokenize(content) for content, _ in documents]
    bm25 = BM25Okapi(corpus_tokens)
    scores = bm25.get_scores(tokenize(query))

    ranked = sorted(zip(scores, documents),
                    key=lambda pair: pair[0], reverse=True)
    return [
        Document(page_content=content, metadata=metadata)
        for score, (content, metadata) in ranked[:k]
        if score > 0
    ]


# --------------------------------------------------------------------------
# Merge: Reciprocal Rank Fusion
# --------------------------------------------------------------------------

def _doc_key(doc: Document) -> str:
    """Identity for dedup/scoring: filename + section beats hashing full text."""
    return f"{doc.metadata.get('filename', '')}|{doc.metadata.get('section', '')}|{doc.page_content[:80]}"


def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]],
    k: int = 60,
) -> list[Document]:
    """Merge several ranked lists into one, best-first.

    RRF score for a doc = sum over lists of 1 / (k + rank_in_that_list).
    A doc that ranks well in *either* list scores well; a doc that ranks
    well in *both* scores best. k=60 is the standard default from the
    original RRF paper — it just softens how much rank 1 dominates rank 2.
    """
    scores: dict[str, float] = {}
    doc_by_key: dict[str, Document] = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list):
            key = _doc_key(doc)
            doc_by_key[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)

    ordered_keys = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [doc_by_key[key] for key in ordered_keys]


# --------------------------------------------------------------------------
# Optional: cross-encoder rerank of the merged candidates
# --------------------------------------------------------------------------

def rerank(query: str, documents: list[Document], top_k: int = 5) -> list[Document]:
    """Re-score merged candidates jointly with the query via Jina's rerank
    API. Falls back to returning documents unchanged (just truncated) if
    the API key is missing or the call fails, so this step is optional
    and never crashes the pipeline.
    """
    if not JINA_API_KEY or not documents:
        return documents[:top_k]

    try:
        response = requests.post(
            JINA_RERANK_URL,
            headers={
                "Authorization": f"Bearer {JINA_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "jina-reranker-v2-base-multilingual",
                "query": query,
                "documents": [doc.page_content for doc in documents],
                "top_n": top_k,
            },
            timeout=10,
        )
        response.raise_for_status()
        results = response.json()["results"]
        return [documents[r["index"]] for r in results]
    except Exception:
        # Network error, bad response shape, rate limit, etc. — skip
        # reranking rather than break the whole search.
        return documents[:top_k]


# --------------------------------------------------------------------------
# Full pipeline
# --------------------------------------------------------------------------

def hybrid_search(query: str, department: str, k: int = 5, use_reranker: bool = False) -> list[Document]:
    """Vector + BM25 -> RRF merge -> (optional) rerank -> top-k."""
    vector_results = vector_search(query, department, k=10)
    bm25_results = bm25_search(query, department, k=10)
    merged = reciprocal_rank_fusion([vector_results, bm25_results])

    if use_reranker:
        return rerank(query, merged, top_k=k)
    return merged[:k]


if __name__ == "__main__":
    results = hybrid_search(QUERY, DEPARTMENT, k=5, use_reranker=False)
    if not results:
        print("No results.")
    for i, doc in enumerate(results, 1):
        filename = doc.metadata.get("filename", "unknown")
        print(f"\n[{i}] {filename}")
        print(doc.page_content[:300])

from __future__ import annotations

import re

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import JinaEmbeddings
from pathlib import Path
from rank_bm25 import BM25Okapi

load_dotenv()
EMBEDDING = JinaEmbeddings(model_name="jina-embeddings-v3")
DB_DIR = Path("chroma_db")

DEPARTMENT = "engineering"                  # <- change this to switch department
# <- change this to test a different query
QUERY = "Details about Company Overview"
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def bm25_search(query: str, department: str, k: int = 5) -> list[dict]:
    """Rank this department's Chroma documents against `query` with BM25."""
    store = Chroma(
        persist_directory=str(DB_DIR / department),
        embedding_function=EMBEDDING,
        collection_name=f"dept_{department}",
    )
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
        {"score": float(score), "content": content, "metadata": metadata}
        for score, (content, metadata) in ranked[:k]
        if score > 0
    ]


if __name__ == "__main__":
    results = bm25_search(QUERY, DEPARTMENT)
    if not results:
        print("No results.")
    for r in results:
        filename = r["metadata"].get("filename", "unknown")
        print(f"\n[score {r['score']:.2f}] {filename}")
        print(r["content"])

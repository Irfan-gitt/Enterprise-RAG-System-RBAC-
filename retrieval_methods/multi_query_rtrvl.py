"""Query expansion for hybrid search: multi-query rewriting + HyDE.

Users often ask indirectly ("why hasn't my travel money come back") while the
document uses other words ("Reimbursement Processing SLA"). Two LLM steps
bridge that gap:

1. Multi-query: rewrite the question in the document's own vocabulary.
2. HyDE: write a short hypothetical answer and search with that text.

Every phrasing goes through vector search and BM25 (HyDE text is vector-only).
All ranked lists are merged with Reciprocal Rank Fusion.

Needs GROQ_API_KEY in .env. Quick manual test, from the project root:
    python -m retrieval_methods.multi_query_rtrvl "your question" -d engineering
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_groq import ChatGroq

from retrieval_methods.hybrid_retrieval import (
    bm25_search,
    reciprocal_rank_fusion,
    rerank,
    vector_search,
)

load_dotenv()
log = logging.getLogger(__name__)

MODEL = "openai/gpt-oss-20b"
MAX_TOKENS = 2000    # gpt-oss "thinks" first and thinking counts toward this limit
CANDIDATES = 10      # results fetched per individual search, before merging

VARIANTS_PROMPT = """A user asked a question about their company's internal documents.
The question may be indirect: it may describe a situation instead of naming
the policy, process or system it is about.

First work out which topic, policy or process a company document would use to
answer it. Then write {n} different search phrasings that such a document
would use, with formal, domain-specific words. Keep the same intent. Do not
answer the question.

User question: "{query}"

Reply with ONLY a JSON array of {n} strings. Example: ["...", "...", "..."]"""

HYDE_PROMPT = """Write a short (2-4 sentence) hypothetical excerpt from a formal company
document (engineering handbook, HR policy or finance report) that would answer
this question. Formal tone, specific-sounding details, no preamble, no
commentary. Just the excerpt.

Question: "{query}\""""


@lru_cache(maxsize=1)
def _llm() -> ChatGroq:
    """Created on first use, so importing this module never needs the API key."""
    return ChatGroq(model=MODEL, temperature=0, max_tokens=MAX_TOKENS)


def _ask(prompt: str, task: str) -> str | None:
    """Call the LLM. Returns None (and logs why) instead of raising."""
    try:
        text = (_llm().invoke(prompt).content or "").strip()
    except Exception as exc:
        log.warning("%s: LLM call failed: %s", task, exc)
        return None
    if not text:
        log.warning(
            "%s: LLM returned empty text (is MAX_TOKENS too low?)", task)
        return None
    return text


def generate_query_variants(query: str, n: int = 3) -> list[str]:
    """Return up to n rewrites of the query, or [] if the LLM step fails."""
    text = _ask(VARIANTS_PROMPT.format(n=n, query=query), "query rewrite")
    if not text:
        return []
    match = re.search(r"\[.*\]", text, re.S)
    try:
        variants = json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        variants = None
    if not (isinstance(variants, list) and all(isinstance(v, str) for v in variants)):
        log.warning("query rewrite: could not parse reply: %r", text[:150])
        return []
    return variants[:n]


def generate_hyde_document(query: str) -> str | None:
    """Return a short hypothetical answer passage, or None if the LLM step fails."""
    return _ask(HYDE_PROMPT.format(query=query), "HyDE")


def expanded_hybrid_search(
    query: str,
    department: str,
    k: int = 5,
    n_variants: int = 3,
    use_hyde: bool = True,
    use_reranker: bool = False,
) -> list[Document]:
    """Multi-query + HyDE + BM25, merged with RRF. Always returns a list.

    If an LLM step fails it is skipped (and logged), so the search still
    works with whatever phrasings are left, down to the plain query.
    """
    variants = generate_query_variants(query, n_variants)
    hyde = generate_hyde_document(query) if use_hyde else None
    log.info("rewrites=%s | hyde=%r", variants, (hyde or "")[:120])

    # dedupe, keep order
    text_queries = list(dict.fromkeys([query, *variants]))
    vector_queries = text_queries + ([hyde] if hyde else [])

    ranked_lists = [vector_search(q, department, k=CANDIDATES)
                    for q in vector_queries]
    ranked_lists += [bm25_search(q, department, k=CANDIDATES)
                     for q in text_queries]

    merged = reciprocal_rank_fusion(ranked_lists)
    return rerank(query, merged, top_k=k) if use_reranker else merged[:k]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("-d", "--department", default="engineering")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    for rank, doc in enumerate(expanded_hybrid_search(args.query, args.department, k=args.k), 1):
        print(
            f"\n[{rank}] {doc.metadata.get('filename', 'unknown')} | {doc.metadata.get('section', '')}")
        print(doc.page_content[:300])

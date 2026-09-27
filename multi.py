"""Query expansion for vector search: multi-query rewriting + HyDE.

Problem this solves: a user asking "why hasn't my travel money come
back" shares almost no vocabulary with a doc titled "Reimbursement
Processing SLA" — plain vector_search(query) embeds the user's literal
wording, and if that wording is far enough from the doc's phrasing, the
embedding similarity is too weak to retrieve it.

Two techniques, both LLM-based, used together here:

1. Multi-query rewriting — ask an LLM for a few alternate phrasings of
   the same question (different vocabulary, same intent), embed and
   search with EACH one, then merge results. Increases the chance that
   at least one phrasing lands close to the doc's actual wording.

2. HyDE (Hypothetical Document Embeddings) — ask an LLM to write a short
   HYPOTHETICAL ANSWER, as if it were a snippet from the actual policy
   doc, then embed THAT instead of the question. Answers tend to be
   semantically closer to other answers/docs than questions are to
   answers, so this often out-performs even the best query rewrite.

Both are merged into the existing hybrid (vector + BM25) pipeline via
the same reciprocal_rank_fusion() used in hybrid_search.py.

Requires: pip install langchain-groq
Needs GROQ_API_KEY set in your environment / .env — get a free key at
https://console.groq.com/keys (no cost, generous free-tier rate limits).
"""

from __future__ import annotations

import json

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.documents import Document

from hybse import bm25_search, reciprocal_rank_fusion, rerank, vector_search

load_dotenv()

# llama-3.1-8b-instant was deprecated by Groq (Aug 2026); gpt-oss-20b is
# their lightweight replacement — plenty for a rewriting task like this,
# no need for the heavier 120b variant.
LLM = ChatGroq(model="openai/gpt-oss-20b", temperature=0.3, max_tokens=400)

DEPARTMENT = "hr"                  # <- change this to switch department
QUERY = "why hasn't my travel money come back yet"   # <- change this to test


# --------------------------------------------------------------------------
# 1. Multi-query rewriting
# --------------------------------------------------------------------------

def generate_query_variants(query: str, n: int = 3) -> list[str]:
    """Ask the LLM for n alternate phrasings of the same question, using
    more formal/policy-style vocabulary. Falls back to just [query] if
    the LLM call or parsing fails, so this step degrades safely.
    """
    prompt = f"""Rewrite the following user question as {n} alternate
phrasings that a formal company policy document would use, while
preserving the exact same intent. Use more formal, domain-specific
vocabulary (e.g. "reimbursement", "processing time", "SLA") instead of
casual phrasing where appropriate.

User question: "{query}"

Respond with ONLY a JSON array of {n} strings, nothing else. Example:
["...", "...", "..."]"""

    try:
        response = LLM.invoke(prompt)
        text = response.content.strip()
        # Strip accidental markdown fences.
        if text.startswith("```"):
            text = text.strip("`").removeprefix("json").strip()
        variants = json.loads(text)
        if isinstance(variants, list) and all(isinstance(v, str) for v in variants):
            return variants[:n]
    except Exception:
        pass
    return [query]


# --------------------------------------------------------------------------
# 2. HyDE — hypothetical document embeddings
# --------------------------------------------------------------------------

def generate_hyde_document(query: str) -> str | None:
    """Ask the LLM to write a short hypothetical passage — as if it were
    an excerpt from the actual policy/report doc — that would answer this
    query. Returns None if generation fails, so this step is skippable.
    """
    prompt = f"""Write a short (2-4 sentence) hypothetical excerpt from a
formal company policy or financial report document that would answer
this question. Write it as if it were pulled directly from that
document — formal tone, specific-sounding details, no meta-commentary,
no "here is" preamble. Just the excerpt itself.

Question: "{query}\""""

    try:
        response = LLM.invoke(prompt)
        text = response.content.strip()
        return text if text else None
    except Exception:
        return None


# --------------------------------------------------------------------------
# Full pipeline: expand, search every variant, merge everything
# --------------------------------------------------------------------------

def expanded_hybrid_search(
    query: str,
    department: str,
    k: int = 5,
    n_variants: int = 3,
    use_hyde: bool = True,
    use_reranker: bool = False,
) -> list[Document]:
    """Multi-query + HyDE + BM25, all merged via RRF, optionally reranked."""
    variants = generate_query_variants(query, n=n_variants)
    search_queries = [query] + variants

    if use_hyde:
        hyde_doc = generate_hyde_document(query)
        if hyde_doc:
            search_queries.append(hyde_doc)

    # One vector_search per phrasing (dedup identical strings to save calls).
    ranked_lists = [
        vector_search(q, department, k=10) for q in dict.fromkeys(search_queries)
    ]
    # BM25 stays on the original query only — expansion is a vector-search
    # fix for vocabulary mismatch; BM25 already handles exact terms fine.
    ranked_lists.append(bm25_search(query, department, k=10))

    merged = reciprocal_rank_fusion(ranked_lists)

    if use_reranker:
        return rerank(query, merged, top_k=k)
    return merged[:k]


if __name__ == "__main__":
    print(f"Query: {QUERY}\n")

    variants = generate_query_variants(QUERY)
    print("Rewritten variants:")
    for v in variants:
        print(f"  - {v}")

    hyde = generate_hyde_document(QUERY)
    print(f"\nHyDE hypothetical doc:\n  {hyde}\n")

    results = expanded_hybrid_search(QUERY, DEPARTMENT, k=5)
    print("Results:")
    if not results:
        print("No results.")
    for i, doc in enumerate(results, 1):
        filename = doc.metadata.get("filename", "unknown")
        print(f"\n[{i}] {filename}")
        print(doc.page_content[:300])

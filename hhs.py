"""Evaluate retrieval against the employee handbook (general department).

This is the actual test case multi-query/HyDE is FOR: casual question
phrasing ("why hasn't my travel money come back yet") vs. formal policy
document wording ("Reimbursement processed with the next payroll
cycle"). Relevance is checked by content substring, not filename — same
lesson as eval_hr_csv.py.

Edit DEPARTMENT/filename assumptions below if your handbook lives under
a different department or filename than "general".
"""

from __future__ import annotations

from hybse import vector_search, bm25_search, reciprocal_rank_fusion, rerank
from multi import expanded_hybrid_search
from langchain_core.documents import Document

DEPARTMENT = "general"
K = 5

# Each entry: a casually-phrased query, and a substring unique to the
# section of the handbook that actually answers it.
GOLDEN_SET = [
    {
        "query": "why hasn't my travel money come back yet",
        "must_contain": "Reimbursement processed with the next payroll cycle",
    },
    {
        "query": "why is my leave request getting rejected",
        "must_contain": "Approval from reporting manager and HR required",
    },
    {
        "query": "how far in advance do I need to apply for maternity leave",
        "must_contain": "at least 8 weeks before due date",
    },
    {
        "query": "can I get reimbursed for my hotel when traveling for work",
        "must_contain": "Hotel stay up to",
    },
    {
        "query": "what happens if I keep showing up late to work",
        "must_contain": "Three or more late arrivals in a month",
    },
]


def precision_at_k(results: list[Document], must_contain: str, k: int) -> float:
    top_k = results[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for doc in top_k if must_contain in doc.page_content)
    return hits / len(top_k)


def recall_at_k(results: list[Document], must_contain: str, k: int) -> float:
    top_k = results[:k]
    return 1.0 if any(must_contain in doc.page_content for doc in top_k) else 0.0


def reciprocal_rank(results: list[Document], must_contain: str) -> float:
    for rank, doc in enumerate(results, start=1):
        if must_contain in doc.page_content:
            return 1.0 / rank
    return 0.0


def run_method(name: str, get_results) -> dict:
    precisions, recalls, rrs = [], [], []
    for item in GOLDEN_SET:
        results = get_results(item["query"])
        target = item["must_contain"]
        precisions.append(precision_at_k(results, target, K))
        recalls.append(recall_at_k(results, target, K))
        rrs.append(reciprocal_rank(results, target))

    return {
        "method": name,
        f"Precision@{K}": sum(precisions) / len(precisions),
        f"Recall@{K}": sum(recalls) / len(recalls),
        "MRR": sum(rrs) / len(rrs),
    }


def vector_only(query: str) -> list[Document]:
    return vector_search(query, DEPARTMENT, k=K)


def bm25_only(query: str) -> list[Document]:
    return bm25_search(query, DEPARTMENT, k=K)


def hybrid_no_rerank(query: str) -> list[Document]:
    v = vector_search(query, DEPARTMENT, k=10)
    b = bm25_search(query, DEPARTMENT, k=10)
    return reciprocal_rank_fusion([v, b])[:K]


def expanded(query: str) -> list[Document]:
    return expanded_hybrid_search(query, DEPARTMENT, k=K)


if __name__ == "__main__":
    rows = [
        run_method("vector only", vector_only),
        run_method("bm25 only", bm25_only),
        run_method("hybrid (no rerank)", hybrid_no_rerank),
        run_method("expanded hybrid", expanded),
    ]

    header = f"{'method':<22}{'Precision@' + str(K):<15}{'Recall@' + str(K):<15}{'MRR':<10}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['method']:<22}"
            f"{row[f'Precision@{K}']:<15.2f}"
            f"{row[f'Recall@{K}']:<15.2f}"
            f"{row['MRR']:<10.2f}"
        )

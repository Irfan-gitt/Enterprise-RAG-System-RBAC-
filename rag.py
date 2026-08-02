"""Role-scoped retrieval tools used by the agent.

Department selection is code-controlled, not an LLM tool argument.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import re

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import JinaEmbeddings
from langchain_core.documents import Document
from langchain_core.tools import tool

from rbac import ROLE_PERMISSIONS

load_dotenv()
EMBEDDING = JinaEmbeddings(model_name="jina-embeddings-v3")
DB_DIR = Path("chroma_db")
EMPLOYEE_ID_PATTERN = re.compile(r"\b(?:[A-Za-z]+)?EMP[-_]?\d+\b", re.IGNORECASE)


def _employee_id_candidates(text: str) -> set[str]:
    """Return normalized IDs, accepting both EMP1002 and FINEMP1002."""
    return {
        re.sub(r"[^A-Z0-9]", "", match.group(0).upper())
        for match in EMPLOYEE_ID_PATTERN.finditer(text)
    }


DEPARTMENT_HINTS = {
    "financial": {
        "finance", "financial", "revenue", "profit", "income", "budget",
        "expense", "expenses", "invoice", "vendor", "cash flow", "q1", "q2", "q3", "q4",
    },
    "hr": {
        "employee", "employees", "payroll", "salary", "leave", "hiring",
        "attendance", "performance review", "employee id",
    },
    "engineering": {
        "engineering", "architecture", "technical", "system design", "api",
        "infrastructure", "deployment", "database",
    },
    "marketing": {
        "marketing", "campaign", "brand", "advertising", "social media", "seo",
    },
}
LIST_STOP_WORDS = {
    "get", "give", "show", "list", "all", "the", "a", "an", "me", "of",
    "details", "detail", "information", "records", "record", "please", "with",
}
LIST_FIELD_ALIASES = {
    "dob": "date_of_birth",
    "birth": "date_of_birth",
    "joining": "date_of_joining",
    "joined": "date_of_joining",
    "salary": "salary",
}


def _search_departments(query: str, allowed_departments: tuple[str, ...]) -> tuple[str, ...]:
    """Prioritize the authorized department most clearly named by the query."""
    query_lower = query.lower()
    for department, hints in DEPARTMENT_HINTS.items():
        if department in allowed_departments and any(hint in query_lower for hint in hints):
            return (department,)
    return allowed_departments


def _stores(departments: Iterable[str]) -> list[Chroma]:
    return [
        Chroma(persist_directory=str(path), embedding_function=EMBEDDING,
               collection_name=f"dept_{department}")
        for department in departments
        if (path := DB_DIR / department).exists()
    ]


def _format(documents: list[Document]) -> str:
    if not documents:
        return "No relevant company documents were found."
    return "\n\n".join(
        f"[Source: {doc.metadata.get('filename', 'unknown')}"
        f" | Section: {doc.metadata.get('section', 'not specified')}]\n"
        f"{doc.page_content}"
        for doc in documents
    )


def create_retrieval_tools(role: str):
    """Return tools whose search scope is permanently bound to *role*."""
    allowed_departments = tuple(ROLE_PERMISSIONS.get(role, {"general"}))

    def search(query: str, k: int) -> list[Document]:
        ranked: list[tuple[Document, float]] = []
        for store in _stores(_search_departments(query, allowed_departments)):
            ranked.extend(store.similarity_search_with_relevance_scores(query, k=k))
        ranked.sort(key=lambda result: result[1], reverse=True)
        return [document for document, _score in ranked[:k]]

    @tool
    def specific_search(query: str) -> str:
        """Search permitted company documents for focused facts, policies, people, or numbers."""
        return _format(search(query, k=5))

    @tool
    def summarize_search(query: str) -> str:
        """Retrieve permitted document sections needed to summarize a report or topic."""
        return _format(search(query, k=12))

    @tool
    def lookup_search(query: str) -> str:
        """Find permitted company records using an exact employee ID, email, phone, or code."""
        requested_ids = _employee_id_candidates(query)
        query_lower = query.lower().strip()
        exact_matches: list[Document] = []
        for store in _stores(allowed_departments):
            data = store.get(include=["documents", "metadatas"])
            for content, metadata in zip(data["documents"], data["metadatas"]):
                record_ids = _employee_id_candidates(content)
                id_match = any(
                    requested == record or record.endswith(requested)
                    for requested in requested_ids
                    for record in record_ids
                )
                # For an employee-ID question, never fall back to a loose
                # document match: it could return the wrong employee record.
                text_match = not requested_ids and query_lower in content.lower()
                if id_match or text_match:
                    exact_matches.append(Document(page_content=content, metadata=metadata))
        if exact_matches:
            return _format(exact_matches[:5])
        if requested_ids:
            return "No relevant company documents were found."
        return _format(search(query, k=5))

    @tool
    def list_search(query: str, page: int = 1, page_size: int = 10) -> str:
        """List every permitted record matching a role, department, or exact attribute.

        Use for requests containing 'all', such as 'all Relationship Managers'.
        Results are paginated so large employee lists are never silently truncated.
        """
        terms = [
            LIST_FIELD_ALIASES.get(term, term.rstrip("s"))
            for term in re.findall(r"[a-z0-9]+", query.lower())
            if len(term) > 2 and term not in LIST_STOP_WORDS
        ]
        if not terms:
            return "No relevant company documents were found."

        matches: list[Document] = []
        for store in _stores(_search_departments(query, allowed_departments)):
            data = store.get(include=["documents", "metadatas"])
            for content, metadata in zip(data["documents"], data["metadatas"]):
                normalized_content = content.lower()
                if all(term in normalized_content for term in terms):
                    matches.append(Document(page_content=content, metadata=metadata))

        matches.sort(key=lambda doc: doc.page_content)
        if not matches:
            return "No relevant company documents were found."

        safe_page_size = max(1, min(page_size, 25))
        total_pages = (len(matches) + safe_page_size - 1) // safe_page_size
        safe_page = max(1, min(page, total_pages))
        start = (safe_page - 1) * safe_page_size
        page_matches = matches[start:start + safe_page_size]
        header = (
            f"[List results: {len(matches)} total matches; page {safe_page} of "
            f"{total_pages}; showing {len(page_matches)} records.]\n\n"
        )
        return header + _format(page_matches)

    return [specific_search, summarize_search, lookup_search, list_search]

import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from typesafe_sdk import Choice, TypeSafeClient

from prompts import SYSTEM_PROMPT
from rbac import ROLE_PERMISSIONS
from retrieval_methods.hybrid_retrieval import hybrid_search
from retrieval_methods.multi_query_rtrvl import expanded_hybrid_search
from retrieval_methods.specific_search_BM25 import bm25_search

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
jev = TypeSafeClient(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api",
)

DEFAULT_METHOD = "topic"
MIN_CONFIDENCE = 0.6
TOP_K = 8


def classify(question: str) -> str:
    try:
        response = jev.system_one(
            model="typesafe/jev-1.13",
            state={"question": question},
            questions={
                "method": Choice(
                    instructions="Which retrieval method fits this question best?",
                    criteria={
                        "exact": "Exact lookup by employee ID, email, phone, or a person's name",
                        "topic": "Direct question about a policy, report, fact, or number",
                        "indirect": "Indirect or situational question where the wording won't match the documents, e.g. 'why was my leave rejected'",
                    },
                )
            },
        )
        answer = response.answers["method"]
        return answer.choice if answer.confidence >= MIN_CONFIDENCE else DEFAULT_METHOD
    except Exception:
        return DEFAULT_METHOD


def normalize(result) -> dict:
    if isinstance(result, dict):
        return result
    return {
        "content": result.page_content,
        "metadata": result.metadata,
        "score": result.metadata.get("score", 0),
    }


def retrieve(question: str, role: str) -> list[dict]:
    method = classify(question)
    results = []
    for department in ROLE_PERMISSIONS.get(role, {"general"}):
        if method == "exact":
            results += bm25_search(question, department, k=TOP_K)
        elif method == "indirect":
            results += expanded_hybrid_search(question, department, k=TOP_K)
        else:
            results += hybrid_search(question, department, k=TOP_K)
    print(f"[method: {method}]")
    results = [normalize(r) for r in results]
    return sorted(results, key=lambda r: r.get("score", 0), reverse=True)[:TOP_K]


def answer(question: str, role: str) -> str:
    results = retrieve(question, role)
    if not results:
        return "I don't have that information in the documents available to me."
    context = "\n\n".join(
        f"[Source: {r['metadata'].get('filename', 'unknown')}]\n{r['content']}"
        for r in results
    )
    system = SYSTEM_PROMPT.format(role=role) + "\n\nCONTEXT:\n" + context
    return llm.invoke([("system", system), ("human", question)]).content


if __name__ == "__main__":
    role = input("role: ").strip() or "employee"
    while True:
        q = input(f"[{role}] > ").strip()
        if q in ("exit", "quit"):
            break
        print(answer(q, role))

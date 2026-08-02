from typing import TypedDict
import re

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from prompts import AGENT_PROMPT, SYSTEM_PROMPT
from rag import create_retrieval_tools

# Retrieval supplies the evidence, so a fast model keeps chat latency low.
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)


class AppContext(TypedDict):
    role: str


def create_crag_agent(role: str):
    """Optional tool-calling agent, retained for experimentation only."""
    from guardrails import guardrail_middleware

    return create_agent(
        model=llm,
        tools=create_retrieval_tools(role),
        system_prompt=AGENT_PROMPT.format(role=role),
        middleware=guardrail_middleware,
        context_schema=AppContext,
    )


OFF_TOPIC_KEYWORDS = {
    "weather", "sports", "cricket", "football", "movie", "movies", "joke",
    "recipe", "celebrity", "bitcoin price", "stock price",
}


def answer_question(question: str, role: str, page: int = 1, page_size: int = 10) -> str:
    """Production RAG path: retrieve authorized context before generation.

    The old agent path made an LLM decide whether to call a search tool. This
    function always retrieves first, and its tools are bound to ``role``.
    """
    normalized = question.lower()
    if any(keyword in normalized for keyword in OFF_TOPIC_KEYWORDS):
        return (
            "I'm AtliQ's internal assistant. I can help with company policies, "
            "finance, HR, engineering, and marketing information."
        )

    tools = {tool.name: tool for tool in create_retrieval_tools(role)}
    is_identifier_lookup = bool(
        re.search(r"\b(?:[a-z]+)?emp[-_]?\d+\b", normalized)
        or "employee id" in normalized
        or "email" in normalized
        or "phone" in normalized
    )
    is_summary_request = any(
        marker in normalized for marker in ("summarize", "summary", "overview")
    )
    is_list_request = "all" in normalized.split()
    # Focused retrieval gives the answer model compact, relevant evidence. Use
    # broad retrieval only when the user explicitly asks for a summary.
    tool_name = (
        "lookup_search" if is_identifier_lookup
        else "list_search" if is_list_request
        else "summarize_search" if is_summary_request
        else "specific_search"
    )
    tool_input = {"query": question}
    if tool_name == "list_search":
        tool_input.update({"page": page, "page_size": page_size})
    context = tools[tool_name].invoke(tool_input)

    if context == "No relevant company documents were found.":
        return "I don't have that information in the documents available to me."

    # Lists are structured data, not a generative task. Sending ten (or more)
    # records through an LLM can silently omit rows, so return the authorized,
    # paginated records directly and preserve completeness.
    if tool_name == "list_search":
        return context

    system_message = SYSTEM_PROMPT.format(role=role) + "\n\nCONTEXT:\n" + context
    try:
        return llm.invoke([
            SystemMessage(content=system_message),
            HumanMessage(content=question),
        ]).content
    except Exception:
        # Do not discard retrieved, authorized evidence when the generator has
        # a temporary API failure.
        return context


if __name__ == "__main__":
    role = "hr"
    print(answer_question(" details of relationship managers", role))

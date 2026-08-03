from langsmith import traceable
from typing import Annotated, TypedDict
import re

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from prompts import AGENT_PROMPT, SYSTEM_PROMPT
from rag import create_retrieval_tools
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


memory = InMemorySaver()


class AppContext(TypedDict):
    role: str


def create_crag_agent(role: str):
    """Optional tool-calling agent, retained for experimentation only."""
    from guardrails import guardrail_middleware

    return create_agent(
        model=llm,
        tools=create_retrieval_tools(role),
        system_prompt=AGENT_PROMPT.format(role=role),
        middleware=[guardrail_middleware],
        context_schema=AppContext,
        checkpointer=memory)


OFF_TOPIC_KEYWORDS = {
    "weather", "sports", "cricket", "football", "movie", "movies", "joke",
    "recipe", "celebrity", "bitcoin price", "stock price",
}


class ChatState(TypedDict):
    """State for the deterministic RAG graph below. ``messages`` is the
    piece that gets checkpointed per thread_id — that's what actually gives
    ``answer_question`` memory."""
    messages: Annotated[list[BaseMessage], add_messages]
    role: str
    page: int
    page_size: int


def _generate(state: ChatState) -> dict:
    """Same routing + retrieval logic answer_question always had, just
    reading its inputs from graph state instead of function arguments."""
    question = state["messages"][-1].content
    role = state["role"]
    page = state["page"]
    page_size = state["page_size"]

    normalized = question.lower()
    if any(keyword in normalized for keyword in OFF_TOPIC_KEYWORDS):
        return {"messages": [AIMessage(content=(
            "I'm AtliQ's internal assistant. I can help with company policies, "
            "finance, HR, engineering, and marketing information."
        ))]}

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
        return {"messages": [AIMessage(
            content="I don't have that information in the documents available to me."
        )]}

    # Lists are structured data, not a generative task. Sending ten (or more)
    # records through an LLM can silently omit rows, so return the authorized,
    # paginated records directly and preserve completeness.
    if tool_name == "list_search":
        return {"messages": [AIMessage(content=context)]}

    system_message = SYSTEM_PROMPT.format(
        role=role) + "\n\nCONTEXT:\n" + context
    # Everything before the question we just received is prior conversation —
    # this is the line that actually makes memory do something, separate from
    # wiring the checkpointer in at all.
    history = state["messages"][:-1]
    try:
        response = llm.invoke([
            SystemMessage(content=system_message),
            *history,
            HumanMessage(content=question),
        ])
        return {"messages": [AIMessage(content=response.content)]}
    except Exception:
        # Do not discard retrieved, authorized evidence when the generator has
        # a temporary API failure.
        return {"messages": [AIMessage(content=context)]}


_graph_builder = StateGraph(ChatState)
_graph_builder.add_node("generate", _generate)
_graph_builder.add_edge(START, "generate")
_graph_builder.add_edge("generate", END)
graph = _graph_builder.compile(checkpointer=memory)


@traceable(
    name="CragBot Query",
    tags=["enterprise-rag"],
)
def answer_question(
    question: str,
    role: str,
    thread_id: str = "default",
    page: int = 1,
    page_size: int = 10,
) -> str:
    """Production RAG path: retrieve authorized context before generation.

    The old agent path made an LLM decide whether to call a search tool. This
    function always retrieves first, and its tools are bound to ``role``.
    ``thread_id`` scopes conversation memory — same thread_id continues the
    same conversation, a new one starts blank.
    """
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(
        {
            "messages": [HumanMessage(content=question)],
            "role": role,
            "page": page,
            "page_size": page_size,
        },
        config=config,
    )
    return result["messages"][-1].content


if __name__ == "__main__":
    role = "hr"
    print(answer_question(" details of relationship managers", role))

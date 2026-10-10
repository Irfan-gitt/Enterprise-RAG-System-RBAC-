from llm_pool import LLMPool
import logging
import os
import re
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from rag import agentic_rag_agent
from typesafe_sdk import Choice, TypeSafeClient


logger = logging.getLogger(__name__)


REFUSAL = (
    "I can only help with company-related questions for your role. "
    "Please ask something related to your work."
)
ERROR_REPLY = "Sorry, something went wrong while processing your request. Please try again."

COMPANY_DESCRIPTION = (
    "An internal assistant for our company's employees. It answers questions "
    "about HR policies, benefits, leave, departments, employees and their roles, "
    "projects, onboarding, and internal processes, based on company documents."
)

BANNED_KEYWORDS = [
    "hack", "exploit", "malware", "jailbreak", "prompt injection",
    "bypass", "circumvent", "cheat", "crack", "pirate", "torrent",
]
_KEYWORD_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(k.lower())
                        for k in BANNED_KEYWORDS) + r")(?:s|es|ing|ed|er)?\b"
)

HISTORY_MESSAGES = 4  # how many earlier messages the guard / rewrite steps can see

# Jev client (guard)
client = TypeSafeClient(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api",
)


rewrite_llm = LLMPool(
    "openai/gpt-oss-20b",
    temperature=0,
    max_tokens=1024,
    timeout=10,
    reasoning_effort="low",
)

REWRITE_PROMPT = """Rewrite the user's latest question as one standalone question.
Use the conversation ONLY to resolve references such as "he", "she", "that policy", "his manager".

Rules:
- Do not answer the question.
- Do not add any information that is not in the conversation.
- If the question is already standalone, return it unchanged.
- Output only the rewritten question, nothing else.

Conversation:
{history}

Latest question: {question}

Standalone question:"""


def build_history(messages: list[BaseMessage], max_chars: int = 300) -> str:
    """Last few human/ai messages as plain text, each trimmed to save tokens."""
    past = [
        f"{m.type}: {m.content[:max_chars]}"
        for m in messages
        if m.type in ("human", "ai") and isinstance(m.content, str) and m.content
    ]
    return "\n".join(past[-HISTORY_MESSAGES:])


def topic_check(question: str, history: str = "") -> str:
    """Returns 'company', 'off_topic' or 'unclear'."""
    response = client.system_one(
        model="typesafe/jev-1.13",
        state={"question": question,
               "recent_conversation": history or "(none)"},
        questions={
            "topic": Choice(
                instructions=(
                    f"You guard an internal company assistant. {COMPANY_DESCRIPTION} "
                    "Decide whether the question is about the company. "
                    "Any person's name or ID may be a colleague, so a question about "
                    "a named person's role, team, manager, email or record is company. "
                    "Use recent_conversation to resolve follow-ups like "
                    "'who is his manager?'. "
                    "If you are unsure, choose unclear."
                ),
                criteria={
                    "company": (
                        "About the company, its employees, teams, roles, projects, "
                        "policies, benefits, leave, reimbursement, finance, security, "
                        "engineering processes or the user's own work. Includes any "
                        "question about a named person or employee ID. "
                        "Examples: 'what is the leave policy', 'who is Vihaan Desai', "
                        "'who is FN300PN', 'what is FINEMP1042's salary', "
                        "'why was my leave rejected', "
                        "'what are the RTO and RPO for disaster recovery'."
                    ),
                    "off_topic": (
                        "General knowledge, news, famous people, sports, entertainment, "
                        "markets, coding help unrelated to company documents, math, "
                        "personal advice, jokes or casual chit-chat. "
                        "Examples: 'who is Cristiano Ronaldo', 'write a python function "
                        "to sort a list', 'tell me a joke', 'capital of France', "
                        "'how are you'."
                    ),
                    "unclear": (
                        "You cannot tell whether it relates to the company. "
                        "Prefer this over a wrong off_topic."
                    ),
                },
            )
        },
    )
    return response.answers["topic"].choice


def check_query(query: str, history: str = "") -> tuple[bool, str]:
    """Runs keyword filter then Jev topic check. Returns (allowed, reason)."""
    match = _KEYWORD_PATTERN.search(query.lower())
    if match:
        return False, f"keyword:{match.group(0)}"

    try:
        label = topic_check(query, history)
    except Exception as e:
        logger.warning("Jev guard failed, letting request through: %s", e)
        return True, "guard_error_fail_open"  # never block a real question on an error

    if label == "off_topic":
        return False, "jev:off_topic"
    return True, f"jev:{label}"


# Graph

class GraphState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]

    role: str
    blocked: bool
    standalone_question: str


def guard_node(state: GraphState) -> dict:
    messages = state["messages"]
    query = messages[-1].content
    allowed, reason = check_query(query, build_history(messages[:-1]))

    if not allowed:
        logger.info("BLOCKED (%s): %s", reason, query[:60])
        return {"blocked": True, "messages": [AIMessage(REFUSAL)]}

    logger.info("ALLOWED (%s): %s", reason, query[:60])

    return {"blocked": False}


def route_after_guard(state: GraphState) -> str:
    return END if state["blocked"] else "rewrite"


def rewrite_node(state: GraphState) -> dict:
    messages = state["messages"]
    question = messages[-1].content
    history = build_history(messages[:-1])

    if not history:  # first message of the session: nothing to resolve
        return {"standalone_question": question}

    try:
        reply = rewrite_llm.invoke(REWRITE_PROMPT.format(
            history=history, question=question))
        rewritten = reply.content.strip() if isinstance(reply.content, str) else ""
    except Exception as e:
        logger.warning("Rewrite failed, using original question: %s", e)
        rewritten = ""

    return {"standalone_question": rewritten or question}


def rag_node(state: GraphState) -> dict:
    try:
        answer = agentic_rag_agent(state["standalone_question"], state["role"])
    except Exception:
        logger.exception("agentic_rag_agent failed")
        answer = ERROR_REPLY
    return {"messages": [AIMessage(answer)]}


def build_app():
    graph = StateGraph(GraphState)
    graph.add_node("guard", guard_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("rag", rag_node)

    graph.add_edge(START, "guard")
    graph.add_conditional_edges("guard", route_after_guard, {
                                "rewrite": "rewrite", END: END})
    graph.add_edge("rewrite", "rag")
    graph.add_edge("rag", END)

    return graph.compile(checkpointer=InMemorySaver())


app = build_app()


def ask(query: str, role: str, session_id: str) -> str:
    """
    query      : the user's message
    role       : resolved from the user's JWT / email by main.py (never from the model)
    session_id : unique per user session; it is the memory key (LangGraph thread_id)
    """
    query = query.strip()
    if not query:
        return "Please type a question."

    result = app.invoke(
        {"messages": [HumanMessage(query)], "role": role},
        config={"configurable": {"thread_id": session_id}},
    )
    return result["messages"][-1].content


# THESE Are tst codes
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    TEST_ROLE = "general"

    print("--- session A ---")
    for q in [
        "who is cristiano ronaldo",   # expect: blocked, no retrieval
        "who is FINEMP1078",          # expect: answer
        "who is her manager?",        # expect: follow-up resolved via memory
        "what is our leave policy?",  # expect: answer
    ]:
        print(f"\n=== {q}")
        print(ask(q, TEST_ROLE, session_id="session-A"))

    print("\n--- session B (separate memory) ---")
    print(ask("who is her manager?", TEST_ROLE, session_id="session-B"))

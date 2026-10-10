import os
import re
from typing import Any

from dotenv import load_dotenv
from groq import BadRequestError
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.runtime import Runtime

# !! Copy this exact import line from the file where retrieval_category works.
# I don't know the module path of your TypeSafe SDK, so I can't write it for you.
from typesafe_sdk import Choice, TypeSafeClient

load_dotenv()  # loads GROQ_API_KEY and OPENROUTER_API_KEY from .env

# Main agent model (unchanged)
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

# Guard model client (Jev), same setup as your retrieval_category
client = TypeSafeClient(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api",
)

REFUSAL = (
    "I can only help with company-related questions for your role. "
    "Please ask something related to your work."
)

# EDIT THIS: the more specific it is, the more accurate the guard becomes
COMPANY_DESCRIPTION = (
    "An internal assistant for our company's employees. It answers questions "
    "about HR policies, benefits, leave, departments, employees and their roles, "
    "projects, onboarding, and internal processes, based on company documents."
)


# ---------------- Layer 1: keyword filter (unchanged) ----------------
class ContentFilterMiddleware(AgentMiddleware):
    """Deterministic guardrail: blocks banned requests before any LLM call."""

    def __init__(self, banned_keywords: list[str]):
        super().__init__()
        alts = "|".join(re.escape(k.lower()) for k in banned_keywords)
        self.pattern = re.compile(rf"\b(?:{alts})(?:s|es|ing|ed|er)?\b")

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        last_human = next(
            (m for m in reversed(state["messages"]) if m.type == "human"), None
        )
        if last_human is None or not isinstance(last_human.content, str):
            return None

        match = self.pattern.search(last_human.content.lower())
        if match:
            print(f"Blocked -- keyword detected: '{match.group(0)}'")
            return {"messages": [AIMessage(REFUSAL)], "jump_to": "end"}
        return None


# ---------------- Layer 2: Jev topic guard (replaces the Groq guard) ----------------
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


class LLMTopicGuardMiddleware(AgentMiddleware):
    """Blocks general-knowledge / coding / chit-chat. Fails open on any error."""

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        msgs = state["messages"]
        last_human = next(
            (m for m in reversed(msgs) if m.type == "human"), None)
        if last_human is None or not isinstance(last_human.content, str):
            return None

        # last 4 earlier messages, each trimmed to save tokens
        past = [
            f"{m.type}: {m.content[:300]}"
            for m in msgs[:-1]
            if m.type in ("human", "ai") and isinstance(m.content, str) and m.content
        ]
        history = "\n".join(past[-4:])

        try:
            label = topic_check(last_human.content, history)
        except Exception as e:
            print(f"Jev guard failed, letting request through: {e}")
            return None  # fail open: never block a real question because of an error

        print(f"Guard label: {label}")
        if label == "off_topic":
            print(
                f"Blocked -- Jev guard: off_topic -> {last_human.content[:60]}")
            return {"messages": [AIMessage(REFUSAL)], "jump_to": "end"}
        return None


@tool
def search_tool(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"


filtered_agent = create_agent(
    model=llm,
    tools=[search_tool],
    system_prompt=(
        "You are an internal company assistant. The ONLY tool you have is "
        "`search_tool`. Never call any other tool (such as open_file, browser, "
        "or python). Answer using search_tool results only. If the results "
        "don't contain the answer, say you couldn't find it."
    ),
    middleware=[
        ContentFilterMiddleware(
            banned_keywords=[
                
                "hack", "exploit", "malware", "jailbreak","prompt injection","bypass", "circumvent", "cheat", "crack", "pirate", "torrent",
            ]
        ),
        LLMTopicGuardMiddleware(),  # runs only if the keyword filter didn't block
    ],
)


def safe_invoke(agent, payload, retries=2):
    for attempt in range(retries + 1):
        try:
            return agent.invoke(payload)
        except BadRequestError as e:
            if "tool_use_failed" in str(e) and attempt < retries:
                print(f"Tool call failed, retrying ({attempt + 1})")
                continue
            raise


if __name__ == "__main__":
    tests = [
        "who is cristiano ronaldo",         # expect: blocked by Jev guard
        "write a python function to sort",  # expect: blocked by Jev guard
        "who is FN300PN",                   # expect: passes (employee ID)
        "what is our leave policy?",        # expect: passes to the agent
    ]
    for q in tests:
        print(f"\n=== {q}")
        result = safe_invoke(
            filtered_agent,
            {"messages": [{"role": "user", "content": q}]},
        )
        print(result["messages"][-1].content)

from groq import BadRequestError
import re
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.runtime import Runtime

load_dotenv()  # loads GROQ_API_KEY from .env

# Main agent model
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

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


# ---------------- Layer 1: keyword filter ----------------
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


# ---------------- Layer 2: LLM topic guard ----------------
GUARD_PROMPT = """You are a topic classifier for an internal company assistant.

Company context: {company}

Classify the user's LATEST message as exactly one label:
- COMPANY: about the company, its employees, teams, roles, projects, policies,
  benefits, processes, or the user's work. Any person's name may be a colleague,
  so questions about a person's role, team, manager, or work are COMPANY even if
  you don't recognise the name.
- OFF_TOPIC: general knowledge, news, famous people, sports, coding help, math,
  advice unrelated to the company, jokes, or casual chit-chat.
- UNCLEAR: you can't tell. Prefer UNCLEAR over a wrong OFF_TOPIC.

Recent conversation (for follow-ups like "who is his manager?"):
{history}

Latest message: {query}

Answer with ONLY one word: COMPANY, OFF_TOPIC, or UNCLEAR."""

guard_llm = ChatGroq(
    model="openai/gpt-oss-20b",  # small + cheap, separate from the main agent
    temperature=0,
    max_tokens=512,              # room for hidden reasoning tokens + the one-word answer
    timeout=10,
    max_retries=1,
    reasoning_effort="low",      # if your langchain-groq version rejects this, upgrade it
)


def parse_label(text: str) -> str:
    """Pull the first valid label out of the model's reply. Default: UNCLEAR."""
    cleaned = text.upper().replace(" ", "_").replace("-", "_")
    found = re.search(r"OFF_TOPIC|COMPANY|UNCLEAR", cleaned)
    return found.group(0) if found else "UNCLEAR"


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
            reply = guard_llm.invoke(
                GUARD_PROMPT.format(
                    company=COMPANY_DESCRIPTION,
                    history=history or "(none)",
                    query=last_human.content,
                )
            )
            raw = reply.content if isinstance(reply.content, str) else ""
            if not raw.strip():
                print("Guard LLM returned empty output, letting request through")
                return None
            label = parse_label(raw)
        except Exception as e:
            print(f"Guard LLM failed, letting request through: {e}")
            return None  # fail open: never block a real question because of an error

        print(f"Guard label: {label}")
        if label == "OFF_TOPIC":
            print(
                f"Blocked -- LLM guard: OFF_TOPIC -> {last_human.content[:60]}")
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
                "weather", "sports", "cricket", "football", "movie", "joke",
                "recipe", "celebrity", "bitcoin", "stock price",
                "hack", "exploit", "malware", "jailbreak",
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
        "who is FN300PN",         # expect: blocked by LLM guard
        "write a python function to sort",  # expect: blocked by LLM guard
        "what is our leave policy?",        # expect: passes to the agent
    ]
    for q in tests:
        print(f"\n=== {q}")
        result = safe_invoke(
            filtered_agent,
            {"messages": [{"role": "user", "content": q}]},
        )
        print(result["messages"][-1].content)

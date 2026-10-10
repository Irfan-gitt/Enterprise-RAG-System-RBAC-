"""Evaluate the RAG pipeline (retrieval + LLM answer).

    python eval_rag.py

RETRIEVAL metrics (is the search good?)
    precision@k : of the chunks we got back, how many are actually useful
    recall@k    : of the things we NEEDED, how many did we get back
    mrr         : how high the first useful chunk is ranked (1.0 = top)

ANSWER metrics (is the LLM good?)  -> judged by an LLM
    faithfulness: every claim in the answer is backed by the retrieved chunks
    correctness : answer matches the ground-truth answer
    relevance   : answer actually addresses the question

If the judge fails or the answer has nothing to check, the score is n/a
(NOT 0), and n/a is left out of the averages.
"""

from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq

# ---------------- CHANGE THESE ----------------
# Your main file name WITHOUT .py (the one that has retriver and rag_agent)
from rag import rag_agent, retriver

ROLE = "engineering"
DATASET = Path("eval_dataset.json")
OUT_CSV = Path("eval_results.csv")
# seconds between questions (Groq free tier = 8000 tokens/min)
SLEEP = 15
MAX_CONTEXT_CHARS = 6000  # smaller = fewer tokens = fewer 429 errors
# ----------------------------------------------

load_dotenv()
judge = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
judge_failures = 0


# ---------- helpers ----------
def with_retry(fn, *args, tries: int = 5):
    """Call fn(*args); if Groq says 429 (rate limit), wait and try again."""
    wait = 5
    for attempt in range(tries):
        try:
            return fn(*args)
        except Exception as exc:
            is_limit = "429" in str(exc) or "rate limit" in str(exc).lower()
            if not is_limit or attempt == tries - 1:
                raise
            print(f"   rate limit, waiting {wait}s ...")
            time.sleep(wait)
            wait *= 2


def to_chunks(results) -> list[dict]:
    """Turn whatever the retriever returns into [{text, meta}, ...]."""
    if isinstance(results, str):                       # retriever returned one big string
        return [{"text": results, "meta": {}}] if results.strip() else []
    if not isinstance(results, list):
        return []
    chunks = []
    for r in results:
        if isinstance(r, dict):
            text = r.get("content") or r.get("page_content") or ""
            meta = r.get("metadata") or {}
        elif isinstance(r, tuple):                      # (Document, score)
            text, meta = r[0].page_content, r[0].metadata
        elif isinstance(r, str):
            text, meta = r, {}
        else:                                           # Document
            text = getattr(r, "page_content", str(r))
            meta = getattr(r, "metadata", {}) or {}
        chunks.append({"text": text, "meta": meta})
    return chunks


def chunk_matches(chunk: dict, gold: str) -> bool:
    haystack = (chunk["text"] + " " +
                str(chunk["meta"].get("section", ""))).lower()
    return gold.lower() in haystack


def ask_judge(prompt: str) -> dict | None:
    """Call the judge LLM and parse its JSON. Returns None if it fails."""
    global judge_failures
    raw = ""
    for _ in range(2):
        raw = with_retry(judge.invoke, prompt).content
        match = re.search(r"\{.*\}", raw, re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    judge_failures += 1
    print(f"   !! judge gave no valid JSON. Raw reply: {raw[:200]!r}")
    return None


def score_from(data: dict | None) -> tuple[float | None, str]:
    if not data or "score" not in data:
        return None, ""
    try:
        return float(data["score"]), str(data.get("reason", ""))
    except (TypeError, ValueError):
        return None, ""


# ---------- retrieval metrics ----------
def retrieval_scores(chunks: list[dict], gold_list: list[str]) -> dict:
    if not gold_list:                       # question has no answer in docs
        return {"precision": None, "recall": None, "mrr": None}
    if not chunks:
        return {"precision": 0.0, "recall": 0.0, "mrr": 0.0}

    relevant_flags = [any(chunk_matches(c, g)
                          for g in gold_list) for c in chunks]
    precision = sum(relevant_flags) / len(chunks)

    found = sum(any(chunk_matches(c, g) for c in chunks) for g in gold_list)
    recall = found / len(gold_list)

    mrr = 0.0
    for rank, flag in enumerate(relevant_flags, start=1):
        if flag:
            mrr = 1 / rank
            break
    return {"precision": precision, "recall": recall, "mrr": mrr}


# ---------- answer metrics (LLM as judge) ----------
def faithfulness(answer: str, context: str) -> float | None:
    prompt = f"""You are a strict fact checker.
Split the ANSWER into small factual claims. For each claim, say if the CONTEXT
supports it. Use ONLY the context, not your own knowledge.

CONTEXT:
{context}

ANSWER:
{answer}

Reply with JSON only, like:
{{"claims": [{{"claim": "...", "supported": true}}]}}
If the answer has no factual claims (for example "I don't have that info"), return {{"claims": []}}."""
    data = ask_judge(prompt)
    claims = (data or {}).get("claims", [])
    if not claims:
        return None                          # nothing to check -> n/a, not a fake 1.0
    return sum(1 for c in claims if c.get("supported")) / len(claims)


def correctness(question: str, answer: str, truth: str) -> tuple[float | None, str]:
    prompt = f"""Compare the ANSWER with the GROUND TRUTH for the QUESTION.

QUESTION: {question}
GROUND TRUTH: {truth}
ANSWER: {answer}

Scoring:
1.0 = all key facts match, nothing wrong
0.5 = partly correct, or missing some key facts
0.0 = wrong, or contradicts the truth
Special case: if GROUND TRUTH is "NOT_IN_DOCS", the correct answer is one that
says the information is not available (give 1.0). Making up a value gives 0.0.

Reply with JSON only: {{"score": 0.0, "reason": "short reason"}}"""
    return score_from(ask_judge(prompt))


def relevance(question: str, answer: str) -> tuple[float | None, str]:
    prompt = f"""Does the ANSWER directly address the QUESTION?

QUESTION: {question}
ANSWER: {answer}

Scoring:
1.0 = fully on topic and answers what was asked
0.5 = partly answers, or has a lot of unneeded extra info
0.0 = off topic or does not answer

Reply with JSON only: {{"score": 0.0, "reason": "short reason"}}"""
    return score_from(ask_judge(prompt))


# ---------- main ----------
def avg(values: list) -> tuple[float | None, int]:
    values = [v for v in values if v is not None]
    return (sum(values) / len(values) if values else None), len(values)


def fmt(v) -> str:
    return "  n/a" if v is None else f"{v:5.2f}"


def main() -> None:
    items = json.loads(DATASET.read_text(encoding="utf-8"))
    rows = []

    for i, item in enumerate(items, start=1):
        question, truth = item["question"], item["ground_truth"]
        print(f"\n[{i}/{len(items)}] {question}")
        row = {"question": question}
        try:
            results = with_retry(retriver, question, ROLE)
            if i == 1:   # DEBUG: show what the retriever really returns
                print(f"   DEBUG retriever returned type={type(results).__name__}, "
                      f"first 300 chars: {str(results)[:300]!r}")
            chunks = to_chunks(results)
            answer = with_retry(rag_agent, question, results)
            context = "\n\n---\n\n".join(c["text"]
                                         for c in chunks)[:MAX_CONTEXT_CHARS]

            row.update(answer=answer, chunks_returned=len(chunks))
            row["retrieved_preview"] = " | ".join(
                (c["meta"].get("section") or c["text"][:60]).replace("\n", " ")
                for c in chunks[:10])
            row.update(retrieval_scores(chunks, item.get("gold", [])))
            row["faithfulness"] = faithfulness(answer, context)
            row["correctness"], row["correctness_reason"] = correctness(
                question, answer, truth)
            row["relevance"], row["relevance_reason"] = relevance(
                question, answer)
        except Exception as exc:                        # one bad question must not kill the run
            print(f"   FAILED: {exc}")
            row["answer"] = f"ERROR: {exc}"

        print(f"   chunks_returned={row.get('chunks_returned', '?')}")
        print("   " + "  ".join(
            f"{k}={fmt(row.get(k))}"
            for k in ("precision", "recall", "mrr", "faithfulness", "correctness", "relevance")))
        rows.append(row)
        time.sleep(SLEEP)

    # ----- summary -----
    print("\n" + "=" * 52)
    print("FINAL RESULTS (average, n/a questions left out)")
    print("=" * 52)
    print("RETRIEVAL")
    for m in ("precision", "recall", "mrr"):
        value, n = avg([r.get(m) for r in rows])
        print(f"  {m:<14}{fmt(value)}   ({n} questions scored)")
    print("ANSWER (LLM)")
    for m in ("faithfulness", "correctness", "relevance"):
        value, n = avg([r.get(m) for r in rows])
        print(f"  {m:<14}{fmt(value)}   ({n} questions scored)")
    print(f"\nJudge failures: {judge_failures}  |  Failed questions: "
          f"{sum(1 for r in rows if str(r.get('answer', '')).startswith('ERROR'))}")

    # ----- save -----
    fields = ["question", "answer", "chunks_returned", "retrieved_preview",
              "precision", "recall", "mrr", "faithfulness",
              "correctness", "correctness_reason", "relevance", "relevance_reason"]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved per-question results to {OUT_CSV}")


if __name__ == "__main__":
    main()

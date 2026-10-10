"""Run the RAG evaluation inside LangSmith.

    python eval_langsmith.py            # run an experiment
    python eval_langsmith.py --resync   # re-upload eval_dataset.json first

What you get in LangSmith (smith.langchain.com):
  * a Dataset with your golden questions
  * an Experiment: one row per question, with precision / recall / mrr /
    faithfulness / correctness / relevance scores
  * click any row -> full trace: retriever -> chunks -> LLM -> judge calls

Needs in your .env:
    LANGSMITH_TRACING=true
    LANGSMITH_API_KEY=lsv2_...
    LANGSMITH_PROJECT=finsolve-rag
    (EU accounts also: LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com)

Install:  pip install -U langsmith
"""

from __future__ import annotations
from eval_rag import (
    MAX_CONTEXT_CHARS,
    ROLE,
    SLEEP,
    correctness,
    faithfulness,
    relevance,
    retrieval_scores,
    to_chunks,
    with_retry,
)
from rag import rag_agent, retriver                      # <- your main file
from langsmith import Client, evaluate, traceable
from pathlib import Path
import time
import os
import json
import argparse

from dotenv import load_dotenv

load_dotenv()   # must run before langchain / langsmith are used


# Reuse the exact same pipeline + metric code as eval_rag.py

# ---------------- CHANGE THESE ----------------
DATASET_FILE = Path("eval_dataset.json")
DATASET_NAME = "finsolve-rag-golden"
EXPERIMENT_PREFIX = "rag-eval"
# Tags saved on the experiment so you can compare runs later in the UI.
# Change these when you change the system (e.g. reranker True -> False).
RUN_METADATA = {
    "retriever": "hybrid (vector + BM25 + RRF)",
    "reranker": False,
    "chunking": "v2-headings",
    "answer_model": "gpt-oss-120b",
}
# ----------------------------------------------


# ---------- 0. check the env ----------
def check_env(client: Client) -> None:
    key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    if not key:
        raise SystemExit(
            "LANGSMITH_API_KEY not found. Add it to your .env file.")
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", "finsolve-rag")
    try:
        # real call = key is valid
        next(iter(client.list_datasets(limit=1)), None)
    except Exception as exc:
        raise SystemExit(f"LangSmith key/endpoint problem: {exc}")
    print(
        f"LangSmith OK  | key {key[:8]}... | project={os.environ['LANGSMITH_PROJECT']}")


# ---------- 1. dataset ----------
def sync_dataset(client: Client, resync: bool) -> None:
    exists = client.has_dataset(dataset_name=DATASET_NAME)
    if exists and not resync:
        print(
            f"Dataset '{DATASET_NAME}' already exists, reusing it (use --resync to re-upload).")
        return
    if exists:
        client.delete_dataset(dataset_name=DATASET_NAME)
        print(f"Deleted old dataset '{DATASET_NAME}'.")

    items = json.loads(DATASET_FILE.read_text(encoding="utf-8"))
    dataset = client.create_dataset(
        DATASET_NAME, description="FinSolve engineering doc golden Q&A for RAG eval")
    client.create_examples(
        dataset_id=dataset.id,
        inputs=[{"question": i["question"]} for i in items],
        outputs=[{"answer": i["ground_truth"]} for i in items],
        metadata=[{"gold": i.get("gold", []), "type": i.get(
            "type", "")} for i in items],
    )
    print(f"Uploaded {len(items)} examples to '{DATASET_NAME}'.")


# ---------- 2. the thing being tested ----------
# Wrapping gives nested spans in the trace: retriever and rag_agent show up as steps.
traced_retriever = traceable(name="retriever", run_type="retriever")(retriver)
traced_agent = traceable(name="rag_agent", run_type="chain")(rag_agent)


def target(inputs: dict) -> dict:
    question = inputs["question"]
    results = with_retry(traced_retriever, question, ROLE)
    chunks = to_chunks(results)
    answer = with_retry(traced_agent, question, results)
    # stay under the Groq tokens-per-minute limit
    time.sleep(SLEEP)
    return {
        "answer": answer,
        "chunks": [{"text": c["text"], "section": c["meta"].get("section", "")} for c in chunks],
    }


# ---------- 3. evaluators ----------
def _chunks(run) -> list[dict]:
    stored = (run.outputs or {}).get("chunks", [])
    return [{"text": c["text"], "meta": {"section": c.get("section", "")}} for c in stored]


def _result(key: str, score, comment: str = "") -> dict:
    out = {"key": key, "score": score}
    if comment or score is None:
        out["comment"] = comment or "n/a"
    return out


def _retrieval_evaluator(metric: str):
    def evaluator(run, example):
        gold = (example.metadata or {}).get("gold", [])
        if not gold:
            return _result(metric, None, "n/a: question has no answer in the docs")
        scores = retrieval_scores(_chunks(run), gold)
        return _result(metric, scores[metric], f"chunks returned: {len(_chunks(run))}")
    evaluator.__name__ = metric
    return evaluator


def faithfulness_evaluator(run, example):
    answer = (run.outputs or {}).get("answer", "")
    context = "\n\n---\n\n".join(c["text"]
                                 for c in _chunks(run))[:MAX_CONTEXT_CHARS]
    if not context:
        return _result("faithfulness", 0.0, "no chunks retrieved, answer has no support")
    score = faithfulness(answer, context)
    return _result("faithfulness", score, "" if score is not None else "no claims to check / judge failed")


def correctness_evaluator(run, example):
    score, reason = correctness(
        example.inputs["question"], (run.outputs or {}).get("answer", ""), example.outputs["answer"])
    return _result("correctness", score, reason)


def relevance_evaluator(run, example):
    score, reason = relevance(
        example.inputs["question"], (run.outputs or {}).get("answer", ""))
    return _result("relevance", score, reason)


# ---------- 4. go ----------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resync", action="store_true",
                        help="re-upload eval_dataset.json")
    args = parser.parse_args()

    client = Client()
    check_env(client)
    sync_dataset(client, args.resync)

    results = evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[
            _retrieval_evaluator("precision"),
            _retrieval_evaluator("recall"),
            _retrieval_evaluator("mrr"),
            faithfulness_evaluator,
            correctness_evaluator,
            relevance_evaluator,
        ],
        experiment_prefix=EXPERIMENT_PREFIX,
        metadata=RUN_METADATA,
        # one question at a time (rate limits)
        max_concurrency=1,
    )
    print("\nDone. Open LangSmith -> Datasets & Experiments -> "
          f"'{DATASET_NAME}' -> experiment '{results.experiment_name}'")


if __name__ == "__main__":
    main()

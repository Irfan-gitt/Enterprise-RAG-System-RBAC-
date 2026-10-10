"""
This is an agentic RAG system that takes a user question and returns a final answer.
User Question
    -> Classify Retrieval Category
    -> Select Retrieval Strategy (BM25 / Hybrid / Expanded Hybrid)
    -> Fuse Results (RRF)
    -> Rerank Results (if multiple departments)
    -> Finalizing Agent (LLM)
    -> Final Answer
"""

from __future__ import annotations


from rbac import ROLE_PERMISSIONS
import os
from typesafe_sdk import Choice, TypeSafeClient

from pathlib import Path
from typing import Iterable
import re

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import JinaEmbeddings
from langchain_core.documents import Document
from langchain_core.tools import tool

from retrieval_methods.hybrid_retrieval import hybrid_search
from retrieval_methods.specific_search_BM25 import bm25_search
from retrieval_methods.multi_query_rtrvl import expanded_hybrid_search
from retrieval_methods.hybrid_retrieval import reciprocal_rank_fusion, rerank
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from llm_pool import LLMPool

EMBEDDING = JinaEmbeddings(model_name="jina-embeddings-v3")
DB_DIR = Path("chroma_db")

OPENROUTER_API_KEY = os.getenv(
    "OPENROUTER_API_KEY")


llm = LLMPool("openai/gpt-oss-20b", temperature=0)


TOP_K = 10

client = TypeSafeClient(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api",
)


def retrieval_category(question):
    response = client.system_one(
        model="typesafe/jev-1.13",
        state={"question": question},
        questions={
            "category": Choice(
                instructions=(
                    "Pick exactly one category. Decide in this order:\n"
                    "1. If the question is about ONE named person's own record "
                    "(employee ID, email, phone, full name) -> lookup_search.\n"
                    "2. Else, if the question already uses the key words a company "
                    "document would use for the topic (a policy name, system name, "
                    "metric or term) and wants one fact from it -> specific_search.\n"
                    "3. Else, if the question describes a situation, problem or goal in "
                    "everyday words without naming the policy or system, or needs "
                    "several sections combined -> summarize_search.\n"
                    "If you are unsure between specific_search and summarize_search, "
                    "choose summarize_search."
                ),
                criteria={
                    "lookup_search": (
                        "The question is about ONE named person's own record or contact "
                        "details, identified by employee ID, email, phone number or full "
                        "name, and asks a fact about that individual (salary, team, "
                        "manager, joining date). It is NOT about a policy, process, "
                        "system or company-wide fact. "
                        "Examples: 'what is FINEMP1042's salary', "
                        "'find isha.chowdhury@fintechco.com', 'who is Vihaan Desai', "
                        "'what is Vihaan Desai's department'. "
                        "NOT this: 'what is the leave policy' (no person), "
                        "'why was my leave rejected' (no named person)."
                    ),
                    "specific_search": (
                        "A DIRECT question that already uses the same key words the "
                        "document would use (policy name, system name, metric, term) and "
                        "wants one fact, number, date, list or definition. The answer "
                        "sits in one section. "
                        "Examples: 'what is the leave policy', 'what was Q1 2024 revenue', "
                        "'when is the reimbursement deadline', 'what was last year's "
                        "company turnover', 'what are the RTO and RPO for disaster "
                        "recovery', 'which databases does FinSolve use', 'what is the "
                        "minimum unit test coverage'. "
                        "NOT this: a question that describes a situation or symptom "
                        "instead of naming the topic."
                    ),
                    "summarize_search": (
                        "An INDIRECT question: it describes a situation, problem, symptom "
                        "or goal in everyday words and does NOT name the policy, process "
                        "or system that answers it, so its words will not match the "
                        "document. Often starts with why / how come / who checks / what "
                        "happens if / how do we stop. Also use it for broad questions "
                        "that need several sections combined (summarize, explain end to "
                        "end, compare). Works for any department. "
                        "Examples: 'why is my leave request getting rejected', 'why "
                        "haven't I got my travel money back yet', 'our cloud bill keeps "
                        "going up, who checks it', 'a test fails randomly then passes, "
                        "what does the pipeline do', 'how do we stop old wrong data from "
                        "being shown after a change', 'a customer is worried their card "
                        "details could be stolen, what protects them', 'summarize our "
                        "security measures'. "
                        "Contrast: 'what is the cache invalidation policy' is "
                        "specific_search, but 'how do we stop old wrong data being shown "
                        "after a change' is summarize_search."
                    ),
                },
            )
        },
    )

    return response.answers["category"].choice


def retriver(question: str, role: str) -> list:

    departments = sorted(ROLE_PERMISSIONS.get(role, {"general"}))
    category = retrieval_category(question)

    if category == "lookup_search":
        results = []
        print("lookup_search")
        for dept in departments:
            results.extend(bm25_search(question, dept))
        return sorted(results, key=lambda r: r["score"], reverse=True)[:TOP_K]

    if category == "summarize_search":
        print("summarize_search")
        def search(dept): return expanded_hybrid_search(
            question, dept, k=TOP_K)
    else:   # specific_search or anything else
        print("specific_search")
        def search(dept): return hybrid_search(question, dept, k=TOP_K)

    candidates = reciprocal_rank_fusion([search(dept) for dept in departments])
    if len(departments) > 1:
        # best chunks across departments
        return rerank(question, candidates, top_k=TOP_K)
    return candidates[:TOP_K]


def rag_agent(question: str, system_answer: str):
    prompt = f"""
    #ROLE
    You are a Finalising agent for a Agentic Rag System 

    #TASK
    Your job is to check is wheather the Rag_output: {system_answer} Fullfill Users Question: {question}

    #CONSTRAIN 
    Dont make up any point just make the answer as user need 

    #EXAMPLE
    Suppose user only need a information about a specific person named 'ayush josh' but the rag give 3-4 names similar too eg: ayush khan, ayush hedje, vishak ayush etc.. , So your job is to give the correct name ,only in this senario so you only give ayush josh's information to user , like  this example act in every where according to the situation as users request

    """
    response = llm.invoke(prompt).content
    return response


def agentic_rag_agent(user_input, role):

    system_answer = retriver(user_input, role)
    final = rag_agent(user_input, system_answer)

    return final


if __name__ == "__main__":
    print(agentic_rag_agent("who is FINEMP1026", "hr"))

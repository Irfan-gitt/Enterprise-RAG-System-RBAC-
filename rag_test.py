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
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

load_dotenv()
EMBEDDING = JinaEmbeddings(model_name="jina-embeddings-v3")
DB_DIR = Path("chroma_db")

OPENROUTER_API_KEY = os.getenv(
    "OPENROUTER_API_KEY")


llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)


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
                instructions="Which retrieval method fits this question best?",
                criteria={
                    "lookup_search": (
                        "The question names a specific person and asks for their own record — "
                        "matched by employee ID, email, phone number, or full name. Not about "
                        "a policy, report, or company-wide fact. Examples: 'what is FINEMP1042's "
                        "salary', 'find isha.chowdhury@fintechco.com', 'who is Vihaan Desai'."
                    ),
                    "specific_search": (
                        "The question can be answered with ONE fact, number, date, or policy "
                        "detail — a single lookup, not a synthesis. No person's identity involved. "
                        "Examples: 'what is the leave policy', 'what was Q1 2024 revenue', "
                        "'when is the reimbursement deadline'.,'how was the previous year company turn over'"
                    ),
                    "summarize_search": (
                        "An indirect or personal question about the user's own situation — often "
                        "'why' something happened, was delayed, or was rejected — where the "
                        "wording won't match how the policy document itself is phrased. "
                        "Examples: 'why is my leave request getting rejected', 'why haven't I "
                        "got my travel money back yet', 'why was my reimbursement denied'."
                    ),
                },
            )
        },
    )

    confidence = response.answers["category"].confidence

    category = response.answers["category"].choice

    return category


def retriver(question: str, role: str):
    category = retrieval_category(question, k=TOP_K)
    for department in ROLE_PERMISSIONS.get(role, department, {"general"}):
        if category == "lookup_search":
            print("lookup_search")
            return bm25_search(question, k=TOP_K)
        elif category == "specific_search":
            print("lookup_search")
            return hybrid_search(question, department, k=TOP_K)
        elif category == "summarize_search":
            return expanded_hybrid_search(question, department, k=TOP_K)
        else:
            return "Failed to find category"


def rag_agent(question, system_answer):
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


def main():
    pass

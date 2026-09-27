from __future__ import annotations
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

from rbac import ROLE_PERMISSIONS

load_dotenv()
EMBEDDING = JinaEmbeddings(model_name="jina-embeddings-v3")
DB_DIR = Path("chroma_db")

OPENROUTER_API_KEY = os.getenv(
    "OPENROUTER_API_KEY")
client = TypeSafeClient(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api",
)


response = client.system_one(
    model="typesafe/jev-1.13",
    state={"question": "find the employee id of the person who joined in 2020 and has a salary greater than 100000"},
    questions={
        "category": Choice(
            instructions="find me the anual salary of the employee who joined in 2020 and has a salary greater than 100000",
            criteria={
                "lookup_search": "Exact employee ID, email, or phone lookup",
                "list_search": "Requests for 'all' of something — every employee, every record of a type",
                "summarize_search": "Summaries or overviews of a report or topic",
                "specific_search": "A specific fact, policy, or number not covered above",
            },
        )
    },
)

category = response.answers["category"].choice

confidence = response.answers["category"].confidence

print(f"Category: {category}, Confidence: {confidence}")

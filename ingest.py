"""Build the Chroma indexes.

Run this file explicitly after changing documents:
    .\\venv\\Scripts\\python.exe ingest.py --reset

It intentionally has no work at import time. Importing a retrieval module must
never create embeddings or append duplicate documents to an index.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import CSVLoader, TextLoader
from langchain_community.embeddings import JinaEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

RESOURCE_DIR = Path("resources")
DB_DIR = Path("chroma_db")
DEPARTMENTS = ("engineering", "financial", "general", "hr", "marketing")


def load_documents() -> list[Document]:
    documents: list[Document] = []
    for department in DEPARTMENTS:
        for path in (RESOURCE_DIR / department).iterdir():
            if path.suffix == ".md":
                loaded = TextLoader(str(path), encoding="utf-8").load()
            elif path.suffix == ".csv":
                loaded = CSVLoader(str(path)).load()
            else:
                continue
            for document in loaded:
                document.metadata.update({"department": department, "filename": path.name})
            documents.extend(loaded)
    return documents


def build_indexes(reset: bool = False) -> None:
    load_dotenv()
    documents = load_documents()
    recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=1_000, chunk_overlap=150)
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False,
    )
    chunks: list[Document] = []
    for document in documents:
        if document.metadata["filename"].endswith(".md"):
            sections = markdown_splitter.split_text(document.page_content)
            for section in sections:
                section.metadata.update(document.metadata)
                # Store the heading separately so it remains available even if
                # a long section is split into multiple smaller chunks.
                section.metadata["section"] = " > ".join(
                    section.metadata[key]
                    for key in ("h1", "h2", "h3")
                    if key in section.metadata
                )
            chunks.extend(recursive_splitter.split_documents(sections))
        else:
            # CSVLoader already emits one document per row, ideal for exact ID lookup.
            chunks.extend(recursive_splitter.split_documents([document]))
    embedding = JinaEmbeddings(model_name="jina-embeddings-v3")
    for department in DEPARTMENTS:
        persist_dir = DB_DIR / department
        if reset and persist_dir.exists():
            shutil.rmtree(persist_dir)
        department_chunks = [chunk for chunk in chunks if chunk.metadata["department"] == department]
        if department_chunks:
            Chroma.from_documents(documents=department_chunks, embedding=embedding,
                                  persist_directory=str(persist_dir),
                                  collection_name=f"dept_{department}")
            print(f"Indexed {len(department_chunks)} chunks for {department}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Rebuild indexes from scratch")
    build_indexes(reset=parser.parse_args().reset)

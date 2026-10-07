"""Build the Chroma indexes.

    .\\venv\\Scripts\\python.exe ingest.py [--reset]

Every chunk is stored as "<filename> > <h1> > <h2> > <h3>\\n<body>" so both the
embedding and BM25 know which section a chunk came from.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator

import tiktoken
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import CSVLoader, TextLoader
from langchain_community.embeddings import JinaEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import (
    Language,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

RESOURCE_DIR = Path("resources")
DB_DIR = Path("chroma_db")
DEPARTMENTS = ("engineering", "financial", "general", "hr", "marketing")

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64          # set to 0 if you also use section_expand.py
MIN_CHUNK_CHARS = 30
BATCH_SIZE = 64
HEADER_RESERVE = 40         # HEADINGS: tokens kept free for the prefix line
HEADER_KEYS = ("h1", "h2", "h3")

log = logging.getLogger("ingest")

LOADERS = {
    ".md": lambda p: TextLoader(str(p), encoding="utf-8"),
    ".csv": lambda p: CSVLoader(str(p), encoding="utf-8"),
}


def make_splitters() -> dict:
    encoder = tiktoken.get_encoding("cl100k_base")

    def token_len(text: str) -> int:
        return len(encoder.encode(text, disallowed_special=()))

    common = dict(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
                  length_function=token_len, add_start_index=True)
    # HEADINGS: markdown bodies are a bit smaller so body + prefix <= CHUNK_SIZE
    md_common = {**common, "chunk_size": CHUNK_SIZE - HEADER_RESERVE}
    return {
        ".md": RecursiveCharacterTextSplitter.from_language(Language.MARKDOWN, **md_common),
        ".csv": RecursiveCharacterTextSplitter(**common),
        # HEADINGS: first pass, split markdown into sections by heading.
        # strip_headers=True because the heading is re-added as the prefix.
        "headers": MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
            strip_headers=True),
    }


def load_documents(department: str) -> list[Document]:
    folder = RESOURCE_DIR / department
    if not folder.is_dir():
        log.warning("Missing folder: %s", folder)
        return []
    documents: list[Document] = []
    for path in sorted(folder.iterdir()):
        loader = LOADERS.get(path.suffix.lower())
        if loader is None:
            continue
        try:
            loaded = loader(path).load()
        except Exception:
            log.exception("Failed to load %s", path)
            continue
        for doc in loaded:
            if doc.page_content.strip():
                doc.metadata.update(department=department, filename=path.name)
                documents.append(doc)
    return documents


def _prefixed(chunks: Iterable[Document], header: str) -> Iterator[Document]:
    """Drop tiny fragments (judged on the body), then prepend the header line."""
    for chunk in chunks:
        body = chunk.page_content.strip()
        if len(body) < MIN_CHUNK_CHARS:
            continue
        chunk.page_content = f"{header}\n{body}"
        yield chunk


def iter_chunks(doc: Document, splitters: dict) -> Iterator[Document]:
    filename = doc.metadata["filename"]
    if Path(filename).suffix.lower() == ".md":
        sections = splitters["headers"].split_text(doc.page_content)
        for sec_id, section in enumerate(sections):
            heading = " > ".join(
                section.metadata[k] for k in HEADER_KEYS if k in section.metadata)
            header = f"{filename} > {heading}" if heading else filename
            section_doc = Document(
                page_content=section.page_content,
                metadata={**doc.metadata, "section": heading,
                          "section_id": f"{filename}#{sec_id}"})
            yield from _prefixed(splitters[".md"].split_documents([section_doc]), header)
    else:
        # CSV: one row = one section; the row's columns are the body.
        row = doc.metadata.get("row", 0)
        doc.metadata["section"] = f"row {row}"
        doc.metadata["section_id"] = f"{filename}#row{row}"
        yield from _prefixed(splitters[".csv"].split_documents([doc]), filename)


def chunk_documents(documents: list[Document], splitters) -> tuple[list[Document], list[str]]:
    chunks: list[Document] = []
    ids: list[str] = []
    counters: dict[str, int] = defaultdict(int)
    seen: set[str] = set()
    for doc in documents:
        for chunk in iter_chunks(doc, splitters):
            text = chunk.page_content
            key = f'{chunk.metadata["department"]}/{chunk.metadata["filename"]}'
            digest = hashlib.sha256(f"{key}\n{text}".encode()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            chunk.metadata["chunk_index"] = counters[key]
            counters[key] += 1
            chunks.append(chunk)
            ids.append(digest[:32])
    return chunks, ids


def sync_department(department: str, embedding, splitters, reset: bool) -> None:
    persist_dir = DB_DIR / department
    if reset and persist_dir.exists():
        shutil.rmtree(persist_dir)

    chunks, ids = chunk_documents(load_documents(department), splitters)
    if not chunks:
        log.warning("No chunks for %s", department)
        return

    store = Chroma(collection_name=f"dept_{department}", embedding_function=embedding,
                   persist_directory=str(persist_dir))

    existing = set(store.get()["ids"])
    stale = existing - set(ids)
    if stale:
        store.delete(ids=list(stale))

    # Only embed chunks that are not already in the index (saves Jina calls).
    new = [(c, i) for c, i in zip(chunks, ids) if i not in existing]
    for start in range(0, len(new), BATCH_SIZE):
        batch = new[start:start + BATCH_SIZE]
        store.add_documents([c for c, _ in batch], ids=[i for _, i in batch])

    log.info("%s: %d chunks total, %d new embedded, %d stale removed",
             department, len(chunks), len(new), len(stale))


def build_indexes(reset: bool = False) -> None:
    load_dotenv()
    embedding = JinaEmbeddings(model_name="jina-embeddings-v3")
    splitters = make_splitters()
    for department in DEPARTMENTS:
        sync_department(department, embedding, splitters, reset)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true",
                        help="Rebuild indexes from scratch")
    build_indexes(reset=parser.parse_args().reset)

"""Prototype API for the Company internal knowledge assistant.

Run locally with:
    .\\venv\\Scripts\\uvicorn.exe main:app --reload
"""

from __future__ import annotations

import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from jose import JWTError, jwt
from pydantic import BaseModel, Field

from agent import answer_question
from rbac import ROLE_PERMISSIONS

load_dotenv()

APP_DIR = Path(__file__).parent
JWT_SECRET_KEY = os.getenv(
    "JWT_SECRET_KEY", "prototype-only-change-this-secret")
JWT_ALGORITHM = "HS256"
TOKEN_EXPIRE_MINUTES = 120

# Prototype-only accounts. Replace this with a database, hashed passwords, or
# an identity provider before production. Roles are never accepted from clients.
DEMO_USERS = {
    "employee@company.com": {"password": "demo123", "role": "employee"},
    "finance@company.com": {"password": "demo123", "role": "finance"},
    "hr@company.com": {"password": "demo123", "role": "hr"},
    "engineering@company.com": {"password": "demo123", "role": "engineering"},
    "marketing@company.com": {"password": "demo123", "role": "marketing"},
    "admin@company.com": {"password": "demo123", "role": "admin"},
}

app = FastAPI(title="Company Internal Knowledge Assistant")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
bearer_scheme = HTTPBearer(auto_error=False)


class LoginRequest(BaseModel):
    email: str
    password: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2_000)
    # Optional client-supplied conversation id. Omit it to start a fresh
    # thread; the server will generate one and hand it back in the response
    # so the frontend can resend it on the next turn to keep the same
    # LangGraph memory thread going.
    conversation_id: str | None = Field(default=None, max_length=64)
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=10, ge=1, le=25)


def create_access_token(email: str, role: str) -> str:
    expires_at = datetime.now(timezone.utc) + \
        timedelta(minutes=TOKEN_EXPIRE_MINUTES)
    return jwt.encode(
        {"sub": email, "role": role, "exp": expires_at},
        JWT_SECRET_KEY,
        algorithm=JWT_ALGORITHM,
    )


def current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
) -> dict[str, str]:
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired session. Please sign in again.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if not credentials or credentials.scheme.lower() != "bearer":
        raise unauthorized
    try:
        payload = jwt.decode(credentials.credentials,
                             JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        email = payload.get("sub")
        role = payload.get("role")
    except JWTError as error:
        raise unauthorized from error
    if not isinstance(email, str) or role not in ROLE_PERMISSIONS:
        raise unauthorized
    return {"email": email, "role": role}


def extract_metadata(answer: str) -> tuple[list[str], dict | None, str]:
    """Adapt the existing RAG string result to a frontend-friendly response."""
    sources = list(dict.fromkeys(re.findall(r"\[Source: ([^|\]]+)", answer)))
    sources.extend(
        source for source in re.findall(r"(?m)^Source:\s*([^\n]+)", answer)
        if source not in sources
    )
    list_match = re.search(
        r"\[List results: (\d+) total matches; page (\d+) of (\d+); showing (\d+) records\.\]",
        answer,
    )
    pagination = None
    if list_match:
        total, page, total_pages, shown = map(int, list_match.groups())
        pagination = {
            "total_matches": total,
            "page": page,
            "page_size": shown,
            "total_pages": total_pages,
        }
    clean_answer = re.sub(r"^\[List results: .*?\]\s*", "", answer)
    return sources, pagination, clean_answer


@app.post("/api/auth/login")
def login(request: LoginRequest) -> dict:
    email = request.email.strip().lower()
    user = DEMO_USERS.get(email)
    if not user or request.password != user["password"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password.")
    role = user["role"]
    return {
        "access_token": create_access_token(email, role),
        "token_type": "bearer",
        "user": {"email": email, "role": role},
    }


@app.post("/api/chat")
def chat(request: ChatRequest, user: Annotated[dict[str, str], Depends(current_user)]) -> dict:
    conversation_id = request.conversation_id or uuid.uuid4().hex
    # Namespace the LangGraph thread with the authenticated email so a
    # guessed or shared conversation_id can never pull up someone else's
    # chat history — the checkpointer only keys on thread_id.
    thread_id = f"{user['email']}:{conversation_id}"
    answer = answer_question(
        request.question.strip(),
        user["role"],
        thread_id=thread_id,
        page=request.page,
        page_size=request.page_size,
    )
    sources, pagination, clean_answer = extract_metadata(answer)
    return {
        "answer": clean_answer,
        "sources": sources,
        "pagination": pagination,
        "conversation_id": conversation_id,
    }


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def frontend() -> FileResponse:
    return FileResponse(APP_DIR / "index.html")


app.mount("/", StaticFiles(directory=APP_DIR, html=True), name="frontend")

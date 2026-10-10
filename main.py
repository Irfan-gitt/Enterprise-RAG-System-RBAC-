"""API for the Company internal knowledge assistant.

Run locally with:
    .\\venv\\Scripts\\uvicorn.exe main:app --reload
"""

import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel, Field

from agent import ask
from rbac import ROLE_PERMISSIONS

load_dotenv()

APP_DIR = Path(__file__).parent
JWT_SECRET_KEY = os.getenv(
    "JWT_SECRET_KEY", "prototype-only-change-this-secret")
JWT_ALGORITHM = "HS256"
TOKEN_EXPIRE_MINUTES = 120

# Prototype-only accounts. Replace with a database / identity provider later.
# Roles are never accepted from clients.
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
    # Optional. Omit to start a new conversation; resend the returned id to continue it.
    conversation_id: str | None = Field(default=None, max_length=64)


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


@app.post("/api/auth/login")
def login(request: LoginRequest) -> dict:
    email = request.email.strip().lower()
    user = DEMO_USERS.get(email)
    if not user or request.password != user["password"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )
    role = user["role"]
    return {
        "access_token": create_access_token(email, role),
        "token_type": "bearer",
        "user": {"email": email, "role": role},
    }


@app.post("/api/chat")
def chat(request: ChatRequest, user: Annotated[dict[str, str], Depends(current_user)]) -> dict:
    conversation_id = request.conversation_id or uuid.uuid4().hex
    # Namespaced with the authenticated email so a guessed conversation_id
    # can never open another user's memory thread.
    session_id = f"{user['email']}:{conversation_id}"
    answer = ask(request.question, user["role"], session_id)
    return {"answer": answer, "conversation_id": conversation_id}


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def login_page() -> FileResponse:
    return FileResponse(APP_DIR / "login.html")


@app.get("/chat")
def chat_page() -> FileResponse:
    return FileResponse(APP_DIR / "chat.html")


@app.get("/style.css")
def style() -> FileResponse:
    return FileResponse(APP_DIR / "style.css")


@app.get("/login.js")
def login_js() -> FileResponse:
    return FileResponse(APP_DIR / "login.js")


@app.get("/chat.js")
def chat_js() -> FileResponse:
    return FileResponse(APP_DIR / "chat.js")

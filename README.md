# Enterprise RAG System with RBAC 🏢🤖

An enterprise-grade internal knowledge assistant built for companies to query their private documents securely. Features Role-Based Access Control, multi-strategy RAG retrieval, guardrails, and JWT authentication.

---

## 🔗 Live Demo
> Deploy link here after Railway deployment

---

## 🧠 What It Does

Companies store sensitive data across departments — HR, Finance, Engineering, Marketing. This system lets employees query their company's internal documents using natural language, while ensuring:

- HR employees can only access HR data
- Finance team can only access financial reports
- Admins have full access
- No sensitive data leaks across departments

---

## ✨ Features

- 🔍 **Multi-Strategy RAG** — Multi-Query + Reranking + BM25 for accurate retrieval
- 🔐 **RBAC** — Role-Based Access Control per department
- 🛡️ **Guardrails** — Off-topic detection + PII protection using LangChain middleware
- 🔑 **JWT Auth** — Secure login with role embedded in token
- ⚡ **Streaming** — Token-by-token streaming responses
- 📊 **LangSmith** — Monitoring and tracing
- 🎯 **Agentic RAG** — LangGraph agent decides which retrieval strategy to use

---

## 🏗️ Architecture

```
User Login (JWT)
      ↓
Role extracted from token
      ↓
Guardrails check (off-topic + RBAC)
      ↓
LangGraph Agent decides retrieval strategy:
  ├── specific_search()   → Multi-Query + Jina Reranking
  ├── summarize_search()  → Broad context retrieval (k=15)
  └── lookup_search()     → BM25 exact keyword search
      ↓
Chroma Vector DB (separate collection per department)
      ↓
Streamed response to frontend
```

---

## 🗂️ Department Access Control

| Role | HR | Finance | Engineering | Marketing | General |
|------|----|---------|-----------  |-----------|---------|
| Admin | ✅ | ✅ | ✅ | ✅ | ✅ |
| HR | ✅ | ❌ | ❌ | ❌ | ✅ |
| Finance | ❌ | ✅ | ❌ | ❌ | ✅ |
| Engineering | ❌ | ❌ | ✅ | ❌ | ✅ |
| Marketing | ❌ | ❌ | ❌ | ✅ | ✅ |
| Employee | ❌ | ❌ | ❌ | ❌ | ✅ |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent Framework | LangChain + LangGraph |
| LLM | Groq (llama-3.3-70b-versatile) |
| Retrieval | Multi-Query + Jina Reranker + BM25 |
| Vector DB | ChromaDB (per-department collections) |
| Embeddings | Jina Embeddings v3 |
| Guardrails | LangChain Middleware + PIIMiddleware |
| Auth | JWT (python-jose + passlib) |
| Backend | FastAPI + Uvicorn |
| Frontend | HTML + CSS + JavaScript |
| Monitoring | LangSmith |
| Deployment | Railway |

---

## 📊 Monitoring & Observability

All queries are automatically traced using **LangSmith**, giving full visibility into every request:

- 🔍 **Query tracking** — every question logged with the user's role
- 📄 **Source tracing** — which documents were retrieved for each answer
- ⏱️ **Response time** — latency tracked per request
- 🔢 **Token usage** — input/output tokens counted automatically
- 🛠️ **Tool traces** — which RAG strategy the agent chose (specific/summarize/lookup)
- ❌ **Error tracking** — failed requests captured with full stack trace

View live traces at [smith.langchain.com](https://smith.langchain.com) under project **CragBot-Enterprise-RAG**.

## 📁 Project Structure

```
Enterprise-RAG-System-RBAC/
├── main.py          # FastAPI app + endpoints
├── agent.py         # LangGraph agent setup
├── rag.py           # 3 retrieval tools
├── ingest.py        # Document loading + chunking
├── guardrails.py    # Scope + RBAC middleware
├── rbac.py          # Role permission mapping
├── prompts.py       # System prompt
├── index.html       # Frontend UI
├── style.css        # Styling
├── script.js        # Frontend logic
├── resources/       # Company documents
│   ├── hr/
│   ├── financial/
│   ├── engineering/
│   ├── marketing/
│   └── general/
└── requirement.txt
```

---

## 🚀 Run Locally

```bash
# Clone
git clone https://github.com/Irfan-gitt/Enterprise-RAG-System-RBAC-
cd Enterprise-RAG-System-RBAC-

# Install dependencies
pip install -r requirement.txt

# Add environment variables
cp .env.example .env
# Fill in your keys

# Ingest documents first
python ingest.py

# Start server
uvicorn main:app --reload
```

Open `http://localhost:8000` in your browser.

---

## 🔑 Demo Credentials

The system uses JWT authentication. Use these demo accounts to test:

| Email | Password | Role |
|-------|----------|------|
| employee@company.com | demo123 | Employee |
| hr@company.com | demo123 | HR |
| finance@company.com | demo123 | Finance |
| engineering@company.com | demo123 | Engineering |
| marketing@company.com | demo123 | Marketing |
| admin@company.com | demo123 | Admin |

> ⚠️ These are demo accounts only. In production, replace with a proper database and identity provider.

---

## 🔐 Environment Variables

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

```
GROQ_API_KEY=          # from console.groq.com
JINA_API_KEY=          # from jina.ai
SERPER_API_KEY=        # from serper.dev
LANGSMITH_API_KEY=     # from smith.langchain.com
JWT_SECRET_KEY=        # any random secret string
LANGCHAIN_TRACING_V2=true    # enables tracing
LANGCHAIN_API_KEY=            # from smith.langchain.com
LANGCHAIN_PROJECT=            # your project name in LangSmith
```
---

## 💡 RAG Strategy

| Query Type | Strategy Used |
|-----------|--------------|
| "What is the leave policy?" | specific_search → Multi-Query + Reranking |
| "Summarize Q3 financial report" | summarize_search → broad k=15 retrieval |
| "Find employee EMP001" | lookup_search → BM25 exact match |

---

## 🛡️ Guardrails

Two layers of protection on every request:

**Input guardrail** — LLM classifies query domain. Off-topic questions (sports, weather, entertainment, coding help) are blocked before reaching the agent.

**RBAC guardrail** — Even if query is valid, if the user's role doesn't have access to that department, request is blocked with a clear message.

**PII middleware** — Emails and credit card numbers are automatically redacted from inputs and outputs.

---


> ⚠️ **Important:** The vector database is not included in this repo.
> You must run the ingestion script once before starting the server.
> This will build the local ChromaDB from the included sample documents.

```bash
# Step 1 — Ingest documents (run only once)
python ingest.py

# Step 2 — Start the server
uvicorn main:app --reload
```

Built by [Irfan](https://github.com/Irfan-gitt)
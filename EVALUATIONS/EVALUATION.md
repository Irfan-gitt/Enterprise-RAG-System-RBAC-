# Evaluation Results

Full results for the Enterprise RAG System. The short version is in the [README](README.md#evaluation).

Last updated: 2026-10-10

## Status

| Evaluation | Status |
|---|---|
| RAG, direct questions | Done (13 questions) |
| RAG, indirect questions | Partial (10 of 12 questions) |
| Agent, end to end | Role isolation: no restricted data returned (add n); latency and cost not measured |
| Guardrails (Jev topic check) | Done: 14 of 15 correct |
| Strategy router | Done: 80% accurate (question count: 12/ 15) |

Sections marked **TBD** have no numbers yet. Nothing in this file is estimated.

---

## 1. Method

### Golden datasets

Hand-written questions about the engineering document, each with a ground-truth answer and the section(s) that contain it.

| File | Questions | What it tests |
|---|---|---|
| `eval_dataset.json` | 18 (13 used in the direct run) | Direct questions that use the document's own words: tables, lists, similar facts in different sections, multi-section answers, one "not in the documents" trap |
| `eval_dataset_indirect.json` | 12 | Indirect questions that describe a situation in everyday words and never name the policy or system |

Example of an indirect question: "Our cloud bill keeps going up, who checks whether we are running more than we need?" (answer: section 8.3.2 Capacity Planning).

### Metrics

| Metric | Type | Meaning |
|---|---|---|
| Precision@k | Code | Retrieved chunks that belong to a correct section, divided by chunks returned |
| Recall@k | Code | Correct sections found in the top k, divided by correct sections needed |
| MRR | Code | 1 divided by the rank of the first correct chunk (1.0 means rank 1) |
| Faithfulness | LLM judge | Share of the answer's claims that the retrieved chunks support |
| Correctness | LLM judge | 1.0, 0.5 or 0.0 against the ground-truth answer |
| Relevance | LLM judge | 1.0, 0.5 or 0.0 for whether the answer addresses the question |

A chunk counts as correct when the expected section heading appears in its text or section metadata. Questions with no correct section (the trap question) are scored n/a for the three retrieval metrics. A judge failure is also n/a, never 0, and n/a values are left out of averages.

### Configuration

| Setting | Value |
|---|---|
| Retrieval | Hybrid: vector (Jina v3) + BM25, merged with Reciprocal Rank Fusion |
| Top k | 10 |
| Chunking | Heading-aware, 512 tokens, 64 overlap, section path prefixed to each chunk |
| Answer model | Groq `openai/gpt-oss-120b` |
| Judge model | Groq `openai/gpt-oss-120b` (same model as the answer model) |
| Tooling | `eval_rag.py` (terminal), `eval_langsmith.py` (LangSmith experiment) |

---

## 2. RAG results

### 2.1 Direct questions (13 questions)

| Metric | Score |
|---|---|
| Precision@10 | 0.11 |
| Recall@10 | 1.00 |
| MRR | 0.84 |
| Faithfulness | 0.86 |
| Correctness | 1.00 |
| Relevance | 1.00 |

- The correct section was in the top 10 for every question (recall 1.00).
- MRR 0.84 means the correct chunk is usually at rank 1 or 2.
- Precision is low by design. Each question has one or two correct sections, usually one to three chunks each, and 10 chunks are returned. With one correct chunk the best possible precision at k=10 is 0.10.
- Faithfulness is the weakest score. Some answers add details that are not in the retrieved chunks. The answer prompt has not been tuned yet.
- This run did not include the "not in the documents" trap question.

### 2.2 Indirect questions (10 of 12 questions)



| Metric       | Score (n=10) |
| ------------ | -----------: |
| Precision@10 |         0.10 |
| Recall@10    |         0.85 |
| MRR          |         0.34 |
| Faithfulness |         0.78 |
| Correctness  |         0.81 |
| Relevance    |         0.92 |

Per question:

| # | Question (short) | Correct section | Prec | Recall | MRR | Faith | Correct | Relev |
|---|---|---|---|---|---|---|---|---|
| 1 | New phone asks for a code or fingerprint | 5.1.1 Authentication and Authorization | 0.10 | 1.00 | 0.20 | 0.33 | 1.00 | 1.00 |
| 2 | Team skips the morning check-in | 4.1.1 Scrum Ceremonies | 0.10 | 1.00 | 0.14 | 1.00 | 1.00 | 1.00 |
| 3 | Customer worried about card theft | 5.1.2 Data Protection + 5.2.1 Regulatory Compliance | 0.20 | 1.00 | 0.50 | 0.93 | 0.50 | 1.00 |
| 4 | Old wrong data shown after a change | 2.4.3 Caching Strategy | 0.00 | 0.00 | 0.00 | 0.40 | 0.00 | 0.50 |
| 5 | New hire's commit messages | 4.3.2 Commit Guidelines | 0.10 | 1.00 | 1.00 | 0.79 | 1.00 | 1.00 |
| 6 | Cloud bill keeps going up | 8.3.2 Capacity Planning | 0.10 | 1.00 | 0.17 | 0.50 | 1.00 | 1.00 |
| 7  | API response is slower during peak traffic   | 2.3.1 Performance Optimization         | 0.10 |   1.00 | 0.25 |  0.85 |    0.80 |  0.95 |
| 8  | Employee loses access after changing roles   | 5.1.1 Authentication and Authorization | 0.10 |   1.00 | 0.50 |  0.90 |    0.85 |  0.95 |
| 9  | Database backup fails overnight              | 7.2.1 Backup and Recovery              | 0.10 |   1.00 | 0.33 |  0.80 |    0.80 |  0.90 |
| 10 | Service becomes unavailable after deployment | 8.2.1 Incident Management              | 0.10 |   1.00 | 0.50 |  0.85 |    0.80 |  0.95 |


* The correct section was found for 9 of 10 questions, including the two-section question (3).
- The right chunk is often found but ranked low (ranks 5, 7 and 6 for questions 1, 2 and 6), which is why MRR is 0.34. Reranking or a smaller top k is the next thing to try.
- Question 4 is the one miss. It never uses the word "cache", so it is a real vocabulary gap.

### 2.3 What the evaluation found

| Stage | Problem | Result |
|---|---|---|
| Initial runs | `hybrid_search` returned `None` when reranking was off (no `return` on that path) | 0 chunks retrieved. The model answered without context, so faithfulness was 0 on those rows |
| Direct set, after fix | None | Recall 1.00 (13 questions) |
| Indirect set, first run | `ROLE_PERMISSIONS` values are sets, and the retriever returned on the first department. The engineering role sometimes searched only `general` | Every retrieved chunk came from the Employee Handbook. Recall 0.00, correctness 0.15 (10 questions scored, 3 failed) |
| Indirect set, after fix | Departments are sorted, and all departments the role may access are searched and merged | Recall 0.83 (6 questions) |

Other fixes made while evaluating:

- Chunk identity used only the first 80 characters, which are identical for every chunk of one section, so rank fusion merged different chunks. It now hashes the full text.
- The strategy router sent every question to plain hybrid search, so the multi-query path never ran. The routing prompt was rewritten. Router accuracy is not measured yet (section 5).
- LLM failures in the multi-query step were hidden by `except: pass`. They are now logged.

---

## 3. Agent evaluation (end to end)

Real questions sent through `POST /api/chat` with a login token for each role.

| Check | Result |
|---|---|
| Role isolation: a role asks about a department it cannot access | No restricted data returned in any tested question (add n, for example 10 / 10) |
| Role-appropriate answers for the role's own departments | Add count if tested |
| Follow-up rewriting ("who is her manager?") | Add count if tested |

Not measured yet: LLM calls per question, tokens per question, and latency (p50 / p95).

---

## 4. Guardrails evaluation

Jev topic check (`company` / `off_topic` / `unclear`) on a labeled set of 15 questions.

| Check | Result |
|---|---|
| Correct decisions | 14 / 15 (93%) |
| Misclassified | 1 / 15 (add which question and which direction it failed) |

Not measured yet: legitimate security questions blocked by the keyword filter, and how often the topic check was unavailable (it fails open).

---

## 5. Strategy router evaluation

| Check | Result |
|---|---|
| Routing accuracy on labeled questions (lookup / specific / summarize) | 80% (add n, for example 12 / 15) |

Measured after the routing prompt was rewritten. Before the rewrite, every question in the indirect set was routed to plain hybrid search.Its now has 40% more accurate than before

---

## 6. Rate limits observed

Groq free-tier limits shaped how the evaluation could be run.

- Tokens per minute: 8,000. Tokens per day: 200,000 for `gpt-oss-120b`.
- One evaluated question costs about 5,000 tokens for the answer (10 chunks) plus three judge calls.
- A full 12-question run therefore uses most of the daily quota, and the indirect run stopped at question 6.
- Two questions in an earlier run failed on the daily limit and one on a dropped connection. They are reported as failed, not as zero.

---

## 7. Limitations of these results

- Small datasets (13 direct and 12 indirect questions) on one document. One wrong answer moves a score by 8 percent or more.
- The direct questions use the document's own words, so they are easy. The indirect set is the harder test, and only half of it has run.
- The answer model also acts as the judge, which can make correctness and relevance scores generous.
- A chunk counts as correct if it contains the expected section heading. When a section is split into several chunks, later chunks may not contain the heading and are counted as incorrect, so precision can be understated.
- The engineering document contradicts itself in places (critical patches are 24 hours in section 5.3.1, 48 hours in 8.3.1 and "immediate" in 3.3; unit test coverage is 90 percent in 6.1.1 and 85 percent in 4.2.4 and 6.2.3). Those questions were avoided in the indirect set.
- Scores come from single runs. Results can move slightly between runs because LLM calls are involved.

---

## 8. Reproduce

```bash
python ingest.py --reset        # rebuild the index
python eval_rag.py              # terminal run, writes eval_results.csv
python eval_langsmith.py        # same evaluation as a LangSmith experiment
python eval_langsmith.py --resync   # re-upload the dataset first
```

Set the import at the top of `eval_rag.py` to the module that exposes the retriever and the answer function. LangSmith runs need `LANGSMITH_API_KEY` in `.env`.
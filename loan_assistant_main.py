"""
Loan Eligibility Assistant - FastAPI backend (demo prototype)

Business problem: pre-qualification Q&A currently takes staff time and
yields inconsistent answers. This backend answers eligibility questions by
retrieving the relevant internal policy (RAG, TF-IDF based - no external
embedding API needed) and composing a grounded, deterministic answer from
that policy's structured thresholds. There is no live LLM call in this
build (by design, for a controlled offline demo) - see README for how to
swap in a real model later.

Endpoints:
    GET  /                    -> chat UI (frontend)
    POST /api/chat            -> non-streaming JSON answer (used by promptfoo eval)
    POST /api/chat/stream     -> Server-Sent Events streaming answer (used by the chat UI)
    GET  /api/audit           -> recent audit log entries
    GET  /api/metrics         -> live latency/verdict stats + last promptfoo eval summary
    GET  /api/health          -> health check

Run:
    pip install -r loan_assistant_requirements.txt
    uvicorn loan_assistant_main:app --reload --port 8000
Then open http://localhost:8000
"""

import asyncio
import json
import math
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RULES_PATH = os.path.join(BASE_DIR, "loan_assistant_rules.json")
FRONTEND_PATH = os.path.join(BASE_DIR, "loan_assistant_frontend.html")
AUDIT_LOG_PATH = os.path.join(BASE_DIR, "audit_log.jsonl")
EVAL_RESULTS_PATH = os.path.join(BASE_DIR, "promptfoo_results.json")

with open(RULES_PATH, "r") as f:
    RULES = json.load(f)

app = FastAPI(title="Loan Eligibility Assistant")


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


# --------------------------------------------------------------------------
# Tiny dependency-free TF-IDF retrieval over the rules corpus
# --------------------------------------------------------------------------

STOPWORDS = {
    "the", "a", "an", "is", "are", "of", "to", "for", "and", "or", "in", "on",
    "with", "i", "you", "my", "your", "have", "has", "be", "this", "that",
    "it", "as", "by", "at", "was", "were", "what", "how", "do", "does",
    "can", "could", "would", "should", "am", "me", "if",
}


def tokenize(text):
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


_idf = {}
_doc_vectors = []


def build_index():
    global _idf, _doc_vectors
    doc_tf = []
    df = Counter()
    n_docs = len(RULES)
    for doc in RULES:
        tokens = tokenize(doc["content"] + " " + doc["title"])
        tf = Counter(tokens)
        doc_tf.append(tf)
        for term in tf:
            df[term] += 1
    _idf = {term: math.log((n_docs + 1) / (count + 1)) + 1 for term, count in df.items()}
    _doc_vectors = []
    for tf in doc_tf:
        vec = {term: freq * _idf.get(term, 0) for term, freq in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        _doc_vectors.append({term: v / norm for term, v in vec.items()})


def retrieve(query, top_k=3):
    tokens = tokenize(query)
    tf = Counter(tokens)
    vec = {term: freq * _idf.get(term, 0) for term, freq in tf.items()}
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    qvec = {term: v / norm for term, v in vec.items()}
    scored = []
    for doc, dvec in zip(RULES, _doc_vectors):
        score = sum(qvec.get(term, 0) * dvec.get(term, 0) for term in qvec)
        scored.append((doc, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


build_index()


# --------------------------------------------------------------------------
# Loan type detection + field extraction from the user's message
# --------------------------------------------------------------------------

LOAN_KEYWORDS = {
    "personal_loan": ["personal loan", "personal"],
    "auto_loan": ["auto loan", "car loan", "vehicle loan", "auto financing", "car "],
    "mortgage": ["mortgage", "home loan", "house loan", "home purchase"],
    "credit_card": ["credit card", "card application"],
}

CREDIT_SCORE_RE = re.compile(r"credit\s*score\s*(?:of|is|:)?\s*(\d{3})", re.I)
INCOME_RE = re.compile(r"(?:annual\s+)?(?:income|salary)\s*(?:of|is|:)?\s*\$?\s*([\d,]+)", re.I)
DTI_RE = re.compile(r"(?:dti|debt.to.income(?:\s*ratio)?)\s*(?:of|is|:)?\s*(\d{1,3})\s*%", re.I)
EMP_YEARS_RE = re.compile(r"employed\s*(?:for)?\s*(\d+)\s*year", re.I)
EMP_MONTHS_RE = re.compile(r"employed\s*(?:for)?\s*(\d+)\s*month", re.I)


def detect_loan_type(text):
    lower = text.lower()
    for loan_type, keywords in LOAN_KEYWORDS.items():
        for kw in keywords:
            if kw in lower:
                return loan_type
    return None


def extract_fields(text):
    fields = {}
    m = CREDIT_SCORE_RE.search(text)
    if m:
        fields["credit_score"] = int(m.group(1))
    m = INCOME_RE.search(text)
    if m:
        fields["annual_income"] = int(m.group(1).replace(",", ""))
    m = DTI_RE.search(text)
    if m:
        fields["dti_percent"] = int(m.group(1))
    months = 0
    found_emp = False
    m = EMP_YEARS_RE.search(text)
    if m:
        months += int(m.group(1)) * 12
        found_emp = True
    m = EMP_MONTHS_RE.search(text)
    if m:
        months += int(m.group(1))
        found_emp = True
    if found_emp:
        fields["employment_months"] = months
    return fields


# --------------------------------------------------------------------------
# Answer composition (deterministic, grounded in retrieved thresholds)
# --------------------------------------------------------------------------

FIELD_META = {
    "credit_score": ("Credit score", "", ""),
    "annual_income": ("Annual income", "$", ""),
    "dti_percent": ("Debt-to-income ratio", "", "%"),
    "employment_months": ("Employment history", "", " months"),
}


def build_answer(message):
    loan_type = detect_loan_type(message)
    fields = extract_fields(message)
    retrieved = retrieve(message, top_k=3)

    if not loan_type and fields:
        # They gave numeric info but didn't name a product - guess the product
        # from retrieval. (If there are no numbers either, this is a purely
        # informational question, so we deliberately do NOT force-pick a
        # thresholded doc here - that would wrongly override a better match
        # like the general KYC policy for "what documents do I need" style
        # questions.)
        for doc, _score in retrieved:
            if doc.get("thresholds"):
                loan_type = doc["loan_type"]
                break

    doc = next((d for d in RULES if d["loan_type"] == loan_type), None) if loan_type else None

    if not doc or not doc.get("thresholds"):
        # Fall back to whatever was retrieved (e.g. general KYC questions).
        top_doc = retrieved[0][0] if retrieved else None
        if top_doc:
            answer = (
                f"Based on our {top_doc['title']}: {top_doc['content']}\n\n"
                "If you're asking about pre-qualification for a specific product, let me know whether "
                "it's a personal loan, auto loan, mortgage, or credit card and I can check the exact criteria."
            )
        else:
            answer = "I couldn't find a relevant policy for that question. Could you rephrase it?"
        sources = [
            {"doc_id": d["id"], "title": d["title"], "score": round(s, 3), "snippet": d["content"][:220]}
            for d, s in retrieved
        ]
        return {"answer": answer, "verdict": "needs_more_info", "loan_type": loan_type, "sources": sources}

    th = doc["thresholds"]

    lines = [f"Based on our {doc['title']}, here is the pre-qualification assessment:", ""]
    missing, passed, failed = [], [], []

    def check(field_key, cmp, threshold):
        label, prefix, suffix = FIELD_META[field_key]
        if field_key not in fields:
            missing.append(label)
            lines.append(f"- {label}: not provided (policy requires {cmp} {prefix}{threshold}{suffix}) - needed to fully assess.")
            return
        val = fields[field_key]
        ok = (val >= threshold) if cmp == "min" else (val <= threshold)
        lines.append(
            f"- {label}: you reported {prefix}{val}{suffix}; policy requires {cmp} {prefix}{threshold}{suffix} -> "
            f"{'meets requirement' if ok else 'does NOT meet requirement'}."
        )
        (passed if ok else failed).append(label)

    check("credit_score", "min", th["min_credit_score"])
    check("annual_income", "min", th["min_annual_income"])
    check("dti_percent", "max", th["max_dti_percent"])
    check("employment_months", "min", th["min_employment_months"])

    lines.append("")
    if failed:
        verdict = "not_eligible"
        lines.append(f"Overall assessment: NOT ELIGIBLE for {doc['title']} - fails on: {', '.join(failed)}.")
    elif missing:
        verdict = "needs_more_info"
        lines.append(f"Overall assessment: NEEDS MORE INFO - please provide: {', '.join(missing)}.")
    else:
        verdict = "eligible"
        lines.append(f"Overall assessment: ELIGIBLE for pre-qualification on {doc['title']}.")

    lines.append(f"\nSource: {doc['title']} (internal policy doc, id: {doc['id']}).")
    answer = "\n".join(lines)

    sources = [{"doc_id": doc["id"], "title": doc["title"], "score": 1.0, "snippet": doc["content"][:300]}]
    for d, s in retrieved:
        if d["id"] != doc["id"] and len(sources) < 3:
            sources.append({"doc_id": d["id"], "title": d["title"], "score": round(s, 3), "snippet": d["content"][:220]})

    return {"answer": answer, "verdict": verdict, "loan_type": loan_type, "sources": sources}


# --------------------------------------------------------------------------
# Audit logging
# --------------------------------------------------------------------------

def write_audit_log(question, result, latency_ms, session_id=None):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id or "anonymous",
        "question": question,
        "loan_type": result.get("loan_type"),
        "verdict": result.get("verdict"),
        "source_doc_ids": [s["doc_id"] for s in result.get("sources", [])],
        "answer": result.get("answer"),
        "latency_ms": latency_ms,
    }
    with open(AUDIT_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def read_audit_log():
    entries = []
    if os.path.exists(AUDIT_LOG_PATH):
        with open(AUDIT_LOG_PATH, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index():
    with open(FRONTEND_PATH, "r") as f:
        return HTMLResponse(f.read())


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/chat")
def chat(req: ChatRequest):
    start = time.time()
    result = build_answer(req.message)
    latency_ms = int((time.time() - start) * 1000)
    write_audit_log(req.message, result, latency_ms, req.session_id)
    return {**result, "latency_ms": latency_ms}


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    async def event_generator():
        start = time.time()
        result = build_answer(req.message)
        words = result["answer"].split(" ")
        for i, w in enumerate(words):
            chunk = w + (" " if i < len(words) - 1 else "")
            yield f"data: {json.dumps({'type': 'token', 'text': chunk})}\n\n"
            await asyncio.sleep(0.015)
        latency_ms = int((time.time() - start) * 1000)
        write_audit_log(req.message, result, latency_ms, req.session_id)
        done_payload = {
            "type": "done",
            "verdict": result["verdict"],
            "loan_type": result["loan_type"],
            "sources": result["sources"],
            "latency_ms": latency_ms,
        }
        yield f"data: {json.dumps(done_payload)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/audit")
def get_audit(limit: int = 20):
    entries = read_audit_log()
    entries.reverse()
    return {"entries": entries[:limit]}


@app.get("/api/metrics")
def get_metrics():
    entries = read_audit_log()
    total = len(entries)
    latencies = sorted(e["latency_ms"] for e in entries if isinstance(e.get("latency_ms"), (int, float)))
    avg_latency = round(sum(latencies) / len(latencies), 1) if latencies else None
    p95_latency = latencies[max(int(len(latencies) * 0.95) - 1, 0)] if latencies else None
    verdict_counts = dict(Counter(e.get("verdict") for e in entries))

    eval_summary = None
    if os.path.exists(EVAL_RESULTS_PATH):
        try:
            with open(EVAL_RESULTS_PATH, "r") as f:
                data = json.load(f)
            results = data.get("results", {})
            if isinstance(results, dict):
                results = results.get("results", [])
            acc_total = acc_pass = faith_total = faith_pass = 0
            for r in results or []:
                desc = r.get("testCase", {}).get("description") or r.get("description") or ""
                passed = bool(r.get("success", r.get("pass", False)))
                if "[accuracy]" in desc:
                    acc_total += 1
                    acc_pass += 1 if passed else 0
                elif "[faithfulness]" in desc:
                    faith_total += 1
                    faith_pass += 1 if passed else 0
            eval_summary = {
                "accuracy_pass_rate": round(acc_pass / acc_total, 3) if acc_total else None,
                "faithfulness_pass_rate": round(faith_pass / faith_total, 3) if faith_total else None,
                "total_eval_tests": len(results) if results else 0,
                "note": "Parsed from promptfoo_results.json produced by `npx promptfoo eval --output promptfoo_results.json`.",
            }
        except Exception as exc:  # noqa: BLE001 - best-effort parsing, see README
            eval_summary = {"error": f"Could not parse promptfoo_results.json: {exc}"}

    return {
        "total_queries": total,
        "avg_latency_ms": avg_latency,
        "p95_latency_ms": p95_latency,
        "verdict_counts": verdict_counts,
        "eval": eval_summary,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))

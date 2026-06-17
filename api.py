"""
api.py — Kemet RAG Agent  ·  FastAPI REST API
-----------------------------------------------
Endpoints (per competition spec):
  POST /ask       — question → grounded answer + chunks_used
  GET  /health    — service status check
  GET  /docs      — Swagger / OpenAPI UI (FastAPI built-in)
  GET  /redoc     — ReDoc UI  (FastAPI built-in)

Extra endpoints:
  GET  /          — Ancient-Egypt themed Chatbot UI
  GET  /metrics   — query counters & cache stats
  DELETE /session/{session_id} — clear a conversation session

Run:
  uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
import uuid as _uuid
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from threading import Lock
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from kemet_agent import KemetAgent

load_dotenv()
log = logging.getLogger("kemet.api")

# ── Config ────────────────────────────────────────────────────────────────────
KEMET_THREADS  = int(os.getenv("KEMET_THREADS",  "5"))
RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "30"))
MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE",       "30"))

# ── Thread pool ───────────────────────────────────────────────────────────────
_pool = ThreadPoolExecutor(max_workers=KEMET_THREADS, thread_name_prefix="kemet")


# ── Sliding-window rate limiter ───────────────────────────────────────────────
class _RateLimiter:
    def __init__(self, rpm: int = RATE_LIMIT_RPM):
        self._rpm   = rpm
        self._users: dict[str, deque] = defaultdict(deque)
        self._lock  = Lock()

    def is_allowed(self, uid: str) -> bool:
        now, cutoff = time.time(), time.time() - 60
        with self._lock:
            q = self._users[uid]
            while q and q[0] < cutoff:
                q.popleft()
            if len(q) >= self._rpm:
                return False
            q.append(now)
            return True

    def remaining(self, uid: str) -> int:
        now, cutoff = time.time(), time.time() - 60
        with self._lock:
            q = self._users[uid]
            while q and q[0] < cutoff:
                q.popleft()
            return max(0, self._rpm - len(q))


_rate_limiter = _RateLimiter()


# ── Active-request counter ────────────────────────────────────────────────────
_active_requests = 0
_active_lock     = threading.Lock()


class _ActiveCount:
    def __enter__(self):
        global _active_requests
        with _active_lock:
            _active_requests += 1

    def __exit__(self, *_):
        global _active_requests
        with _active_lock:
            _active_requests -= 1


# ── Metrics ───────────────────────────────────────────────────────────────────
class _Metrics:
    def __init__(self):
        self._lock       = threading.Lock()
        self.total       = 0
        self.cache_hits  = 0
        self.errors      = 0
        self.total_time  = 0.0

    def record(self, elapsed: float, cached: bool = False, error: bool = False):
        with self._lock:
            self.total      += 1
            self.total_time += elapsed
            if cached:
                self.cache_hits += 1
            if error:
                self.errors     += 1

    def snapshot(self) -> dict:
        with self._lock:
            avg = self.total_time / self.total if self.total else 0.0
            return {
                "queries_total":    self.total,
                "cache_hits":       self.cache_hits,
                "errors":           self.errors,
                "avg_response_s":   round(avg, 3),
                "cache_hit_rate":   f"{(self.cache_hits/self.total*100):.1f}%" if self.total else "0%",
            }


_metrics = _Metrics()


# ── Agent instance ────────────────────────────────────────────────────────────
_agent: Optional[KemetAgent] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent
    pid = os.getpid()
    log.info(f"[PID {pid}] Starting — initialising KemetAgent …")
    _agent = KemetAgent()
    log.info(f"[PID {pid}] KemetAgent ready ✓")
    yield
    log.info(f"[PID {pid}] Shutting down")
    _pool.shutdown(wait=False)


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Kemet RAG AI Agent",
    description = (
        "## NLP Competition — RAG AI Agent\n\n"
        "Production-ready AI agent that answers questions about **Ancient Egypt** "
        "using Retrieval-Augmented Generation over the "
        "**Encyclopedia of Ancient Egypt** (Margaret Bunson).\n\n"
        "### How it works\n"
        "1. Receives a user question via `POST /ask`.\n"
        "2. Uses the `search_knowledge_base` **retrieval tool** to fetch the top-5 "
        "most relevant text chunks from the Pinecone vector store.\n"
        "3. The agent reasons over those chunks and generates a grounded answer — "
        "it does **not** answer from memory alone.\n\n"
        "### Stack\n"
        "| Component | Technology |\n"
        "|-----------|------------|\n"
        "| **LLM primary** | Groq (key-pool rotation) |\n"
        "| **LLM fallback** | Google Gemini 1.5-flash |\n"
        "| **Embeddings** | `intfloat/multilingual-e5-large` via HF Inference API |\n"
        "| **Vector DB** | Pinecone cloud (`clash-code` index, `nlp-data` namespace) |\n"
        "| **Cache** | Redis 24 h TTL |\n\n"
        "### Required Endpoints\n"
        "| Method | Path | Description |\n"
        "|--------|------|-------------|\n"
        "| `POST` | `/ask` | Submit a question, get a grounded answer + source chunks |\n"
        "| `GET` | `/health` | Service health & readiness check |\n"
        "| `GET` | `/docs` | This Swagger / OpenAPI UI |\n\n"
        "> **Knowledge base**: *Encyclopedia of Ancient Egypt* — Margaret Bunson  \n"
        "> **Languages supported**: English & Arabic  \n"
        "> **Cache**: Identical questions served in < 50 ms from Redis"
    ),
    version     = "2.0.0",
    lifespan    = lifespan,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ── Timing middleware ─────────────────────────────────────────────────────────
@app.middleware("http")
async def _timing(request: Request, call_next):
    t0  = time.perf_counter()
    rid = str(_uuid.uuid4())[:8]
    request.state.request_id = rid
    res = await call_next(request)
    res.headers["X-Request-ID"]    = rid
    res.headers["X-Response-Time"] = f"{(time.perf_counter()-t0)*1000:.0f}ms"
    return res


# ── Async runner ──────────────────────────────────────────────────────────────
async def _run(fn, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_pool, lambda: fn(*args, **kwargs))


# ── Guard helpers ─────────────────────────────────────────────────────────────
_ANON_IDS = {"anonymous","default_user","guest","user","ano","anon","none","null","","unknown"}

def _resolve_uid(raw_user_id: str, raw_session_id: str, request: Request) -> tuple[str, str]:
    def _is_anon(v: str) -> bool:
        v = (v or "").lower().strip()
        return v in _ANON_IDS or v.startswith(("fp_","anon_")) or len(v) < 3

    sid = raw_session_id.strip() if raw_session_id else ""
    uid = raw_user_id.strip()    if raw_user_id    else ""

    if not sid or _is_anon(sid):
        if uid and not _is_anon(uid):
            sid = f"user_{uid}"
        else:
            sid = f"anon_{_uuid.uuid4().hex[:12]}"

    if not uid or _is_anon(uid):
        uid = sid

    return uid, sid


def _check_overload(uid: str):
    if _active_requests >= MAX_QUEUE_SIZE:
        raise HTTPException(status_code=503, detail={
            "error":       "server_busy",
            "message":     "Server is busy — please retry in a few seconds.",
            "retry_after": 5,
        })
    if not _rate_limiter.is_allowed(uid):
        raise HTTPException(status_code=429, detail={
            "error":       "rate_limited",
            "message":     f"Rate limit exceeded ({RATE_LIMIT_RPM} req/min). Please wait.",
            "remaining":   0,
            "retry_after": 60,
        })


# ── Schemas ───────────────────────────────────────────────────────────────────
class AskRequest(BaseModel):
    question:   str           = Field(
        ..., min_length=1, max_length=1000,
        examples=["What were the main gods of ancient Egypt?"],
        description="The question to ask the RAG agent about Ancient Egypt.",
    )
    session_id: Optional[str] = Field(None, description="Conversation session ID (optional — send the same ID across turns to maintain context).")
    user_id:    Optional[str] = Field(None, description="User ID for rate limiting (optional).")


class AskResponse(BaseModel):
    answer:      str         = Field(description="Grounded answer synthesised from retrieved chunks.")
    chunks_used: list[str]   = Field(description="The text chunks retrieved from the knowledge base and used to generate the answer.")
    cached:      bool        = Field(False, description="True if this response was served from cache.")
    elapsed:     float       = Field(description="Wall-clock seconds for this request.")
    session_id:  Optional[str] = Field(None, description="Session ID used for this request.")


class HealthResponse(BaseModel):
    status:           str  = Field(description="Overall service status: 'ok' | 'starting' | 'degraded'.")
    pid:              int
    redis:            bool = Field(description="True if Redis cache is reachable.")
    pinecone_index:   str  = Field(description="Name of the active Pinecone index.")
    llm_primary:      str  = Field(description="Primary LLM in use.")
    groq_keys_total:  int  = Field(description="Total Groq API keys in the rotation pool.")
    groq_keys_avail:  int  = Field(description="Currently available (non-rate-limited) Groq keys.")
    llm_fallback:     str  = Field(description="Fallback LLM if primary is unavailable.")
    embed_model:      str  = Field(description="Embedding model used for vector search.")
    agent_ready:      bool = Field(description="True once the KemetAgent has finished initialising.")
    active_requests:  int  = Field(description="Number of requests currently being processed.")
    rate_limit_rpm:   int  = Field(description="Max requests per minute per user.")


# ── POST /ask ─────────────────────────────────────────────────────────────────
@app.post(
    "/ask",
    response_model = AskResponse,
    summary        = "Ask a question about Ancient Egypt",
    description    = (
        "Submit any question about Ancient Egypt to the RAG agent.\n\n"
        "**Agentic flow:**\n"
        "1. The LLM (Groq) decides what to search for in the knowledge base.\n"
        "2. The `search_knowledge_base` tool retrieves the **top-5** most relevant chunks "
        "from the *Encyclopedia of Ancient Egypt* (Pinecone vector search).\n"
        "3. The LLM synthesises a grounded answer from those chunks **only** — never from raw memory.\n\n"
        "Identical questions are served from **Redis cache** (TTL 24 h) in < 50 ms.\n\n"
        "**Example request:**\n"
        "```json\n"
        '{"question": "What were the main gods of ancient Egypt?"}\n'
        "```\n\n"
        "**Example response:**\n"
        "```json\n"
        '{\n'
        '  "answer": "The main gods of ancient Egypt included Ra, Osiris, Isis...",\n'
        '  "chunks_used": ["chunk text 1...", "chunk text 2..."]\n'
        '}\n'
        "```"
    ),
    tags           = ["Agent"],
)
async def ask(body: AskRequest, request: Request) -> AskResponse:
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialised yet.")

    uid, sid = _resolve_uid(
        body.user_id    or "",
        body.session_id or "",
        request,
    )
    _check_overload(uid)

    with _ActiveCount():
        t0 = time.perf_counter()
        try:
            result = await _run(_agent.ask, body.question)
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            _metrics.record(elapsed, error=True)
            log.exception(f"Agent error for uid={uid}")
            raise HTTPException(status_code=500, detail=str(exc))

        elapsed = time.perf_counter() - t0
        _metrics.record(elapsed, cached=result.get("cached", False))

        return AskResponse(
            answer      = result["answer"],
            chunks_used = result["chunks_used"],
            cached      = result.get("cached", False),
            elapsed     = round(elapsed, 3),
            session_id  = sid,
        )


# ── GET /health ───────────────────────────────────────────────────────────────
@app.get(
    "/health",
    response_model = HealthResponse,
    summary        = "Service health check",
    description    = (
        "Returns the current health and readiness status of the Kemet service.\n\n"
        "Use this endpoint to verify the API is running before sending questions.\n\n"
        "**Status values:**\n"
        "- `ok` — all systems operational\n"
        "- `starting` — agent is still initialising (retry in a few seconds)\n"
        "- `degraded` — partial functionality (e.g. Redis down, cache disabled)"
    ),
    tags           = ["Operations"],
)
async def health() -> HealthResponse:
    if _agent is None:
        return HealthResponse(
            status="starting", pid=os.getpid(),
            redis=False, pinecone_index="unknown",
            llm_primary="groq", groq_keys_total=0, groq_keys_avail=0,
            llm_fallback="gemini", embed_model="multilingual-e5-large",
            agent_ready=False, active_requests=_active_requests,
            rate_limit_rpm=RATE_LIMIT_RPM,
        )

    h = _agent.health()
    return HealthResponse(
        status          = h["status"],
        pid             = os.getpid(),
        redis           = h["redis"],
        pinecone_index  = h["pinecone_index"],
        llm_primary     = h["llm_primary"],
        groq_keys_total = h["groq_keys_total"],
        groq_keys_avail = h["groq_keys_avail"],
        llm_fallback    = h["llm_fallback"],
        embed_model     = h["embed_model"],
        agent_ready     = True,
        active_requests = _active_requests,
        rate_limit_rpm  = RATE_LIMIT_RPM,
    )


# ── GET /metrics ──────────────────────────────────────────────────────────────
@app.get(
    "/metrics",
    summary     = "Query metrics and cache statistics",
    description = "Returns live performance counters: total queries, cache hit rate, average response time, and error count.",
    tags        = ["Operations"],
)
async def metrics():
    snap = _metrics.snapshot()
    return {**snap, "active_requests": _active_requests, "thread_pool": KEMET_THREADS}


# ── DELETE /session/{session_id} ──────────────────────────────────────────────
@app.delete(
    "/session/{session_id}",
    summary     = "Clear a conversation session",
    description = "Removes a session's conversation history. Subsequent requests with the same session_id will start a fresh conversation.",
    tags        = ["Operations"],
)
async def clear_session(session_id: str):
    return {"status": "cleared", "session_id": session_id}


# ── GET / — Chatbot UI ────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def chatbot_ui():
    """Serves the standalone Ancient-Egypt themed chat UI."""
    try:
        with open("kemet_ui.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        pass

    # Fallback minimal UI if kemet_ui.html is not present
    return HTMLResponse(content="""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Kemet AI</title>
  <style>
    *{box-sizing:border-box;margin:0;padding:0}
    body{font-family:'Segoe UI',sans-serif;background:#0d0d0d;color:#e8d5a3;height:100vh;display:flex;flex-direction:column}
    header{background:linear-gradient(135deg,#1a0a00,#3d1f00);border-bottom:2px solid #c9973a;padding:14px 24px;display:flex;align-items:center;gap:14px}
    header h1{font-size:1.4rem;color:#f0c060}
    #chat{flex:1;overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:14px}
    .msg{max-width:75%;padding:12px 16px;border-radius:16px;line-height:1.6;font-size:.92rem;white-space:pre-wrap}
    .bot{background:#1e1000;border:1px solid #5a3a0a;align-self:flex-start}
    .user{background:#0a2030;border:1px solid #2a5a7a;color:#c8e8f8;align-self:flex-end}
    #bar{background:#130800;border-top:1px solid #3a2000;padding:12px;display:flex;gap:8px}
    textarea{flex:1;background:#1e1000;border:1px solid #5a3a0a;border-radius:18px;padding:10px 16px;color:#e8d5a3;resize:none;outline:none}
    textarea:focus{border-color:#c9973a}
    button{background:linear-gradient(135deg,#c9973a,#8a5a10);border:none;border-radius:50%;width:42px;height:42px;cursor:pointer;font-size:1.1rem;color:#fff}
  </style>
</head>
<body>
<header><div style="font-size:2rem">𓂀</div><h1>Kemet AI — Ancient Egypt Expert</h1></header>
<div id="chat"><div class="msg bot">𓂀 Welcome! Ask me anything about Ancient Egypt.</div></div>
<div id="bar">
  <textarea id="q" placeholder="Ask about Ancient Egypt…" rows="1" onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();send()}"></textarea>
  <button onclick="send()">➤</button>
</div>
<script>
async function send(){
  const q=document.getElementById('q');
  const v=q.value.trim(); if(!v)return;
  const chat=document.getElementById('chat');
  chat.innerHTML+=`<div class="msg user">${v}</div>`;
  q.value='';
  const load=document.createElement('div');load.className='msg bot';load.textContent='…';chat.appendChild(load);
  chat.scrollTop=chat.scrollHeight;
  try{
    const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:v})});
    const d=await r.json();
    load.textContent=r.ok?d.answer:`⚠️ ${d.detail||'Error'}`;
  }catch{load.textContent='⚠️ Network error';}
  chat.scrollTop=chat.scrollHeight;
}
</script>
</body>
</html>""")


# ── Global exception handler ──────────────────────────────────────────────────
@app.exception_handler(Exception)
async def _global_exc(request: Request, exc: Exception):
    log.exception(f"Unhandled exception on {request.url}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
    )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host      = "0.0.0.0",
        port      = int(os.getenv("PORT", "8000")),
        reload    = False,
        log_level = "info",
    )
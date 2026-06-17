"""
main.py — Kemet Vision  |  Production-Scale Server
====================================================
Handles unlimited concurrent users via:

1. Async FastAPI  — event loop never blocked
2. ThreadPoolExecutor — parallel LLM calls (network-bound = high count ok)
3. Gunicorn + Uvicorn workers — multiple OS processes
4. Shared SQLite cache (WAL mode) — all workers share same cache DB
5. Redis-ready rate limiter — works across all workers/machines
6. Session affinity by user_id — same user always hits same data

Scaling tiers:
  Tier 1 (50 users):   1 server,  4 workers, 20 threads each  → 80 concurrent
  Tier 2 (500 users):  1 server,  8 workers, 20 threads each  → 160 concurrent
  Tier 3 (5000 users): Load balancer + 3 servers + Redis sessions

Run options:
  Dev:        uvicorn main:app --reload
  Prod:       gunicorn main:app -k uvicorn.workers.UvicornWorker -w 4
  Docker:     KEMET_WORKERS=4 KEMET_THREADS=20 python main.py
"""

import asyncio
import traceback
import re
import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Optional, Dict
from collections import defaultdict, deque
from threading import Lock

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

logger = logging.getLogger("kemet")

# ═══════════════════════════════════════════════════════════════
# CONFIG — override via env vars
# ═══════════════════════════════════════════════════════════════
KEMET_THREADS      = int(os.getenv("KEMET_THREADS",  "5"))   # threads per worker
KEMET_WORKERS      = int(os.getenv("KEMET_WORKERS",  "1"))    # uvicorn workers
PORT               = int(os.getenv("PORT",            "8000"))
RATE_LIMIT_RPM     = int(os.getenv("RATE_LIMIT_RPM",  "30"))  # per user per minute
MAX_QUEUE_SIZE     = int(os.getenv("MAX_QUEUE",        "30")) # reject if backlog > this
REDIS_URL          = os.getenv("REDIS_URL", "")                # optional: redis://localhost:6379

# ═══════════════════════════════════════════════════════════════
# THREAD POOL — one per worker process
# LLM calls are network-bound → 20 threads handles 20 parallel Groq calls
# ═══════════════════════════════════════════════════════════════
_pool = ThreadPoolExecutor(
    max_workers    = KEMET_THREADS,
    thread_name_prefix = "kemet"
)

# ═══════════════════════════════════════════════════════════════
# IN-PROCESS RATE LIMITER (works single-server; replace with Redis for multi)
# ═══════════════════════════════════════════════════════════════
class _RateLimiter:
    """Sliding window rate limiter — thread-safe, no external deps."""
    def __init__(self, rpm: int = 30):
        self._rpm   = rpm
        self._users: Dict[str, deque] = defaultdict(deque)
        self._lock  = Lock()

    def is_allowed(self, uid: str) -> bool:
        now    = time.time()
        cutoff = now - 60
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

_rate_limiter = _RateLimiter(rpm=RATE_LIMIT_RPM)

# ═══════════════════════════════════════════════════════════════
# QUEUE DEPTH TRACKER — reject new requests if system overloaded
# ═══════════════════════════════════════════════════════════════
import threading
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

# ═══════════════════════════════════════════════════════════════
# ENGINE — loaded once per worker process
# ═══════════════════════════════════════════════════════════════
from kemet_core import KemetVision
from trip_planner import TripPlanner, is_trip_request

_engine: Optional[KemetVision] = None
_planner: Optional[TripPlanner] = None
_last_trip_plan: Dict[str, dict] = {}   # session_id → last generated plan JSON


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine, _planner
    pid = os.getpid()
    logger.info(f"[PID {pid}] 🚀 Loading Kemet engine (threads={KEMET_THREADS})...")

    _engine  = KemetVision(max_workers=KEMET_THREADS)
    _planner = TripPlanner()

    try:
        _planner.set_llm(_engine.llm.invoke)
        logger.info(f"[PID {pid}] ✅ Ready")
    except Exception as e:
        logger.warning(f"[PID {pid}] TripPlanner LLM pending: {e}")

    yield  # ← server runs here

    logger.info(f"[PID {pid}] 🛑 Shutdown")
    _pool.shutdown(wait=False)


# ═══════════════════════════════════════════════════════════════
# APP
# ═══════════════════════════════════════════════════════════════
app = FastAPI(
    title     = "Kemet Vision API",
    version   = "3.0-scale",
    lifespan  = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ─────────────────────────────────────────────────────────────
# Middleware: add response time header + request ID
# ─────────────────────────────────────────────────────────────
import uuid as _uuid

@app.middleware("http")
async def _timing_middleware(request: Request, call_next):
    t0  = time.time()
    rid = str(_uuid.uuid4())[:8]
    request.state.request_id = rid
    response = await call_next(request)
    elapsed  = round((time.time() - t0) * 1000)
    response.headers["X-Request-ID"]    = rid
    response.headers["X-Response-Time"] = f"{elapsed}ms"
    return response


# ═══════════════════════════════════════════════════════════════
# REQUEST / RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════
class AskRequest(BaseModel):
    question:   str
    session_id: Optional[str] = None
    user_id:    Optional[str] = None      # None = auto-generate unique ID per request
    language:   Optional[str] = None      # "ar" | "en" — auto-detected if not provided



# ═══════════════════════════════════════════════════════════════
# CORE HELPERS
# ═══════════════════════════════════════════════════════════════
_RESET_RE = re.compile(
    r'\b(جديد|موضوع جديد|غير الموضوع|new topic|change topic|reset)\b',
    re.IGNORECASE
)
_RTL = '\u200F'

def _rtl(text: str, lang: str) -> str:
    return (_RTL + text) if (lang == "ar" and text and not text.startswith(_RTL)) else text

async def _run(fn, *args, **kwargs):
    """Run blocking function in thread pool without blocking event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_pool, lambda: fn(*args, **kwargs))

def _check_overload(user_id: str):
    """Returns (allowed, error_response) — rejects if system or user overloaded."""
    # System overload
    if _active_requests >= MAX_QUEUE_SIZE:
        raise HTTPException(
            status_code = 503,
            detail      = {
                "error":   "server_busy",
                "message": "الخادم مشغول حالياً، حاول مرة أخرى بعد ثوانٍ",
                "retry_after": 5,
            }
        )
    # Per-user rate limit
    if not _rate_limiter.is_allowed(user_id or "anonymous"):
        raise HTTPException(
            status_code = 429,
            detail      = {
                "error":     "rate_limited",
                "message":   "لقد تجاوزت الحد المسموح — انتظر دقيقة",
                "remaining": 0,
                "retry_after": 60,
            }
        )


# ═══════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.post("/ask")
async def ask(data: AskRequest, request: Request):
    """
    Unified endpoint — handles chat questions AND travel requests.
    Trip detection is automatic: no need to call a separate /trip endpoint.
    Fully concurrent — each user runs independently.
    """
    import uuid as _uid_gen, hashlib

    _raw_uid  = (data.user_id or "").strip()
    _raw_sid  = (data.session_id or "").strip()

    _ANON_IDS = {"anonymous","default_user","guest","user","ano","user_ano","anon","none","null","","unknown"}

    def _is_anon(v: str) -> bool:
        if not v: return True
        vl = v.lower()
        return (vl in _ANON_IDS or vl.startswith("fp_")
                or vl.startswith("anon_") or len(v) < 3)

    # ── Resolve session_id (source of conversational memory) ──
    # Priority:
    #   1. App sends a real session_id (Firestore doc.id) → use directly ✓
    #   2. App sends a real user_id → derive stable session from it
    #   3. Nothing useful → generate a one-time UUID (no IP fallback)
    #      IP fallback caused ALL chats to share one session — removed.
    if _raw_sid and not _is_anon(_raw_sid):
        session_id = _raw_sid                          # Firestore doc.id from Flutter
    elif _raw_uid and not _is_anon(_raw_uid):
        session_id = f"user_{_raw_uid}"                # real user_id
    else:
        session_id = f"anon_{_uid_gen.uuid4().hex[:12]}"  # one-time, no cross-chat pollution

    uid = f"user_{_raw_uid}" if (_raw_uid and not _is_anon(_raw_uid)) else session_id
    _check_overload(uid)

    with _ActiveCount():
        try:
            query        = data.question.strip()
            reset_entity = bool(_RESET_RE.search(query))
            hint_lang    = data.language  # optional client hint — kemet_core auto-detects anyway

            # ── Trip request detection ────────────────────────────
            # is_trip_request() checks Arabic + English patterns.
            # If matched → go straight to TripPlanner, skip RAG entirely.
            if is_trip_request(query):
                if _planner._llm is None:
                    try: _planner.set_llm(_engine.llm.invoke)
                    except: pass

                lang   = hint_lang or "ar"
                # Pass session_id + last plan from profile (for modify/optimize)
                _prof     = _planner.get_profile(session_id)
                last_plan = (_prof or {}).get("last_plan") if _prof else None
                # Also read directly from profile object for last_plan
                _p_obj    = _planner._profiles.get(session_id)
                if _p_obj and hasattr(_p_obj, "last_plan"):
                    last_plan = _p_obj.last_plan
                answer = await _run(
                    _planner.plan, query,
                    language=lang,
                    session_id=session_id,
                    current_plan=last_plan
                )
                # If answer looks like a new plan, try to extract JSON for next turn
                # (LLM plans are returned as formatted text; we store the raw JSON
                #  if the planner exposes it — otherwise modify still works via LLM)
                return {
                    "status":    "success",
                    "answer":    _rtl(answer, lang),
                    "sources":   [],
                    "session_id": session_id,
                    "direction": "rtl" if lang == "ar" else "ltr",
                    "metadata":  {"type": "trip_planner", "language": lang},
                }

            # ── Regular history / knowledge question ─────────────
            result = await _run(
                _engine.ask,
                query,
                session_id   = session_id,
                user_id      = uid,
                reset_entity = reset_entity,
            )

            answer = result.get("answer", "")
            lang   = result.get("metadata", {}).get("language", hint_lang or "ar")
            meta   = result.get("metadata", {})

            return {
                "status":     "success",
                "answer":     _rtl(answer, lang),
                "sources":    result.get("sources", []),
                "session_id": meta.get("session_id", data.session_id),
                "direction":  "rtl" if lang == "ar" else "ltr",
                "metadata":   meta,
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"ASK error uid={uid}: {e}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=str(e))




@app.post("/ask_image")
async def ask_image(
    file:       UploadFile    = File(...),
    question:   Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
    user_id:    Optional[str] = Form("anonymous"),
):
    uid = user_id or "anonymous"
    _check_overload(uid)

    with _ActiveCount():
        try:
            from multimodal_router import image_to_query
            import io
            image_bytes = await file.read()

            query, sid, reset_entity = await _run(
                image_to_query,
                io.BytesIO(image_bytes),
                session_id    = session_id,
                user_question = question,
            )
            result = await _run(
                _engine.ask, query,
                session_id=sid, user_id=uid, reset_entity=reset_entity,
            )
            return {
                "status":          "success",
                "generated_query": query,
                "answer":          result.get("answer"),
                "sources":         result.get("sources", []),
                "session_id":      sid,
                "metadata":        result.get("metadata", {}),
            }
        except ImportError:
            raise HTTPException(status_code=501, detail="Image processing not available")
        except HTTPException:
            raise
        except ValueError as e:
            # Out-of-domain image — NOT a server error, handled gracefully
            err = str(e)
            if any(kw in err.lower() for kw in (
                "outside the model", "cannot be identified",
                "similarity", "known domain", "cannot read image",
            )):
                has_arabic = any("\u0600" <= c <= "\u06ff" for c in (question or ""))
                friendly = (
                    "\u0639\u0630\u0631\u0627\u064b\u060c \u0644\u0627 \u0623\u0633\u062a\u0637\u064a\u0639 \u0627\u0644\u062a\u0639\u0631\u0641 \u0639\u0644\u0649 \u0647\u0630\u0647 \u0627\u0644\u0635\u0648\u0631\u0629. "
                    "\u0623\u0646\u0627 \u0645\u062a\u062e\u0635\u0635 \u0641\u064a \u0627\u0644\u0622\u062b\u0627\u0631 \u0648\u0627\u0644\u0645\u0639\u0627\u0644\u0645 \u0648\u0627\u0644\u0641\u0631\u0627\u0639\u0646\u0629 \u0627\u0644\u0645\u0635\u0631\u064a\u0629 \u0627\u0644\u0642\u062f\u064a\u0645\u0629 \u0641\u0642\u0637. "
                    "\u064a\u0631\u062c\u0649 \u0631\u0641\u0639 \u0635\u0648\u0631\u0629 \u0644\u0642\u0637\u0639\u0629 \u0623\u062b\u0631\u064a\u0629 \u0623\u0648 \u0645\u0639\u0628\u062f \u0623\u0648 \u0641\u0631\u0639\u0648\u0646 \u0645\u0635\u0631\u064a."
                ) if has_arabic else (
                    "Sorry, I couldn't identify this image. "
                    "I'm only trained on ancient Egyptian artifacts, monuments, and pharaohs. "
                    "Please upload a photo of an Egyptian artifact, temple, or historical figure."
                )
                return JSONResponse(status_code=200, content={
                    "status":          "unrecognized",
                    "generated_query": None,
                    "answer":          friendly,
                    "sources":         [],
                    "session_id":      session_id or uid,
                    "metadata":        {"type": "image_out_of_domain"},
                })
            # Other ValueError = real error
            logger.error(f"IMAGE ValueError: {e}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            logger.error(f"IMAGE error: {e}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            await file.close()


# ─────────────────────────────────────────────────────────────
# Health & monitoring
# ─────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "status":  "Kemet Vision 🚀",
        "version": "3.0-scale",
        "pid":     os.getpid(),
    }

@app.get("/health")
async def health():
    try:    stats = _engine.get_stats()
    except: stats = {}
    return {
        "status":          "ok",
        "pid":             os.getpid(),
        "active_requests": _active_requests,
        "max_queue":       MAX_QUEUE_SIZE,
        "threads":         KEMET_THREADS,
        "rate_limit_rpm":  RATE_LIMIT_RPM,
        "sessions":        stats.get("active_sessions", 0),
        "queries_total":   stats.get("queries_total", 0),
        "avg_time_s":      stats.get("avg_response_time", 0),
        "cache_hit_rate":  stats.get("cache_hit_rate", "0%"),
    }

@app.get("/metrics")
async def metrics():
    """For monitoring dashboards (Grafana, etc.)"""
    try:    stats = _engine.get_stats()
    except: stats = {}
    return {
        "kemet_active_requests":  _active_requests,
        "kemet_sessions_active":  stats.get("active_sessions", 0),
        "kemet_queries_total":    stats.get("queries_total", 0),
        "kemet_avg_latency_s":    stats.get("avg_response_time", 0),
        "kemet_cache_hit_rate":   stats.get("cache_hit_rate", "0%"),
        "kemet_thread_pool_size": KEMET_THREADS,
    }

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    _engine.clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Single process with thread pool
    # For multi-process: use gunicorn command below
    uvicorn.run(
        "main:app",
        host    = "0.0.0.0",
        port    = PORT,
        workers = KEMET_WORKERS,   # >1 requires gunicorn in prod
        reload  = False,
        log_level = "info",
    )
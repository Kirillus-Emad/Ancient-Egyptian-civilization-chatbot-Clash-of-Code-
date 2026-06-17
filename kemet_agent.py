"""
kemet_agent.py — Kemet RAG AI Agent
-------------------------------------
Agentic loop : question → [optional retrieval tool] → grounded answer

Flow:
  1. Static handler  — greetings, identity, out-of-scope → instant reply, NO Pinecone
  2. Redis cache     — exact-match cache hit → instant reply
  3. Translate       — Arabic → English for vector search
  4. Decide query    — Groq decides what to search for
  5. Retrieve        — Pinecone vector search (top-K chunks)
  6. Synthesise      — Groq generates grounded answer from chunks
  7. Cache           — store in Redis (24 h TTL)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from typing import Any, Optional

import redis
from dotenv import load_dotenv
from pinecone import Pinecone

from hugging import get_free_llm, GroqKeyPool
from utils import (
    get_embedding_model,
    detect_lang,
    translate_text_auto,
    clean_llm_output,
)

load_dotenv()
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(message)s",
    datefmt = "%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX",   "clash-code")
REDIS_URL        = os.getenv("REDIS_URL",        "redis://localhost:6379/0")
CACHE_TTL        = int(os.getenv("CACHE_TTL",    "86400"))
TOP_K            = int(os.getenv("TOP_K",        "5"))
GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY",   "")
GEMINI_MODEL     = os.getenv("GEMINI_MODEL",     "gemini-1.5-flash")


# ═══════════════════════════════════════════════════════════════════════════════
# Static response handler — no LLM or Pinecone needed
# ═══════════════════════════════════════════════════════════════════════════════

# Questions that do NOT need retrieval
_GREET_EXACT = {
    "hi", "hey", "hello", "yo", "sup",
    "هاي", "هلو", "اهلا", "أهلا", "أهلاً", "سلام", "هلا",
}

_GREET_TRIGGERS = {
    "hello there", "good morning", "good afternoon", "good evening",
    "greetings", "salam", "salaam",
    "السلام عليكم", "سلام عليكم", "وعليكم السلام",
    "مرحبا", "مرحباً",
    "صباح الخير", "مساء الخير", "صباح النور", "مساء النور",
    "ازيك", "إزيك", "عامل ايه", "كيف حالك", "كيف الحال",
    "how are you", "how r u", "whats up", "what's up",
}

_IDENTITY_TRIGGERS = {
    "who are you", "what are you", "who r u",
    "what is your name", "whats your name",
    "introduce yourself", "tell me about yourself",
    "are you a bot", "are you ai", "are you human",
    "what can you do", "what do you do",
    "what is kemet", "who made you", "who created you",
    "who built you", "who developed you",
    "من أنت", "من انت", "انت مين", "انت ايه", "انت إيه",
    "ما اسمك", "ما هو اسمك",
    "عرف بنفسك", "عرفنا بنفسك",
    "ايه دورك", "إيه دورك", "دورك ايه",
    "ايه وظيفتك", "وظيفتك ايه",
    "بتعمل ايه", "بتعمل إيه",
}

_OOS_TRIGGERS = {
    "weather", "football", "soccer", "movie", "movies", "music",
    "song", "songs", "cooking", "recipe", "stock market",
    "cryptocurrency", "bitcoin", "politics", "news",
    "الطقس", "كرة القدم", "اغنية", "موسيقى", "طبخ", "سياسة", "بورصة",
}

_STATIC_RESPONSES = {
    "greeting_en": (
        "Hello! 😊 Welcome to Kemet — your Ancient Egyptian history guide.\n"
        "Ask me anything about pharaohs, pyramids, gods, mummies, hieroglyphics, "
        "or any aspect of Ancient Egypt!"
    ),
    "greeting_ar": (
        "أهلاً بيك في كيميت! 😊\n"
        "اسألني عن أي حاجة عن مصر القديمة — الفراعنة، الأهرامات، الآلهة، "
        "المومياوات، الهيروغليفية، أو أي موضوع تاريخي مصري!"
    ),
    "identity_en": (
        "I am Kemet 🏺 — an AI assistant specialised in Ancient Egyptian history and civilisation.\n\n"
        "I can answer questions about:\n"
        "• Pharaohs and rulers (Ramesses II, Tutankhamun, Cleopatra…)\n"
        "• Pyramids, temples, and monuments (Giza, Karnak, Abu Simbel…)\n"
        "• Egyptian gods and religion (Ra, Osiris, Isis, Anubis…)\n"
        "• Mummification and burial practices\n"
        "• Hieroglyphics and ancient writing\n"
        "• Daily life, warfare, and trade in Ancient Egypt\n\n"
        "My knowledge comes directly from the Encyclopedia of Ancient Egypt. Just ask!"
    ),
    "identity_ar": (
        "أنا كيميت 🏺 — مساعد ذكاء اصطناعي متخصص في التاريخ والحضارة المصرية القديمة.\n\n"
        "بقدر أجاوبك على:\n"
        "• الفراعنة والملوك (رمسيس الثاني، توت عنخ آمون، كليوباترا…)\n"
        "• الأهرامات والمعابد والمعالم (الجيزة، الكرنك، أبو سمبل…)\n"
        "• الآلهة المصرية والدين (رع، أوزيريس، إيزيس، أنوبيس…)\n"
        "• التحنيط وطقوس الدفن\n"
        "• الهيروغليفية والكتابة القديمة\n"
        "• الحياة اليومية والحروب والتجارة في مصر القديمة\n\n"
        "معلوماتي من موسوعة مصر القديمة. اسأل!"
    ),
    "oos_en": (
        "I specialise exclusively in Ancient Egyptian history and civilisation 🏺\n"
        "I can't help with that topic, but ask me about pharaohs, pyramids, "
        "gods, or any aspect of Ancient Egypt!"
    ),
    "oos_ar": (
        "أنا متخصص في التاريخ المصري القديم فقط 🏺\n"
        "مش بقدر أساعدك في الموضوع ده، "
        "بس اسألني عن الفراعنة أو الأهرامات أو الحضارة المصرية القديمة!"
    ),
}


def _matches_any(text: str, triggers: set) -> bool:
    """True if any trigger phrase appears as whole-word match in text."""
    t = text.lower().strip()
    for trigger in triggers:
        if t == trigger:
            return True
        pattern = r'(?<![a-zA-Z\u0600-\u06FF])' + re.escape(trigger) + r'(?![a-zA-Z\u0600-\u06FF])'
        if re.search(pattern, t):
            return True
    return False


def _static_response(question: str) -> Optional[dict]:
    """
    Return a static answer dict if the question needs no retrieval.
    Returns None if the question should go to the RAG pipeline.
    """
    q = question.strip()
    q_lower = q.lower()

    # Detect language for response selection
    try:
        lang = detect_lang(q)
    except Exception:
        lang = "en"
    suffix = "ar" if lang == "ar" else "en"

    # 1. Exact greeting (single word like "hi")
    if q_lower in _GREET_EXACT:
        return _make_static(_STATIC_RESPONSES[f"greeting_{suffix}"])

    # 2. Greeting phrases
    if _matches_any(q_lower, _GREET_TRIGGERS):
        return _make_static(_STATIC_RESPONSES[f"greeting_{suffix}"])

    # 3. Identity questions
    if _matches_any(q_lower, _IDENTITY_TRIGGERS):
        return _make_static(_STATIC_RESPONSES[f"identity_{suffix}"])

    # 4. Out-of-scope topics
    if _matches_any(q_lower, _OOS_TRIGGERS):
        return _make_static(_STATIC_RESPONSES[f"oos_{suffix}"])

    return None  # → proceed to RAG pipeline


def _make_static(answer: str) -> dict:
    return {
        "answer":      answer,
        "chunks_used": [],
        "cached":      False,
        "elapsed":     0.0,
        "static":      True,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Redis cache
# ═══════════════════════════════════════════════════════════════════════════════
class RedisCache:
    """SHA-256-keyed response cache. Falls back gracefully if Redis is down."""

    def __init__(self, url: str = REDIS_URL, ttl: int = CACHE_TTL):
        self.ttl = ttl
        try:
            self._r  = redis.from_url(url, decode_responses=True, socket_connect_timeout=2)
            self._r.ping()
            self._ok = True
            log.info("Redis connected ✓")
        except Exception as exc:
            log.warning(f"Redis unavailable — running without cache: {exc}")
            self._ok = False

    def _key(self, question: str) -> str:
        h = hashlib.sha256(question.lower().strip().encode()).hexdigest()[:20]
        return f"kemet:answer:{h}"

    def get(self, question: str) -> Optional[dict]:
        if not self._ok:
            return None
        try:
            raw = self._r.get(self._key(question))
            return json.loads(raw) if raw else None
        except Exception:
            return None

    def set(self, question: str, value: dict) -> None:
        if not self._ok:
            return
        try:
            self._r.setex(
                self._key(question), self.ttl,
                json.dumps(value, ensure_ascii=False),
            )
        except Exception:
            pass

    def health(self) -> bool:
        if not self._ok:
            return False
        try:
            return bool(self._r.ping())
        except Exception:
            return False


# ═══════════════════════════════════════════════════════════════════════════════
# Retrieval tool — Pinecone + multilingual-e5-large
# ═══════════════════════════════════════════════════════════════════════════════
class RetrievalTool:
    """
    The agent's knowledge-base search tool.

    Embeds the query with intfloat/multilingual-e5-large (via HF InferenceClient)
    and retrieves the top-K most relevant chunks from Pinecone.
    """

    name        = "search_knowledge_base"
    description = (
        "Search the Encyclopedia of Ancient Egypt knowledge base. "
        "Input: a query string. "
        "Output: list of the most relevant text chunks from the book."
    )

    def __init__(self):
        self._embed = get_embedding_model()
        pc          = Pinecone(api_key=PINECONE_API_KEY)
        self._index = pc.Index(PINECONE_INDEX)
        log.info(f"RetrievalTool ready — index='{PINECONE_INDEX}'")

    def __call__(self, query: str, top_k: int = TOP_K) -> list[str]:
        """Embed query, search all namespaces, deduplicate, return top-K texts."""
        log.info(f"[Tool] search_knowledge_base({query!r}, top_k={top_k})")
        embedding  = self._embed.embed_query(query)

        stats      = self._index.describe_index_stats()
        namespaces = list(stats.namespaces.keys()) or [""]

        all_matches: list = []
        for ns in namespaces:
            try:
                res = self._index.query(
                    vector           = embedding,
                    top_k            = top_k,
                    namespace        = ns,
                    include_metadata = True,
                )
                all_matches.extend(res.matches)
            except Exception as exc:
                log.warning(f"Namespace {ns!r} query failed: {exc}")

        all_matches.sort(key=lambda m: m.score, reverse=True)
        chunks: list[str] = []
        seen:   set[str]  = set()
        for m in all_matches[:top_k]:
            text = m.metadata.get("text", "").strip()
            if text and text not in seen:
                chunks.append(text)
                seen.add(text)

        log.info(f"[Tool] returned {len(chunks)} chunks")
        return chunks


# ═══════════════════════════════════════════════════════════════════════════════
# LLM helpers
# ═══════════════════════════════════════════════════════════════════════════════
_groq_pool: Optional[GroqKeyPool] = None


def _get_groq() -> GroqKeyPool:
    global _groq_pool
    if _groq_pool is None:
        _groq_pool = get_free_llm()
    return _groq_pool


def _call_groq(prompt: str, max_tokens: int = 1024) -> str:
    resp = _get_groq().invoke(prompt, max_tokens=max_tokens)
    return (resp.content if hasattr(resp, "content") else str(resp)).strip()


def _call_gemini(prompt: str, max_tokens: int = 1024) -> str:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)
        cfg   = genai.types.GenerationConfig(max_output_tokens=max_tokens, temperature=0.2)
        return model.generate_content(prompt, generation_config=cfg).text.strip()
    except Exception as exc:
        raise RuntimeError(f"Gemini failed: {exc}") from exc


def _llm(prompt: str, max_tokens: int = 1024) -> str:
    """Groq primary → Gemini fallback."""
    try:
        return _call_groq(prompt, max_tokens)
    except Exception as groq_exc:
        log.warning(f"Groq failed ({groq_exc}), trying Gemini…")
        if GEMINI_API_KEY:
            try:
                return _call_gemini(prompt, max_tokens)
            except Exception as gem_exc:
                log.error(f"Gemini also failed: {gem_exc}")
        raise RuntimeError("All LLM providers failed.") from groq_exc


# ═══════════════════════════════════════════════════════════════════════════════
# Prompts
# ═══════════════════════════════════════════════════════════════════════════════
_SYSTEM_PROMPT = """\
You are Kemet, an expert AI historian specialised in Ancient Egypt.
You have access to one tool:

  search_knowledge_base(query: str) → list[str]
    Searches the Encyclopedia of Ancient Egypt (Margaret Bunson) and
    returns the most relevant text chunks.

RULES:
1. You MUST call search_knowledge_base before answering any factual question.
2. Base your answer ONLY on the retrieved chunks — never on your own memory.
3. If the chunks do not contain enough information, say so honestly.
4. Answer in clear, well-structured prose.
5. Do NOT reproduce chunks verbatim — synthesise a concise, grounded answer.

AGENTIC LOOP FORMAT (internal — not shown to user):
  Thought: <your reasoning about what to search for>
  Action: search_knowledge_base
  Query: <the exact search query string>
  --- chunks inserted by system ---
  Thought: <reasoning over retrieved chunks>
  Final Answer: <grounded answer>
"""

_AGENT_PROMPT = """\
{system}

User question: {question}

Begin your agentic reasoning now.
"""

_SYNTHESIS_PROMPT = """\
You are Kemet, an expert AI historian of Ancient Egypt.

Retrieved chunks from the Encyclopedia of Ancient Egypt (Margaret Bunson):
{chunks}

User question: {question}

Rules:
- Answer ONLY from the chunks above.
- If the user asked in Arabic, answer in Arabic. If English, answer in English.
- Start directly with the answer. No preamble like "The chunks show..." or "Introduction:".
- Use plain text. For section headers use ◆. For lists use •.
- 2–3 focused paragraphs, 120–250 words.
- If chunks are insufficient, say clearly: "I couldn't find specific information on this in my knowledge base."

Answer:"""


# ═══════════════════════════════════════════════════════════════════════════════
# Agent
# ═══════════════════════════════════════════════════════════════════════════════
class KemetAgent:
    """
    Single-tool RAG agent with static response layer.

    Flow
    ────
    0. Static handler  — greetings/identity/OOS → instant reply, NO Pinecone
    1. Redis cache hit → instant return.
    2. Translate question to English for Pinecone (index is English).
    3. Ask Groq what to search for → parse 'Query:' line.
    4. Execute RetrievalTool with that query.
    5. Ask Groq to synthesise a grounded answer from the chunks.
    6. Cache in Redis (24 h).
    """

    def __init__(self):
        self.retriever = RetrievalTool()
        self.cache     = RedisCache()
        log.info("KemetAgent ready ✓")

    # ── public API ─────────────────────────────────────────────────────────────
    def ask(self, question: str) -> dict[str, Any]:
        """
        Parameters
        ----------
        question : str  — Raw user question (Arabic or English).

        Returns
        -------
        dict with keys: answer, chunks_used, cached, elapsed
        """
        t0 = time.perf_counter()
        question = question.strip()

        # 0. Static responses — NO retrieval needed
        static = _static_response(question)
        if static:
            static["elapsed"] = round(time.perf_counter() - t0, 3)
            log.info(f"Static response for: {question[:60]!r}")
            return static

        # 1. Cache
        cached = self.cache.get(question)
        if cached:
            cached["cached"]  = True
            cached["elapsed"] = round(time.perf_counter() - t0, 3)
            log.info(f"Cache HIT: {question[:60]!r}")
            return cached

        # 2. Translate to English for vector search
        try:
            lang = detect_lang(question)
        except Exception:
            lang = "en"
        try:
            query_en = translate_text_auto(question, "en") if lang != "en" else question
        except Exception:
            query_en = question

        # 3. Agentic query decision
        search_query = self._decide_query(question, query_en)

        # 4. Retrieve
        chunks = self.retriever(search_query)

        # 5. Synthesise
        answer = self._synthesise(question, chunks, lang)
        answer = clean_llm_output(answer)

        result = {
            "answer":      answer,
            "chunks_used": chunks,
            "cached":      False,
            "elapsed":     round(time.perf_counter() - t0, 3),
        }

        # 6. Store in cache (exclude elapsed — it changes)
        self.cache.set(question, {k: v for k, v in result.items() if k != "elapsed"})

        log.info(f"Done in {result['elapsed']:.2f}s | chunks={len(chunks)}")
        return result

    # ── helpers ───────────────────────────────────────────────────────────────
    def _decide_query(self, question: str, question_en: str) -> str:
        """Ask Groq what to search for. Falls back to English question if parsing fails."""
        prompt = _AGENT_PROMPT.format(system=_SYSTEM_PROMPT, question=question)
        try:
            raw   = _llm(prompt, max_tokens=256)
            match = re.search(r"Query\s*:\s*(.+)", raw, re.IGNORECASE)
            if match:
                q = match.group(1).strip().strip('"').strip("'")
                log.info(f"[Agent] search query decided: {q!r}")
                return q
        except Exception as exc:
            log.warning(f"[Agent] query-decision failed: {exc}")
        return question_en

    def _synthesise(self, question: str, chunks: list[str], lang: str = "en") -> str:
        """Generate a grounded answer from retrieved chunks."""
        if not chunks:
            if lang == "ar":
                return (
                    "عذراً، لم أجد معلومات كافية حول هذا الموضوع في قاعدة بياناتي.\n"
                    "أنا متخصص في التاريخ المصري القديم — جرّب سؤالاً عن الفراعنة أو المعابد أو الحضارة المصرية."
                )
            return (
                "I couldn't find relevant information in the knowledge base to answer your question.\n"
                "I specialise in Ancient Egyptian history — try asking about pharaohs, pyramids, "
                "gods, or ancient civilisation."
            )

        chunk_block = "\n\n---\n\n".join(
            f"[Chunk {i+1}]\n{c}" for i, c in enumerate(chunks)
        )
        prompt = _SYNTHESIS_PROMPT.format(chunks=chunk_block, question=question)
        try:
            return _llm(prompt, max_tokens=1024)
        except Exception as exc:
            log.error(f"Synthesis failed: {exc}")
            return "An error occurred while generating the answer. Please try again."

    # ── health ────────────────────────────────────────────────────────────────
    def health(self) -> dict:
        pool = _get_groq()
        s    = pool.status()
        return {
            "status":          "ok",
            "redis":           self.cache.health(),
            "pinecone_index":  PINECONE_INDEX,
            "llm_primary":     "groq (openai/gpt-oss-120b)",
            "groq_keys_total": s.get("keys_total", 0),
            "groq_keys_avail": s.get("keys_available", 0),
            "llm_fallback":    f"gemini ({GEMINI_MODEL})" if GEMINI_API_KEY else "none",
            "embed_model":     "intfloat/multilingual-e5-large (HF API)",
        }


# ── standalone smoke-test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    agent = KemetAgent()

    tests = [
        "hi",
        "who are you",
        "Who was Ramesses II and what were his major achievements?",
        "كيف بنوا الأهرامات؟",
    ]

    for q in tests:
        print(f"\nQ: {q}\n{'─'*60}")
        r = agent.ask(q)
        print(f"Answer:\n{r['answer']}")
        print(f"Chunks: {len(r['chunks_used'])} | Cached: {r['cached']} | Static: {r.get('static', False)} | {r['elapsed']}s")
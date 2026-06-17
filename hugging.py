"""
hugging.py — Groq Key Pool Manager
====================================
Multiple GROQ_API_KEY support with auto-rotation on rate limit.

.env format:
    GROQ_API_KEY=gsk_aaa...
    GROQ_API_KEY_2=gsk_bbb...
    GROQ_API_KEY_3=gsk_ccc...
    (add as many as you want)
"""

import os
import re
import time
import threading
import logging
from typing import Optional, List
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("kemet.llm")


# ═══════════════════════════════════════════════
# Read all GROQ keys from .env
# ═══════════════════════════════════════════════
def _collect_keys() -> List[str]:
    keys = []
    v = os.getenv("GROQ_API_KEY", "").strip()
    if v:
        keys.append(v)
    for i in range(2, 100):
        v = os.getenv(f"GROQ_API_KEY_{i}", "").strip()
        if v:
            keys.append(v)
        else:
            break
    return keys


# ═══════════════════════════════════════════════
# Per-key cooldown tracker
# ═══════════════════════════════════════════════
class _KeyState:
    def __init__(self, key: str):
        self.key           = key
        self._lock         = threading.Lock()
        self._exhausted_at: Optional[float] = None
        self._cooldown     = 0

    def mark_exhausted(self, cooldown_s: int = 60):
        with self._lock:
            self._exhausted_at = time.time()
            self._cooldown     = cooldown_s
        logger.warning(f"[Groq] Key {self.key[:12]}... rate-limited — cooldown {cooldown_s}s")

    def is_available(self) -> bool:
        with self._lock:
            if self._exhausted_at is None:
                return True
            if time.time() - self._exhausted_at >= self._cooldown:
                self._exhausted_at = None
                logger.info(f"[Groq] Key {self.key[:12]}... back online")
                return True
            return False

    def wait_seconds(self) -> float:
        with self._lock:
            if self._exhausted_at is None:
                return 0.0
            return max(0.0, self._cooldown - (time.time() - self._exhausted_at))


# ═══════════════════════════════════════════════
# Groq LLM with auto key rotation
# ═══════════════════════════════════════════════
class GroqKeyPool:
    """
    Wraps ChatGroq with multiple API keys.
    On 429 -> marks key exhausted, rotates to next, retries.
    Same .invoke() interface as ChatGroq — drop-in replacement.
    """

    def __init__(self, keys: List[str]):
        if not keys:
            raise RuntimeError("No GROQ_API_KEY found in .env")

        from langchain_groq import ChatGroq
        self._ChatGroq   = ChatGroq
        self._states     = [_KeyState(k) for k in keys]
        self._index      = 0
        self._lock       = threading.Lock()
        self._instances: dict = {}

        # validate first key on startup
        first = self._next_key()
        self._build(first)
        logger.info(f"[Groq] Ready — {len(keys)} key(s) in pool")

    def _next_key(self) -> Optional[str]:
        with self._lock:
            n = len(self._states)
            for _ in range(n):
                s = self._states[self._index % n]
                self._index = (self._index + 1) % n
                if s.is_available():
                    return s.key
        return None

    def _build(self, key: str):
        if key not in self._instances:
            self._instances[key] = self._ChatGroq(
                model        = "openai/gpt-oss-120b",
                temperature  = 0.2,
                max_tokens   = 2048,
                groq_api_key = key,
            )
        return self._instances[key]

    def _exhaust(self, key: str, cooldown_s: int):
        for s in self._states:
            if s.key == key:
                s.mark_exhausted(cooldown_s)

    def invoke(self, prompt, **kwargs):
        last_err = None
        for _ in range(len(self._states)):
            key = self._next_key()
            if key is None:
                wait = min(s.wait_seconds() for s in self._states)
                raise RuntimeError(
                    f"[Groq] All {len(self._states)} key(s) rate-limited. "
                    f"Retry in {wait:.0f}s"
                )
            try:
                return self._build(key).invoke(prompt, **kwargs)

            except Exception as e:
                err = str(e).lower()
                last_err = e

                if any(x in err for x in [
                    "429", "rate_limit", "rate limit",
                    "too many requests", "quota", "exceeded",
                    "tokens per", "requests per minute",
                ]):
                    cooldown = 60
                    m = re.search(r'(\d+)\s*second', err)
                    if m:
                        cooldown = int(m.group(1)) + 2
                    self._exhaust(key, cooldown)
                    continue

                elif any(x in err for x in ["401", "403", "invalid_api_key"]):
                    logger.error(f"[Groq] Key {key[:12]}... invalid")
                    self._exhaust(key, 86400)
                    continue

                else:
                    raise

        raise last_err or RuntimeError("[Groq] All keys failed")

    def status(self) -> dict:
        avail = sum(1 for s in self._states if s.is_available())
        return {
            "provider":       "Groq",
            "model":          "openai/gpt-oss-120b",
            "keys_total":     len(self._states),
            "keys_available": avail,
            "keys_exhausted": len(self._states) - avail,
        }


# ═══════════════════════════════════════════════
# Entry point — called by kemet_core.py
# ═══════════════════════════════════════════════
def get_free_llm() -> GroqKeyPool:
    keys = _collect_keys()
    if not keys:
        raise RuntimeError(
            "No GROQ_API_KEY in .env\n"
            "Add: GROQ_API_KEY=gsk_...\n"
            "Extra keys: GROQ_API_KEY_2=gsk_...\n"
            "Free keys: https://console.groq.com"
        )
    return GroqKeyPool(keys)


# ═══════════════════════════════════════════════
# Test
# ═══════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("\n🔑 Groq Key Pool Test\n")
    llm = get_free_llm()
    print(f"Status: {llm.status()}")
    resp   = llm.invoke("What is the Great Pyramid? One sentence.")
    answer = resp.content if hasattr(resp, "content") else str(resp)
    print(f"Answer: {answer}\n✅ Done!")
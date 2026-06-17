"""
retrieval_engine.py — Intelligent Retrieval Engine v2
=======================================================
Drop-in upgrade for RetrievalStrategy in kemet_core.py.

Improvements over the original:
  1. Multi-Query Expansion (MQE)     — 2-3 semantic variants per query
  2. HyDE                            — hypothetical answer embedding for dense recall
  3. Intent-Adaptive Top-K           — broader fetch for broad questions
  4. Topic-Aware Pinecone Filtering  — route query to the right metadata bucket
  5. Reciprocal Rank Fusion (RRF)    — fuse multiple ranked lists intelligently
  6. MMR Deduplication               — diversity-aware chunk selection (no copy-paste repeats)
  7. Score-Based Confidence Gate     — auto-escalate to web when RAG confidence is low
  8. Contextual Chunk Trimming       — surface the most relevant sentence window per chunk

Usage (replace in kemet_core.py):
    from retrieval_engine import IntelligentRetriever
    self.retrieval_strategy = IntelligentRetriever(
        pinecone_manager=self.pinecone_manager,
        web_search=self.web_search,
        llm=self.llm,           # GroqKeyPool  (for MQE + HyDE)
    )
    # Then call exactly as before:
    chunks, sources, meta = self.retrieval_strategy.retrieve(query_en, active_subject=subject)
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("kemet.retrieval")


# ──────────────────────────────────────────────────────────────────────────────
# Topic → TOPIC_KEYWORDS mapping (mirrors pinecone_smart_memory.py)
# ──────────────────────────────────────────────────────────────────────────────
_TOPIC_BUCKETS: Dict[str, List[str]] = {
    "pyramids": [
        "pyramid", "giza", "sphinx", "khufu", "khafre", "menkaure",
        "great pyramid", "construction", "limestone", "burial chamber",
    ],
    "pharaohs": [
        "pharaoh", "king", "ruler", "dynasty", "reign", "ramesses",
        "tutankhamun", "cleopatra", "akhenaten", "thutmose", "throne",
        "hatshepsut", "amenhotep", "narmer", "ptolemy",
    ],
    "religion": [
        "god", "goddess", "deity", "temple", "priest", "worship",
        "ra", "osiris", "isis", "horus", "anubis", "ritual",
        "afterlife", "sacred", "divine", "amun", "thoth", "bastet",
    ],
    "mummification": [
        "mummy", "mummification", "embalming", "natron", "canopic",
        "preservation", "sarcophagus", "tomb", "coffin", "viscera",
    ],
    "daily_life": [
        "daily life", "food", "clothing", "family", "agriculture",
        "farming", "trade", "bread", "beer", "linen", "jewelry",
        "cosmetics", "games", "children", "work",
    ],
    "writing": [
        "hieroglyph", "hieratic", "demotic", "scribe", "papyrus",
        "inscription", "rosetta stone", "alphabet", "symbol",
    ],
    "architecture": [
        "temple", "column", "monument", "karnak", "luxor", "abu simbel",
        "obelisk", "pylon", "hypostyle", "sanctuary", "building",
    ],
    "warfare": [
        "war", "battle", "army", "soldier", "weapon", "chariot",
        "military", "conquest", "fortress", "bow", "sword", "campaign",
    ],
}


def _detect_topic(query: str) -> Optional[str]:
    """
    Map a query to the best-matching TOPIC_KEYWORDS bucket.
    Returns None when no bucket scores above threshold.
    """
    q = query.lower()
    best_topic, best_score = None, 0
    for topic, keywords in _TOPIC_BUCKETS.items():
        score = sum(1 for kw in keywords if kw in q)
        if score > best_score:
            best_score, best_topic = score, topic
    # Require at least 1 keyword hit to trust the topic
    return best_topic if best_score >= 1 else None


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _chunk_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:16]


def _cosine(a: List[float], b: List[float]) -> float:
    """Dot-product cosine similarity (vectors assumed unit-length from Pinecone)."""
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb  = math.sqrt(sum(y * y for y in b)) or 1e-9
    return dot / (na * nb)


def _sentence_window(text: str, query: str, window: int = 4) -> str:
    """
    Contextual chunk trimming: return the `window`-sentence neighbourhood
    around the sentence most lexically similar to the query.
    Falls back to the full text when sentences are few or extraction fails.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(sentences) <= window + 2:
        return text                              # short chunk — keep as-is

    q_words = set(re.findall(r'\w+', query.lower()))
    best_idx, best_score = 0, -1
    for i, sent in enumerate(sentences):
        s_words = set(re.findall(r'\w+', sent.lower()))
        score   = len(q_words & s_words)
        if score > best_score:
            best_score, best_idx = score, i

    start = max(0, best_idx - window // 2)
    end   = min(len(sentences), start + window)
    return " ".join(sentences[start:end])


# ──────────────────────────────────────────────────────────────────────────────
# Reciprocal Rank Fusion
# ──────────────────────────────────────────────────────────────────────────────

def _rrf_fuse(
    ranked_lists: List[List[Tuple[str, str, float]]],
    k: int = 60,
) -> List[Tuple[str, str, float]]:
    """
    Reciprocal Rank Fusion over multiple ranked lists.

    Args:
        ranked_lists: Each list is [(text, source, score), ...] sorted best-first.
        k:            RRF constant (60 is standard).

    Returns:
        Single merged list sorted by RRF score, deduplicated by chunk hash.
    """
    rrf_scores: Dict[str, float] = defaultdict(float)
    chunk_data:  Dict[str, Tuple[str, str]] = {}

    for ranked in ranked_lists:
        for rank, (text, source, _score) in enumerate(ranked, start=1):
            h = _chunk_hash(text)
            rrf_scores[h]  += 1.0 / (k + rank)
            chunk_data[h]   = (text, source)

    # Sort by fused score descending
    ordered = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [(chunk_data[h][0], chunk_data[h][1], score) for h, score in ordered]


# ──────────────────────────────────────────────────────────────────────────────
# Maximal Marginal Relevance (MMR)
# ──────────────────────────────────────────────────────────────────────────────

def _mmr_select(
    candidates: List[Tuple[str, str, float]],
    embedder,
    query_embedding: List[float],
    top_n:   int   = 5,
    lambda_: float = 0.65,
) -> List[Tuple[str, str, float]]:
    """
    MMR: balance relevance vs. diversity.

    lambda_=1.0  → pure relevance (= normal ranking)
    lambda_=0.0  → pure diversity

    Falls back to plain top-N if embedder is unavailable.
    """
    if embedder is None or len(candidates) <= top_n:
        return candidates[:top_n]

    try:
        texts   = [c[0] for c in candidates]
        embeds  = [embedder.embed_query(t) for t in texts]

        selected_idx: List[int] = []
        remaining    = list(range(len(candidates)))

        while len(selected_idx) < top_n and remaining:
            scores = []
            for i in remaining:
                relevance = _cosine(embeds[i], query_embedding)
                if not selected_idx:
                    redundancy = 0.0
                else:
                    redundancy = max(
                        _cosine(embeds[i], embeds[j]) for j in selected_idx
                    )
                mmr_score = lambda_ * relevance - (1 - lambda_) * redundancy
                scores.append((mmr_score, i))

            _, best_i = max(scores)
            selected_idx.append(best_i)
            remaining.remove(best_i)

        return [candidates[i] for i in selected_idx]

    except Exception as e:
        logger.warning(f"[MMR] Fallback to top-N: {e}")
        return candidates[:top_n]


# ──────────────────────────────────────────────────────────────────────────────
# Main Engine
# ──────────────────────────────────────────────────────────────────────────────

class IntelligentRetriever:
    """
    Intelligent retrieval engine with:
      - Multi-Query Expansion  (MQE)
      - Hypothetical Document Embedding  (HyDE)
      - Topic-Aware Pinecone Metadata Filtering
      - Reciprocal Rank Fusion  (RRF)
      - MMR diversity-aware selection
      - Score-Based Confidence Gating → web fallback
      - Contextual chunk trimming
      - Entity-aware chunk scoring (retained from original)

    Drop-in replacement for RetrievalStrategy — same .retrieve() signature.
    """

    # Confidence threshold: if best RAG chunk score < this, escalate to web
    CONFIDENCE_THRESHOLD = 0.42       # cosine similarity (0–1)
    # Minimum chunks before we try MQE boost
    MQE_MIN_CHUNKS_FALLBACK = 2

    def __init__(
        self,
        pinecone_manager,
        web_search,
        llm=None,
        embedder=None,
    ):
        """
        Args:
            pinecone_manager: SmartPineconeManager instance.
            web_search:       EgyptWebSearch instance.
            llm:              GroqKeyPool — used for MQE and HyDE prompt calls.
                              If None, MQE/HyDE are skipped gracefully.
            embedder:         The embedding model (for MMR cosine calc).
                              If None, MMR falls back to plain top-N.
        """
        self.pinecone   = pinecone_manager
        self.web        = web_search
        self.llm        = llm
        self.embedder   = embedder      # injected later via set_embedder()

    def set_embedder(self, embedder):
        """Late-inject the embedding model (called after models are loaded)."""
        self.embedder = embedder

    # ── Public interface ─────────────────────────────────────────────────────

    def retrieve(
        self,
        query_en:       str,
        strategy:       str = "auto",
        active_subject: str = "",
    ) -> Tuple[List[str], List[str], Dict]:
        """
        Unified retrieval entry-point.  Same signature as the original
        RetrievalStrategy.retrieve() — fully backwards-compatible.

        Returns:
            (chunks, sources, metadata_dict)
        """
        if strategy == "rag":
            return self._rag_pipeline(query_en, active_subject)
        elif strategy == "web":
            return self._web_pipeline(query_en)
        elif strategy == "hybrid":
            return self._hybrid_pipeline(query_en, active_subject)
        else:
            return self._auto_pipeline(query_en, active_subject)

    # ── Strategy dispatchers ─────────────────────────────────────────────────

    def _auto_pipeline(
        self, query_en: str, active_subject: str
    ) -> Tuple[List[str], List[str], Dict]:
        """Smart auto: RAG first, escalate when confidence is low."""
        chunks, sources, meta = self._rag_pipeline(query_en, active_subject)

        confidence = meta.get("confidence", 0.0)
        too_few    = len(chunks) < 3
        low_conf   = confidence < self.CONFIDENCE_THRESHOLD

        if too_few or low_conf:
            reason = "too_few_chunks" if too_few else "low_confidence"
            logger.info(
                f"[Retrieval] Escalating to hybrid — {reason} "
                f"(chunks={len(chunks)}, conf={confidence:.3f})"
            )
            h_chunks, h_sources, h_meta = self._hybrid_pipeline(query_en, active_subject)
            h_meta["escalation_reason"] = reason
            return h_chunks, h_sources, h_meta

        meta["strategy"] = "rag"
        return chunks, sources, meta

    def _rag_pipeline(
        self, query_en: str, active_subject: str
    ) -> Tuple[List[str], List[str], Dict]:
        """Full intelligent RAG pipeline."""
        start = time.time()

        # 1. Detect topic for Pinecone metadata filter
        topic = _detect_topic(query_en)
        if not topic and active_subject:
            topic = _detect_topic(active_subject)

        # 2. Adaptive top-k based on query breadth
        top_k = self._adaptive_top_k(query_en)

        # 3. Build query variants: original + MQE + (optionally) HyDE
        variants = self._expand_queries(query_en, active_subject)

        # 4. Fetch ranked lists for each variant
        ranked_lists = []
        best_score   = 0.0

        for variant in variants:
            ranked = self._fetch_ranked(variant, top_k=top_k, topic=topic)
            if ranked:
                ranked_lists.append(ranked)
                best_score = max(best_score, ranked[0][2])

        if not ranked_lists:
            return [], [], {
                "time": time.time() - start,
                "count": 0,
                "strategy": "rag",
                "confidence": 0.0,
            }

        # 5. Reciprocal Rank Fusion across all variants
        fused = _rrf_fuse(ranked_lists)

        # 6. Entity-aware reranking (keeps original behaviour)
        if active_subject:
            fused = self._entity_rerank(fused, active_subject)

        # 7. MMR diversity selection
        query_emb = self._safe_embed(query_en)
        final     = _mmr_select(fused, self.embedder, query_emb, top_n=5)

        # 8. Contextual trimming + dedup
        chunks, sources = [], []
        seen: set = set()
        for text, src, _score in final:
            h = _chunk_hash(text)
            if h in seen:
                continue
            seen.add(h)
            trimmed = _sentence_window(text, query_en, window=5)
            chunks.append(trimmed)
            sources.append(src)

        elapsed = time.time() - start
        logger.info(
            f"[Retrieval] RAG done — {len(chunks)} chunks, "
            f"conf={best_score:.3f}, topic={topic}, "
            f"variants={len(variants)}, {elapsed:.2f}s"
        )
        return chunks, sources, {
            "time":       elapsed,
            "count":      len(chunks),
            "strategy":   "rag",
            "confidence": best_score,
            "topic":      topic,
            "variants":   len(variants),
        }

    def _web_pipeline(
        self, query_en: str
    ) -> Tuple[List[str], List[str], Dict]:
        """Web-only retrieval."""
        start = time.time()
        try:
            results = self.web.search(query_en, max_results=4)
            chunks, sources = [], []
            for item in results:
                content = item.get("content", "")
                source  = item.get("source", "web")
                if content and len(content) > 100:
                    trimmed = _sentence_window(content, query_en, window=5)
                    chunks.append(trimmed[:900])
                    sources.append(f"[Web] {source}")
            return chunks, sources, {
                "time":     time.time() - start,
                "count":    len(chunks),
                "strategy": "web",
            }
        except Exception as e:
            logger.error(f"[Retrieval] Web pipeline error: {e}")
            return [], [], {"time": 0, "count": 0, "error": str(e)}

    def _hybrid_pipeline(
        self, query_en: str, active_subject: str
    ) -> Tuple[List[str], List[str], Dict]:
        """Fused RAG + web using RRF."""
        start = time.time()
        try:
            rag_c, rag_s, _ = self._rag_pipeline(query_en, active_subject)
            web_c, web_s, _ = self._web_pipeline(query_en)

            # Build ranked lists (score web slightly lower)
            rag_ranked = [(c, s, 0.9) for c, s in zip(rag_c, rag_s)]
            web_ranked = [(c, s, 0.7) for c, s in zip(web_c, web_s)]
            fused      = _rrf_fuse([rag_ranked, web_ranked])

            # Final dedup
            chunks, sources = [], []
            seen: set = set()
            for text, src, _ in fused[:6]:
                h = _chunk_hash(text)
                if h not in seen:
                    seen.add(h)
                    chunks.append(text)
                    sources.append(src)

            has_web = any("[Web]" in s for s in sources)
            return chunks, sources, {
                "time":     time.time() - start,
                "count":    len(chunks),
                "strategy": "hybrid",
                "has_web":  has_web,
            }
        except Exception as e:
            logger.error(f"[Retrieval] Hybrid pipeline error: {e}")
            return [], [], {"time": 0, "count": 0, "error": str(e)}

    # ── Core helpers ─────────────────────────────────────────────────────────

    def _fetch_ranked(
        self, query: str, top_k: int, topic: Optional[str]
    ) -> List[Tuple[str, str, float]]:
        """
        Query Pinecone (with optional topic filter) and return a ranked list.
        """
        try:
            results = self.pinecone.query(
                query,
                top_k=top_k,
                topic_filter=topic,
            )
            ranked = []
            seen:  set = set()
            for match in results:
                text   = match.metadata.get("text", "").strip()
                source = match.metadata.get("filename", "unknown")
                score  = float(match.score)
                if text and len(text) > 80:
                    h = _chunk_hash(text)
                    if h not in seen:
                        seen.add(h)
                        ranked.append((text, source, score))
            return ranked          # already sorted by Pinecone score
        except Exception as e:
            logger.warning(f"[Retrieval] Pinecone fetch error: {e}")
            return []

    def _adaptive_top_k(self, query: str) -> int:
        """
        Broader questions need more candidates; specific entity lookups need fewer.
        """
        q = query.lower()
        broad_signals  = ["what", "how", "explain", "describe", "overview",
                          "tell me", "history of", "all", "list"]
        entity_signals = ["who", "who was", "who is", "when did", "when was",
                          "where is", "where was"]

        is_broad  = any(s in q for s in broad_signals)
        is_entity = any(s in q for s in entity_signals)

        if is_broad and not is_entity:
            return 15   # fetch more candidates for broad questions
        elif is_entity:
            return 8    # focused — fewer but more precise
        else:
            return 10   # default

    # ── Multi-Query Expansion ────────────────────────────────────────────────

    def _expand_queries(
        self, query: str, active_subject: str = ""
    ) -> List[str]:
        """
        Build 2-3 query variants:
          variant[0] = original query        (always included)
          variant[1] = MQE rephrasing        (LLM, when available)
          variant[2] = HyDE document         (LLM, for specific entity queries)

        Gracefully degrades to [original] when LLM is unavailable.
        """
        variants = [query]

        if self.llm is None:
            return variants

        # MQE — rephrase for better vocabulary coverage
        mqe = self._run_mqe(query)
        if mqe and mqe.lower().strip() != query.lower().strip():
            variants.append(mqe)

        # HyDE — only for specific entity / factual questions
        if self._should_use_hyde(query):
            hyde_doc = self._run_hyde(query, active_subject)
            if hyde_doc:
                variants.append(hyde_doc)

        return variants

    def _run_mqe(self, query: str) -> Optional[str]:
        """
        Ask the LLM for one alternative phrasing of the query.
        Returns None on failure.
        """
        prompt = (
            "You are a query expansion assistant for an Ancient Egypt knowledge base.\n"
            "Rephrase the following search query in ONE alternative way that would help "
            "find relevant information in a vector database. "
            "Output ONLY the rephrased query — no explanation, no quotes.\n\n"
            f"Original query: {query}\n\nRephrased query:"
        )
        try:
            resp   = self.llm.invoke(prompt, max_tokens=80, temperature=0.3)
            result = resp.content if hasattr(resp, "content") else str(resp)
            result = result.strip().split("\n")[0].strip()
            if result and len(result) > 5:
                return result
        except Exception as e:
            logger.debug(f"[MQE] Skipped: {e}")
        return None

    def _should_use_hyde(self, query: str) -> bool:
        """
        HyDE helps most for specific factual/entity questions.
        Skip for vague, comparative, or meta questions.
        """
        q = query.lower()
        hyde_triggers  = [
            "who was", "who is", "what was", "what is",
            "when did", "where was", "tell me about", "describe",
            "built", "reign", "discovered",
        ]
        return any(t in q for t in hyde_triggers)

    def _run_hyde(self, query: str, subject: str = "") -> Optional[str]:
        """
        Generate a short hypothetical passage that would answer the question.
        This passage is used as a second query vector (HyDE technique).
        """
        subject_hint = f" The question is about {subject}." if subject else ""
        prompt = (
            "You are a concise Ancient Egypt historian.{subject_hint}\n"
            "Write a 2-3 sentence factual passage from an encyclopedia that directly "
            "answers the following question. Write ONLY the passage, nothing else.\n\n"
            f"Question: {query}\n\nPassage:"
        ).format(subject_hint=subject_hint)
        try:
            resp   = self.llm.invoke(prompt, max_tokens=120, temperature=0.1)
            result = resp.content if hasattr(resp, "content") else str(resp)
            result = result.strip()
            if result and len(result) > 30:
                return result
        except Exception as e:
            logger.debug(f"[HyDE] Skipped: {e}")
        return None

    # ── Entity-aware reranking (from original RetrievalStrategy) ─────────────

    def _entity_score(self, text: str, subject: str) -> int:
        text_lower  = text.lower()
        subj_lower  = subject.lower()
        subj_words  = [w for w in subj_lower.split() if len(w) > 2]
        score       = text_lower.count(subj_lower) * 3
        for word in subj_words:
            score += text_lower.count(word)
        return score

    def _entity_rerank(
        self,
        fused: List[Tuple[str, str, float]],
        subject: str,
    ) -> List[Tuple[str, str, float]]:
        """Boost chunks that mention the active subject."""
        scored = [
            (self._entity_score(text, subject), text, src, rrf)
            for text, src, rrf in fused
        ]
        scored.sort(key=lambda x: x[0], reverse=True)

        top_score = scored[0][0] if scored else 0
        if top_score > 0:
            hits  = [(t, s, r) for sc, t, s, r in scored if sc > 0]
            miss  = [(t, s, r) for sc, t, s, r in scored if sc == 0]
            if len(hits) >= 3:
                return hits[:5]
            return hits + miss
        return [(t, s, r) for _, t, s, r in scored]

    # ── Embedding helper ─────────────────────────────────────────────────────

    def _safe_embed(self, text: str) -> List[float]:
        """Embed text, returning empty list on failure (MMR will degrade gracefully)."""
        if self.embedder is None:
            return []
        try:
            return self.embedder.embed_query(text)
        except Exception as e:
            logger.debug(f"[Embed] Failed: {e}")
            return []


# ──────────────────────────────────────────────────────────────────────────────
# Integration patch for KemetVision._ensure_models()
# ──────────────────────────────────────────────────────────────────────────────
# Replace the HeritageAgent block in _ensure_models() with:
#
#     if self.retrieval_strategy is None:
#         from retrieval_engine import IntelligentRetriever
#         self.retrieval_strategy = IntelligentRetriever(
#             pinecone_manager = self.pinecone_manager,
#             web_search       = self.web_search,
#             llm              = self.llm,
#             embedder         = getattr(self.pinecone_manager, 'embedding_model', None),
#         )
#         logger.info("IntelligentRetriever initialized ✅")
#
# And in _rag_pipeline / ask(), replace:
#     chunks, sources, meta = self.retrieval_strategy.retrieve(query_en, active_subject=subject)
# ──────────────────────────────────────────────────────────────────────────────

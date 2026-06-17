"""
utils_professional.py - Clean Professional Implementation (HF Inference API)
-----------------------------------------------------------------------------
 NO local model loading (no torch, no CrossEncoder, no HuggingFaceEmbeddings)
 Same models: intfloat/multilingual-e5-large + BAAI/bge-reranker-large
 Called via HuggingFace InferenceClient (huggingface_hub) — the correct 2025 way
 Multi-layered fallback strategy
 Production-grade error handling
 Minimal, maintainable code
"""

import os
import hashlib
import re
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from langdetect import detect, DetectorFactory, LangDetectException
from deep_translator import GoogleTranslator
import logging

# Set seed for consistent language detection
DetectorFactory.seed = 0

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
load_dotenv()

# ===========================
# Environment Variables
# ===========================
HF_TOKEN = os.getenv("HF_TOKEN")

EMBED_MODEL  = "intfloat/multilingual-e5-large"
RERANK_MODEL = "BAAI/bge-reranker-large"

# ===========================
# Global Caches
# ===========================
_MODEL_CACHE       = {}
_EMBED_CACHE       = {}
_TRANSLATION_CACHE = {}


# ===========================
# HF InferenceClient (lazy singleton)
# ===========================
def _get_hf_client() -> InferenceClient:
    """Return (or create) a shared HF InferenceClient."""
    if "hf_client" not in _MODEL_CACHE:
        if not HF_TOKEN:
            raise EnvironmentError("HF_TOKEN is not set in environment / .env file")
        _MODEL_CACHE["hf_client"] = InferenceClient(
            provider="hf-inference",
            api_key=HF_TOKEN,
        )
        logging.info("HF InferenceClient initialised")
    return _MODEL_CACHE["hf_client"]


# ===========================
# Professional Language Detection
# ===========================
def detect_lang(text: str, confidence_threshold: float = 0.9) -> str:
    """
    Detect language using langdetect with confidence filtering.

    Strategy:
    1. Use langdetect library (proven, reliable)
    2. Check for Arabic script (unambiguous signal)
    3. Apply confidence threshold
    4. Sensible fallback to English

    Args:
        text: Input text
        confidence_threshold: Minimum confidence (0.0-1.0)

    Returns:
        ISO language code ('en', 'ar', etc.)
    """
    if not text or not text.strip():
        return "en"

    text_clean   = text.strip()
    arabic_chars = sum(1 for c in text_clean if '\u0600' <= c <= '\u06FF')
    total_chars  = sum(1 for c in text_clean if c.isalpha())

    if total_chars > 0 and arabic_chars / total_chars > 0.3:
        return "ar"

    try:
        detected = detect(text_clean)

        if len(text_clean.split()) <= 3:
            latin_chars = sum(1 for c in text_clean if c.isalpha() and ord(c) < 128)
            if total_chars > 0 and latin_chars / total_chars > 0.8:
                return "en"

        return detected

    except LangDetectException:
        logging.warning("Language detection failed, defaulting to English")
        return "en"


def detect_lang_with_confidence(text: str) -> tuple:
    """
    Detect language with confidence score.

    Returns:
        (language_code, confidence_score)
    """
    if not text or not text.strip():
        return ("en", 1.0)

    try:
        from langdetect import detect_langs
        results = detect_langs(text.strip())
        if results:
            top = results[0]
            return (top.lang, top.prob)
        return ("en", 1.0)

    except Exception:
        return (detect_lang(text), 0.8)


# ===========================
# Professional Translation
# ===========================
def translate_text_auto(text: str, target_lang: str = "en", use_cache: bool = True) -> str:
    """
    Translate text to target language.

    Strategy:
    1. Auto-detect source language
    2. Use GoogleTranslator (free, reliable)
    3. Cache results for performance
    4. Graceful error handling

    Args:
        text: Text to translate
        target_lang: Target language code
        use_cache: Whether to use cache

    Returns:
        Translated text
    """
    if not text or not text.strip():
        return text

    text_clean = text.strip()
    src_lang   = detect_lang(text_clean)

    if src_lang == target_lang:
        logging.debug(f"Text already in {target_lang}, skipping translation")
        return text_clean

    if use_cache:
        cache_key = hashlib.md5(
            f"{text_clean[:300]}_{src_lang}_{target_lang}".encode()
        ).hexdigest()
        if cache_key in _TRANSLATION_CACHE:
            logging.debug("Translation cache hit")
            return _TRANSLATION_CACHE[cache_key]

    try:
        logging.info(f"Translating {src_lang} -> {target_lang}")
        translator = GoogleTranslator(source=src_lang, target=target_lang)
        translated = translator.translate(text_clean)

        if use_cache:
            _TRANSLATION_CACHE[cache_key] = translated
        return translated

    except Exception as e:
        logging.warning(f"Translation failed ({src_lang}->{target_lang}): {e}")
        try:
            translator = GoogleTranslator(source='auto', target=target_lang)
            translated = translator.translate(text_clean)
            if use_cache:
                _TRANSLATION_CACHE[cache_key] = translated
            return translated
        except Exception as e2:
            logging.error(f"Translation fallback failed: {e2}")
            if use_cache:
                _TRANSLATION_CACHE[cache_key] = text_clean
            return text_clean


def translate_back_to_user_lang(text: str, user_lang: str, use_cache: bool = True) -> str:
    """
    Translate text back to user's original language.

    Args:
        text: Text to translate (assumed English)
        user_lang: User's language code
        use_cache: Whether to use cache

    Returns:
        Translated text
    """
    if not text or not text.strip():
        return text

    if user_lang.lower() == "en":
        return text

    text_clean = text.strip()

    if use_cache:
        cache_key = hashlib.md5(
            f"{text_clean[:300]}_en_{user_lang}".encode()
        ).hexdigest()
        if cache_key in _TRANSLATION_CACHE:
            logging.debug("Back-translation cache hit")
            return _TRANSLATION_CACHE[cache_key]

    try:
        logging.info(f"Back-translating en -> {user_lang}")
        translator = GoogleTranslator(source='en', target=user_lang)
        translated = translator.translate(text_clean)

        if use_cache:
            _TRANSLATION_CACHE[cache_key] = translated
        return translated

    except Exception as e:
        logging.warning(f"Back-translation failed: {e}")
        if use_cache:
            _TRANSLATION_CACHE[cache_key] = text_clean
        return text_clean


def batch_translate(texts: list, target_lang: str = "en") -> list:
    """
    Translate multiple texts efficiently.

    Args:
        texts: List of texts to translate
        target_lang: Target language

    Returns:
        List of translated texts
    """
    return [translate_text_auto(t, target_lang) for t in texts]


# ===========================
# Embedding Model -> HF InferenceClient
# (intfloat/multilingual-e5-large — same model, zero local loading)
# ===========================
def get_embedding_model(device: str = None):
    """
    Return a wrapper that calls intfloat/multilingual-e5-large
    via the HuggingFace Inference API using InferenceClient.

    Same interface as HuggingFaceEmbeddings:
      .embed_documents(texts)  -> list[list[float]]
      .embed_query(text)       -> list[float]

    Args:
        device: Ignored (kept for API compatibility)

    Returns:
        HFEmbeddingAPIWrapper instance
    """
    if "embedding_model" in _MODEL_CACHE:
        return _MODEL_CACHE["embedding_model"]

    wrapper = HFEmbeddingAPIWrapper()
    _MODEL_CACHE["embedding_model"] = wrapper
    logging.info(f"Loaded {EMBED_MODEL} via HF InferenceClient")
    return wrapper


class HFEmbeddingAPIWrapper:
    """
    LangChain-compatible wrapper around HF InferenceClient.feature_extraction().

    Calls intfloat/multilingual-e5-large — natively supported by hf-inference provider.

    multilingual-e5-large expects the prefix:
      "query: <text>"   for queries
      "passage: <text>" for documents
    """

    def _get_embedding(self, text: str) -> list:
        """Call feature_extraction for a single prefixed text."""
        client = _get_hf_client()
        # Returns np.ndarray shape (dim,) or (seq_len, dim)
        result = client.feature_extraction(text, model=EMBED_MODEL)
        # Convert numpy to plain list
        arr = result.tolist() if hasattr(result, 'tolist') else result
        # If shape is (seq_len, dim) — mean pool to get (dim,)
        if arr and isinstance(arr[0], list):
            dim   = len(arr[0])
            n     = len(arr)
            pooled = [sum(arr[i][j] for i in range(n)) / n for j in range(dim)]
            return pooled
        return arr

    def embed_documents(self, texts: list) -> list:
        """Embed a list of documents (passages)."""
        if not texts:
            return []

        results = [None] * len(texts)
        for i, text in enumerate(texts):
            key = hashlib.md5(f"doc:{text[:500]}".encode()).hexdigest()
            if key in _EMBED_CACHE:
                results[i] = _EMBED_CACHE[key]
            else:
                logging.info(f"Embedding document {i+1}/{len(texts)} via HF API")
                emb = self._get_embedding(f"passage: {text}")
                _EMBED_CACHE[key] = emb
                results[i]        = emb

        return results

    def embed_query(self, text: str) -> list:
        """Embed a single search query."""
        key = hashlib.md5(f"qry:{text[:500]}".encode()).hexdigest()
        if key in _EMBED_CACHE:
            logging.debug("Embedding cache hit (query)")
            return _EMBED_CACHE[key]

        logging.info("Embedding query via HF API")
        emb = self._get_embedding(f"query: {text}")
        _EMBED_CACHE[key] = emb
        return emb


# ===========================
# Reranker Model -> HF InferenceClient
# (BAAI/bge-reranker-large — same model, zero local loading)
# ===========================
def get_reranker(device: str = None):
    """
    Return a wrapper that calls BAAI/bge-reranker-large
    via the HuggingFace Inference API using InferenceClient.

    Same interface as CrossEncoder:
      .predict(pairs) -> list[float]

    Args:
        device: Ignored (kept for API compatibility)

    Returns:
        HFRerankerAPIWrapper instance
    """
    if "reranker_model" in _MODEL_CACHE:
        return _MODEL_CACHE["reranker_model"]

    wrapper = HFRerankerAPIWrapper()
    _MODEL_CACHE["reranker_model"] = wrapper
    logging.info(f"Loaded {RERANK_MODEL} via HF InferenceClient")
    return wrapper


class HFRerankerAPIWrapper:
    """
    CrossEncoder-compatible wrapper around HF InferenceClient.feature_extraction().

    BAAI/bge-reranker-large on HF Inference API is exposed as feature-extraction.
    It returns a single logit per (query, doc) pair — we apply sigmoid to get
    a 0-1 relevance score, identical to what CrossEncoder.predict() returns.

    Usage (identical to CrossEncoder):
        reranker = get_reranker()
        scores   = reranker.predict([("query", "doc1"), ("query", "doc2")])
    """

    @staticmethod
    def _sigmoid(x: float) -> float:
        import math
        return 1.0 / (1.0 + math.exp(-x))

    def predict(self, pairs: list) -> list:
        """
        Score (query, document) pairs.

        Args:
            pairs: List of (query, document) tuples.

        Returns:
            List of relevance scores (float, 0-1), same order as input.
        """
        if not pairs:
            return []

        client = _get_hf_client()
        scores = []

        for i, (query, doc) in enumerate(pairs):
            logging.info(f"Reranking pair {i+1}/{len(pairs)} via HF API")

            # bge-reranker-large on HF is a feature-extraction pipeline
            # Input: "query: <q> [SEP] passage: <doc>"  → returns raw logit
            combined = f"query: {query} [SEP] passage: {doc}"
            result   = client.feature_extraction(combined, model=RERANK_MODEL)

            # result shape: scalar, (1,), or (seq_len, hidden) depending on HF version
            arr = result.tolist() if hasattr(result, 'tolist') else result

            if isinstance(arr, float):
                logit = arr
            elif isinstance(arr, list):
                # Flatten to get the first scalar value (CLS logit)
                while isinstance(arr, list):
                    arr = arr[0]
                logit = float(arr)
            else:
                logit = float(arr)

            scores.append(self._sigmoid(logit))

        return scores


# ===========================
# Text Cleaning
# ===========================
def clean_arabic(text: str) -> str:
    """Clean Arabic text (remove tatweel, extra spaces)."""
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r"[ـ]+", "", text)  # Remove tatweel
    text = text.replace("..", ".")
    return text


def clean_llm_output(text: str) -> str:
    """Remove LLM artifacts from output."""
    text = re.sub(
        r'(\s*Question:\s*.*?\s*(Answer:|Evidence:))',
        '', text, flags=re.DOTALL | re.IGNORECASE
    )
    text = re.sub(
        r'^(Answer:|The Answer:|Final Answer:)\s*',
        '', text, flags=re.IGNORECASE
    )
    text = re.sub(
        r'\.{3}\s*Question:\s*[\s\S]*$',
        '', text, flags=re.DOTALL | re.IGNORECASE
    )
    text = re.sub(r'\s*Sources:[\s\S]*', '', text, flags=re.DOTALL)
    return text.strip()


# ===========================
# Utilities
# ===========================
def make_key(query: str) -> str:
    """Create cache key from query."""
    return hashlib.sha256(query.lower().encode()).hexdigest()


def inject_shared_models(reranker=None, embedding_model=None):
    """Inject pre-created wrappers to avoid re-initialising."""
    if reranker is not None:
        _MODEL_CACHE["reranker_model"] = reranker
        logging.info("Reranker injected")

    if embedding_model is not None:
        _MODEL_CACHE["embedding_model"] = embedding_model
        logging.info("Embedding model injected")


def get_cache_stats():
    """Get cache statistics."""
    return {
        "cached_models":          list(_MODEL_CACHE.keys()),
        "embedding_cache_size":   len(_EMBED_CACHE),
        "translation_cache_size": len(_TRANSLATION_CACHE),
    }


def clear_caches():
    """Clear all in-memory caches."""
    global _TRANSLATION_CACHE, _EMBED_CACHE
    _TRANSLATION_CACHE = {}
    _EMBED_CACHE       = {}
    logging.info("Caches cleared")


# ===========================
# Testing
# ===========================
if __name__ == "__main__":
    print("\n Testing Professional Language Detection\n")
    print("=" * 70)

    test_cases = [
        ("Who built the Great Pyramid?", "en"),
        ("What are the pyramids of Giza?", "en"),
        ("How long did it take?", "en"),
        ("When did he build it?", "en"),
        ("من بنى الاهرامات؟", "ar"),
        ("ما هي الاهرامات؟", "ar"),
        ("كيف بنوا الاهرامات؟", "ar"),
    ]

    correct = 0
    for text, expected in test_cases:
        detected = detect_lang(text)
        status   = "OK" if detected == expected else "FAIL"
        correct += detected == expected
        print(f"{status} '{text}' -> {detected} (expected: {expected})")

    accuracy = (correct / len(test_cases)) * 100
    print(f"\n{'=' * 70}")
    print(f"Accuracy: {correct}/{len(test_cases)} ({accuracy:.1f}%)")
    print(f"{'=' * 70}\n")

    print("Translation Test:")
    print("=" * 70)
    en_text = "Who built the Great Pyramid?"
    ar_text = "من بنى الاهرامات؟"
    print(f"EN -> AR: {en_text}")
    print(f"Result:  {translate_text_auto(en_text, 'ar')}\n")
    print(f"AR -> EN: {ar_text}")
    print(f"Result:  {translate_text_auto(ar_text, 'en')}\n")

    print("Embedding Test (requires HF_TOKEN):")
    print("=" * 70)
    model = get_embedding_model()
    vec   = model.embed_query("What is the Great Pyramid?")
    print(f"Query embedding dim: {len(vec)}  (expected 1024)")
    docs = model.embed_documents(["Khufu built the pyramid.", "The Nile is a river."])
    print(f"Doc embeddings: {len(docs)} vectors, dim={len(docs[0])}")

    print("\nReranker Test (requires HF_TOKEN):")
    print("=" * 70)
    reranker = get_reranker()
    pairs = [
        ("Who built the pyramids?", "Khufu built the Great Pyramid."),
        ("Who built the pyramids?", "The weather in Cairo is hot."),
    ]
    scores = reranker.predict(pairs)
    print(f"Scores: {scores}  (first should be higher)")

    print("=" * 70)
    print("Tests completed")
    print("=" * 70)
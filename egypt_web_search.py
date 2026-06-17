"""
egypt_web_search.py - Web Search Tool
--------------------------------------
Fallback web search for queries not answered by local RAG.
Uses Tavily (FREE tier: 1000 searches/month) with caching.
"""

import os
import json
import hashlib
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from colorama import Fore, Style, init
import logging

init(autoreset=True)
load_dotenv()
logging.basicConfig(level=logging.INFO)

# Config
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
CACHE_FILE = "web_search_cache.json"

# Trusted domains for Egyptian heritage
# Tier 1 — Official Egyptian sources (highest priority)
TIER1_DOMAINS = [
    "egymonuments.gov.eg",          # وزارة الآثار المصرية
    "egyptianmuseumcairo.eg",       # المتحف المصري بالقاهرة
    "nmec.gov.eg",                  # المتحف القومي للحضارة
    "mota.gov.eg",                  # وزارة السياحة والآثار
    "grandegyptianmuseum.org",      # المتحف المصري الكبير
    "egypt.travel",                 # هيئة تنشيط السياحة المصرية
]

# Tier 2 — International academic & museum sources
TIER2_DOMAINS = [
    "metmuseum.org",                # Metropolitan Museum of Art
    "britishmuseum.org",            # British Museum
    "worldhistory.org",             # World History Encyclopedia (أكاديمي)
    "ancientegyptonline.co.uk",     # متخصص 100% في مصر القديمة
    "arce.org",                     # American Research Center in Egypt
    "ucl.ac.uk",                    # UCL Digital Egypt for Universities
    "britannica.com",               # Encyclopaedia Britannica
]

# Tier 3 — Quality general sources
TIER3_DOMAINS = [
    "en.wikipedia.org",             # ويكيبيديا — محتوى ضخم وموثق
    "nationalgeographic.com",       # ناشيونال جيوجرافيك
    "smithsonianmag.com",           # Smithsonian Magazine
    "ancient.eu",                   # محول لـ worldhistory — legacy support
]

# Combined list for Tavily include_domains
TRUSTED_DOMAINS = TIER1_DOMAINS + TIER2_DOMAINS + TIER3_DOMAINS


class EgyptWebSearch:
    """
    Web search tool for Egyptian heritage queries.
    Uses Tavily with caching and reranking.
    """
    
    def __init__(self, reranker=None):
        """
        Initialize web search tool.
        
        Args:
            reranker: Optional CrossEncoder for result reranking
        """
        self.cache = self._load_cache()
        self.reranker = reranker
        
        # Initialize Tavily (if API key available)
        if TAVILY_API_KEY:
            try:
                self.search_tool = TavilySearch(
                    api_key=TAVILY_API_KEY,
                    k=5,
                    search_depth="advanced",
                    include_domains=TRUSTED_DOMAINS
                )
                logging.info(" Web search initialized (Tavily)")
            except Exception as e:
                logging.warning(f"Tavily init failed: {e}")
                self.search_tool = None
        else:
            logging.warning("  No Tavily key - web search disabled")
            self.search_tool = None
    
    def _load_cache(self):
        """Load search cache from disk."""
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_cache(self):
        """Save cache to disk."""
        try:
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.warning(f"Failed to save cache: {e}")
    
    def _make_key(self, query: str) -> str:
        """Generate cache key."""
        return hashlib.sha256(query.lower().encode()).hexdigest()
    
    def search(self, query: str, max_results: int = 5):
        """
        Search the web for Egyptian heritage information.
        
        Args:
            query: Search query
            max_results: Max number of results
            
        Returns:
            list: Search results with 'source' and 'content' keys
        """
        if not self.search_tool:
            logging.warning("Web search not available")
            return []
        
        # Check cache
        cache_key = self._make_key(query)
        if cache_key in self.cache:
            logging.info(f"{Fore.YELLOW}[WEB CACHE HIT]{Style.RESET_ALL}")
            return self.cache[cache_key][:max_results]
        
        logging.info(f"{Fore.CYAN} Web search: {query}{Style.RESET_ALL}")
        
        try:
            # Execute search
            results = self.search_tool.invoke(query)
            results = self._normalize_results(results)
            
            # Filter trusted sources
            valid_results = self._filter_trusted(results)
            
            if not valid_results:
                logging.warning("No trusted sources found")
                return []
            
            # Rerank if reranker available
            if self.reranker:
                valid_results = self._rerank_results(query, valid_results)
            
            # Cache results
            self.cache[cache_key] = valid_results
            self._save_cache()
            
            logging.info(f"{Fore.GREEN} Found {len(valid_results)} results{Style.RESET_ALL}")
            return valid_results[:max_results]
            
        except Exception as e:
            logging.error(f"Web search failed: {e}")
            return []
    
    def _normalize_results(self, results):
        """Normalize Tavily response format."""
        if isinstance(results, dict) and "results" in results:
            return results["results"]
        if isinstance(results, str):
            return [{"url": "N/A", "content": results}]
        if isinstance(results, list):
            return results
        return []
    
    def _filter_trusted(self, results):
        """Keep only results from trusted domains, with tier-based scoring."""
        valid = []
        for res in results:
            url     = res.get("url", "")
            content = res.get("content", "").strip()

            if len(content) < 100:
                continue

            # Tier scoring: higher = better source
            tier = 0
            if any(d in url for d in TIER1_DOMAINS):   tier = 3
            elif any(d in url for d in TIER2_DOMAINS): tier = 2
            elif any(d in url for d in TIER3_DOMAINS): tier = 1

            if tier > 0:
                valid.append({
                    "source": url,
                    "content": content,
                    "tier": tier,
                })

        # Sort by tier descending — official Egyptian sources first
        valid.sort(key=lambda x: x["tier"], reverse=True)
        return valid
    
    def _rerank_results(self, query: str, results):
        """Rerank results by relevance using CrossEncoder."""
        if not self.reranker or not results:
            return results
        
        try:
            pairs = [(query, r["content"]) for r in results]
            scores = self.reranker.predict(pairs)
            ranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
            return [r for r, _ in ranked]
        except Exception as e:
            logging.warning(f"Reranking failed: {e}")
            return results


# ===========================
# Factory Function
# ===========================
def get_egypt_web_search_tool(reranker=None):
    """
    Get web search tool instance.
    
    Args:
        reranker: Optional CrossEncoder for reranking
    """
    return EgyptWebSearch(reranker=reranker)


# ===========================
# Test
# ===========================
if __name__ == "__main__":
    print("\n Testing Web Search\n")
    
    searcher = EgyptWebSearch()
    query = "latest archaeological discoveries in Egypt"
    
    results = searcher.search(query)
    
    print(f"\n Results for: '{query}'\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. Source: {r['source']}")
        print(f"   Content: {r['content'][:200]}...\n")
"""
pinecone_smart_memory.py - FIXED: Uses metadata filtering instead of 100+ namespaces
---------------------------------------------------------------------------------------
 Changes:
  ONE namespace per book (or single namespace for all)
  Topics stored in metadata, not namespaces
  Uses Pinecone metadata filters for queries
  No more 100 namespace limit errors!
  NEW: Resume upload - only uploads new/unprocessed files!

Example Structure:
 Namespace: "ancient-egypt-encyclopedia"
   ├── vector_1 (metadata: {topic: "pyramids", book: "ancient-egypt-encyclopedia"})
   ├── vector_2 (metadata: {topic: "pharaohs", book: "ancient-egypt-encyclopedia"})
   └── vector_3 (metadata: {topic: "religion", book: "ancient-egypt-encyclopedia"})
"""

import os
import json
import hashlib
import re
import logging
from typing import List, Dict, Optional, Tuple, Set
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict

from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import torch
import numpy as np

from utils import get_embedding_model, get_reranker

# ===========================
# Configuration
# ===========================
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
DATA_PATH = "data/"
INDEX_NAME = "clash-code" \
""
DIMENSION = 1024
METRIC = "cosine"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
BATCH_SIZE = 50
CACHE_FILE = "pinecone_smart_cache.json"
UPLOAD_TRACKER_FILE = "pinecone_upload_tracker.json"  #  NEW: Track uploaded files

# NEW: Namespace strategy
# Options: "single" (one namespace for all), "per_book" (one per book)
NAMESPACE_STRATEGY = "per_book"  # Change to "single" if you want one namespace total

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# ===========================
# Topic Keywords (Egyptian History)
# ===========================
TOPIC_KEYWORDS = {
    'pyramids': [
        'pyramid', 'giza', 'sphinx', 'khufu', 'khafre', 'menkaure',
        'construction', 'great pyramid', 'stone blocks', 'limestone',
        'burial chamber', 'pyramid complex', 'causeway'
    ],
    'pharaohs': [
        'pharaoh', 'king', 'ruler', 'dynasty', 'reign', 'succession',
        'ramesses', 'tutankhamun', 'cleopatra', 'akhenaten', 'thutmose',
        'royal', 'throne', 'coronation', 'cartouche'
    ],
    'religion': [
        'god', 'goddess', 'deity', 'temple', 'priest', 'worship',
        'ra', 'osiris', 'isis', 'horus', 'anubis', 'ritual',
        'offering', 'prayer', 'sacred', 'divine', 'afterlife'
    ],
    'mummification': [
        'mummy', 'mummification', 'embalming', 'natron', 'canopic',
        'preservation', 'burial', 'sarcophagus', 'tomb', 'coffin',
        'bandages', 'wrapping', 'viscera', 'death ritual'
    ],
    'daily_life': [
        'daily life', 'food', 'clothing', 'house', 'family', 'children',
        'work', 'agriculture', 'farming', 'trade', 'market', 'bread',
        'beer', 'linen', 'jewelry', 'cosmetics', 'games'
    ],
    'writing': [
        'hieroglyph', 'hieratic', 'demotic', 'scribe', 'papyrus',
        'writing', 'text', 'inscription', 'cartouche', 'rosetta stone',
        'alphabet', 'symbol', 'reading', 'literature'
    ],
    'architecture': [
        'temple', 'building', 'column', 'architecture', 'monument',
        'karnak', 'luxor', 'abu simbel', 'construction', 'design',
        'obelisk', 'pylon', 'hypostyle', 'sanctuary'
    ],
    'warfare': [
        'war', 'battle', 'army', 'soldier', 'weapon', 'chariot',
        'military', 'conquest', 'enemy', 'victory', 'campaign',
        'fortress', 'siege', 'bow', 'arrow', 'sword'
    ]
}


# ===========================
#  NEW: Upload Tracker
# ===========================
class UploadTracker:
    """
    Tracks which files have been successfully uploaded to Pinecone.
    Enables resume functionality.
    """
    
    def __init__(self, tracker_file: str = UPLOAD_TRACKER_FILE):
        """
        Initialize upload tracker.
        
        Args:
            tracker_file: JSON file to store upload history
        """
        self.tracker_file = tracker_file
        self.uploaded_files = self._load_tracker()
    
    def _load_tracker(self) -> Dict[str, Dict]:
        """Load upload history from disk."""
        if os.path.exists(self.tracker_file):
            try:
                with open(self.tracker_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Failed to load upload tracker: {e}")
                return {}
        return {}
    
    def _save_tracker(self):
        """Save upload history to disk."""
        try:
            with open(self.tracker_file, 'w', encoding='utf-8') as f:
                json.dump(self.uploaded_files, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.warning(f"Failed to save upload tracker: {e}")
    
    def is_uploaded(self, filepath: str) -> bool:
        """
        Check if file has been uploaded.
        
        Args:
            filepath: Path to PDF file
            
        Returns:
            True if file was previously uploaded successfully
        """
        # Generate file hash for verification
        file_hash = self._get_file_hash(filepath)
        
        if filepath in self.uploaded_files:
            stored_hash = self.uploaded_files[filepath].get('hash')
            
            # Check if file was modified (hash changed)
            if stored_hash == file_hash:
                return True
            else:
                logging.info(f"    File modified since last upload: {filepath}")
                return False
        
        return False
    
    def _get_file_hash(self, filepath: str) -> str:
        """Get MD5 hash of file for change detection."""
        try:
            with open(filepath, 'rb') as f:
                # Hash first 1MB for speed (enough to detect changes)
                return hashlib.md5(f.read(1024 * 1024)).hexdigest()
        except Exception as e:
            logging.warning(f"Failed to hash file: {e}")
            return ""
    
    def mark_uploaded(self, filepath: str, chunks_count: int, namespace: str):
        """
        Mark file as successfully uploaded.
        
        Args:
            filepath: Path to PDF file
            chunks_count: Number of chunks uploaded
            namespace: Pinecone namespace used
        """
        file_hash = self._get_file_hash(filepath)
        
        self.uploaded_files[filepath] = {
            'hash': file_hash,
            'chunks': chunks_count,
            'namespace': namespace,
            'uploaded_at': datetime.now().isoformat(),
            'size': os.path.getsize(filepath)
        }
        
        self._save_tracker()
    
    def get_uploaded_files(self) -> Set[str]:
        """Get set of all uploaded file paths."""
        return set(self.uploaded_files.keys())
    
    def get_stats(self) -> Dict:
        """Get upload statistics."""
        total_chunks = sum(f['chunks'] for f in self.uploaded_files.values())
        
        return {
            'total_files': len(self.uploaded_files),
            'total_chunks': total_chunks,
            'files': list(self.uploaded_files.keys())
        }
    
    def remove_file(self, filepath: str):
        """Remove file from tracker (for re-upload)."""
        if filepath in self.uploaded_files:
            del self.uploaded_files[filepath]
            self._save_tracker()
            logging.info(f" Removed {filepath} from tracker")
    
    def clear(self):
        """Clear all tracking data (force re-upload all)."""
        self.uploaded_files = {}
        self._save_tracker()
        logging.info(" Upload tracker cleared")


# ===========================
# Content Classifier
# ===========================
class ContentClassifier:
    """
    Classifies document chunks into topics using:
    1. Keyword matching
    2. Semantic similarity (optional)
    3. Heading detection
    """
    
    def __init__(self, use_semantic: bool = False):
        """
        Args:
            use_semantic: Use embeddings for classification (slower but more accurate)
        """
        self.use_semantic = use_semantic
        if use_semantic:
            self.embedding_model = get_embedding_model()
    
    def classify_chunk(self, text: str, metadata: Dict = None) -> str:
        """
        Classify a chunk into a topic.
        
        Args:
            text: Chunk text
            metadata: Chunk metadata (may contain page, heading info)
            
        Returns:
            Topic name (e.g., 'pyramids', 'pharaohs', 'general')
        """
        text_lower = text.lower()
        
        # Method 1: Check for explicit headings/titles
        heading = self._extract_heading(text)
        if heading:
            topic = self._classify_by_heading(heading)
            if topic:
                return topic
        
        # Method 2: Keyword matching with scoring
        topic_scores = defaultdict(int)
        
        for topic, keywords in TOPIC_KEYWORDS.items():
            for keyword in keywords:
                # Count occurrences (weighted by keyword importance)
                count = text_lower.count(keyword.lower())
                # Longer keywords = more specific = higher weight
                weight = len(keyword.split())
                topic_scores[topic] += count * weight
        
        # Get top topic
        if topic_scores:
            best_topic = max(topic_scores.items(), key=lambda x: x[1])
            # Require minimum score to avoid random classification
            if best_topic[1] >= 2:
                return best_topic[0]
        
        # Method 3: Semantic similarity (if enabled and no keyword match)
        if self.use_semantic:
            topic = self._classify_semantic(text)
            if topic:
                return topic
        
        # Default: general
        return 'general'
    
    def _extract_heading(self, text: str) -> Optional[str]:
        """Extract heading from text (first line if ALL CAPS or #-marked)."""
        lines = text.split('\n', 2)
        if not lines:
            return None
        
        first_line = lines[0].strip()
        
        # Check if heading (all caps, short, or markdown-style)
        if len(first_line) < 100 and (
            first_line.isupper() or
            first_line.startswith('#') or
            first_line.startswith('Chapter') or
            first_line.startswith('CHAPTER')
        ):
            return first_line
        
        return None
    
    def _classify_by_heading(self, heading: str) -> Optional[str]:
        """Classify based on heading text."""
        heading_lower = heading.lower()
        
        # Direct matches
        for topic, keywords in TOPIC_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in heading_lower:
                    return topic
        
        return None
    
    def _classify_semantic(self, text: str) -> Optional[str]:
        """Classify using semantic similarity (expensive!)."""
        # This is a placeholder - in production, you'd:
        # 1. Embed the text
        # 2. Compare to topic embeddings
        # 3. Return closest topic
        # Skipped for now to keep it fast
        return None
    
    def batch_classify(self, chunks: List) -> Dict[str, List]:
        """
        Classify multiple chunks and group by topic.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            Dict mapping topic → list of chunks
        """
        topics_map = defaultdict(list)
        
        logging.info(" Classifying chunks by content...")
        
        for chunk in tqdm(chunks, desc="Classifying"):
            topic = self.classify_chunk(chunk.page_content, chunk.metadata)
            # Store topic in metadata
            chunk.metadata['topic'] = topic
            topics_map[topic].append(chunk)
        
        # Log distribution
        logging.info(f" Content distribution:")
        for topic, topic_chunks in sorted(topics_map.items(), key=lambda x: len(x[1]), reverse=True):
            logging.info(f"   {topic}: {len(topic_chunks)} chunks")
        
        return topics_map


# ===========================
# Smart Pinecone Manager
# ===========================
class SmartPineconeManager:
    """
    Enhanced Pinecone manager with intelligent content splitting.
    NOW USES METADATA FILTERING INSTEAD OF 100+ NAMESPACES!
     NEW: Resume upload functionality
    """
    
    def __init__(self, 
                 index_name: str = INDEX_NAME,
                 use_content_classification: bool = True,
                 namespace_strategy: str = NAMESPACE_STRATEGY):
        """
        Args:
            index_name: Pinecone index name
            use_content_classification: Enable smart content-based splitting
            namespace_strategy: "single" (one namespace) or "per_book" (one per book)
        """
        self.index_name = index_name
        self.use_classification = use_content_classification
        self.namespace_strategy = namespace_strategy
        
        # Initialize
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = self._get_or_create_index()
        self.embedding_model = get_embedding_model()
        self.reranker = get_reranker()
        
        #  NEW: Upload tracker
        self.tracker = UploadTracker()
        
        if use_content_classification:
            self.classifier = ContentClassifier(use_semantic=False)
        
        logging.info(f" Smart Pinecone manager ready")
        logging.info(f"   Content classification: {'ON' if use_content_classification else 'OFF'}")
        logging.info(f"   Namespace strategy: {namespace_strategy}")
    
    def _get_or_create_index(self):
        """Get or create Pinecone index."""
        existing = [idx.name for idx in self.pc.list_indexes()]
        
        if self.index_name not in existing:
            logging.info(f"🔨 Creating index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=DIMENSION,
                metric=METRIC,
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        
        return self.pc.Index(self.index_name)
    
    def _sanitize_namespace(self, name: str) -> str:
        """Sanitize namespace name."""
        import re
        
        # Remove extension
        name = os.path.splitext(name)[0]
        
        # Lowercase and sanitize
        name = name.lower()
        name = re.sub(r'[^a-z0-9]+', '-', name)
        name = name.strip('-')
        
        # Limit length
        if len(name) > 50:
            name = name[:50].rstrip('-')
        
        return name or 'default'
    
    def _get_namespace_for_book(self, book_name: str) -> str:
        """
        Get namespace based on strategy.
        
        Args:
            book_name: Name of the book
            
        Returns:
            Namespace name
        """
        if self.namespace_strategy == "single":
            return "all-books"
        elif self.namespace_strategy == "per_book":
            return self._sanitize_namespace(book_name)
        else:
            raise ValueError(f"Unknown namespace strategy: {self.namespace_strategy}")
    
    def index_documents(self, data_path: str = DATA_PATH, force_reindex: bool = False):
        """
        Load and index documents with smart content splitting.
         NEW: Only uploads new/modified files (resume functionality)
        
        Args:
            data_path: Path to data directory
            force_reindex: If True, re-upload all files (ignore tracker)
        """
        logging.info("="*60)
        logging.info(" SMART DOCUMENT INDEXING WITH RESUME")
        logging.info("="*60)
        
        # Load PDFs
        pdf_files = [f for f in os.listdir(data_path) if f.endswith('.pdf')]
        
        if not pdf_files:
            logging.warning(f" No PDFs found in {data_path}")
            return
        
        logging.info(f" Found {len(pdf_files)} PDF files")
        
        #  Check which files need uploading
        if force_reindex:
            logging.info(" Force re-index enabled - uploading all files")
            self.tracker.clear()
            files_to_upload = pdf_files
        else:
            uploaded_stats = self.tracker.get_stats()
            logging.info(f" Previously uploaded: {uploaded_stats['total_files']} files, {uploaded_stats['total_chunks']} chunks")
            
            files_to_upload = []
            for pdf_file in pdf_files:
                filepath = os.path.join(data_path, pdf_file)
                if not self.tracker.is_uploaded(filepath):
                    files_to_upload.append(pdf_file)
                else:
                    logging.info(f" Skipping (already uploaded): {pdf_file}")
        
        if not files_to_upload:
            logging.info("="*60)
            logging.info(" ALL FILES ALREADY UPLOADED!")
            logging.info("   Use force_reindex=True to re-upload")
            logging.info("="*60)
            return
        
        logging.info(f" Files to upload: {len(files_to_upload)}/{len(pdf_files)}")
        for f in files_to_upload:
            logging.info(f"   • {f}")
        
        total_uploaded = 0
        
        # Process each book
        for pdf_file in files_to_upload:
            filepath = os.path.join(data_path, pdf_file)
            
            logging.info("="*60)
            logging.info(f" Processing: {pdf_file}")
            logging.info("="*60)
            
            try:
                # Load book
                loader = PyMuPDFLoader(filepath)
                docs = loader.load()
                
                logging.info(f" Pages: {len(docs)}")
                
                # Chunk
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP
                )
                chunks = splitter.split_documents(docs)
                
                logging.info(f" Chunks: {len(chunks)}")
                
                # Classify by content (if enabled) - stores topic in metadata
                if self.use_classification:
                    topics_map = self.classifier.batch_classify(chunks)
                else:
                    # No classification - all chunks are 'general'
                    for chunk in chunks:
                        chunk.metadata['topic'] = 'general'
                    topics_map = {'general': chunks}
                
                # Get namespace for this book
                book_base = os.path.splitext(pdf_file)[0]
                namespace = self._get_namespace_for_book(book_base)
                
                logging.info(f"\n Namespace: {namespace}")
                logging.info(f" Total chunks: {len(chunks)}")
                
                # Upload ALL chunks to ONE namespace (topic is in metadata now!)
                uploaded = self._upload_chunks(chunks, namespace, pdf_file, book_base)
                total_uploaded += uploaded
                
                #  Mark as uploaded
                self.tracker.mark_uploaded(filepath, len(chunks), namespace)
                
                logging.info(f"\n Book complete: {pdf_file} ({uploaded} chunks)")
                
            except Exception as e:
                logging.error(f" Failed to process {pdf_file}: {e}")
                import traceback
                logging.error(traceback.format_exc())
                continue
        
        logging.info("="*60)
        logging.info(f" UPLOAD COMPLETE")
        logging.info(f"   New files uploaded: {len(files_to_upload)}")
        logging.info(f"   New chunks uploaded: {total_uploaded}")
        logging.info(f"   Total files in database: {len(self.tracker.get_uploaded_files())}")
        logging.info("="*60)
    
    def _upload_chunks(self, chunks: List, namespace: str, source_file: str, book_name: str) -> int:
        """Upload chunks to Pinecone namespace with topic in metadata."""
        uploaded = 0
        
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            vectors = []
            
            for chunk in batch:
                try:
                    # Embed
                    embedding = self.embedding_model.embed_query(chunk.page_content)
                    
                    # ID
                    chunk_hash = hashlib.md5(chunk.page_content.encode()).hexdigest()
                    
                    # Metadata - NOW INCLUDES TOPIC AND BOOK!
                    metadata = {
                        'text': chunk.page_content[:1000],  # Truncate for metadata size limit
                        'filename': source_file,
                        'book': book_name,  # NEW: Book name
                        'topic': chunk.metadata.get('topic', 'general'),  # NEW: Topic
                        'page': chunk.metadata.get('page', 0),
                        'created_at': datetime.now().isoformat()
                    }
                    
                    vectors.append((chunk_hash, embedding, metadata))
                    
                except Exception as e:
                    logging.warning(f"Failed to embed chunk: {e}")
                    continue
            
            # Upload
            if vectors:
                try:
                    self.index.upsert(vectors=vectors, namespace=namespace)
                    uploaded += len(vectors)
                except Exception as e:
                    logging.error(f"Upload failed: {e}")
                    # Continue with other batches
            
            # Progress
            if (i + BATCH_SIZE) % 200 == 0:
                logging.info(f"    Uploaded {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)}...")
        
        return uploaded
    
    def query(self, 
              query_text: str,
              top_k: int = 5,
              topic_filter: Optional[str] = None,
              book_filter: Optional[str] = None) -> List:
        """
        Smart query with topic and book filtering using METADATA.
        
        Args:
            query_text: Search query
            top_k: Number of results
            topic_filter: Filter by topic (e.g., 'pyramids')
            book_filter: Filter by book (e.g., 'ancient-egypt')
            
        Returns:
            Search results with metadata filtering
        """
        # Generate embedding
        query_embedding = self.embedding_model.embed_query(query_text)
        
        # Build metadata filter
        filter_dict = {}
        if topic_filter:
            filter_dict['topic'] = topic_filter
        if book_filter:
            filter_dict['book'] = {"$eq": book_filter}
        
        # Get relevant namespaces
        stats = self.index.describe_index_stats()
        all_namespaces = list(stats.namespaces.keys())
        
        logging.info(f" Searching across {len(all_namespaces)} namespace(s)")
        if filter_dict:
            logging.info(f"    Filters: {filter_dict}")
        
        # Query each namespace with metadata filter
        all_results = []
        for ns in all_namespaces:
            try:
                results = self.index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    namespace=ns,
                    include_metadata=True,
                    filter=filter_dict if filter_dict else None
                )
                
                if results and results.matches:
                    all_results.extend(results.matches)
            except Exception as e:
                logging.warning(f"Failed to query {ns}: {e}")
        
        # Sort by score
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        return all_results[:top_k * 2]  # Return top results overall
    
    def list_topics(self, book_name: Optional[str] = None) -> Dict[str, List[str]]:
        """
        List all topics from metadata (no longer from namespaces).
        NOTE: This requires scanning vectors - expensive for large datasets.
        Use sparingly or implement a metadata tracking system.
        
        Returns:
            Dict mapping book → list of topics
        """
        # This is now more complex since topics are in metadata, not namespaces
        # For demonstration, we'll return a placeholder
        logging.warning("list_topics() now requires scanning metadata - not implemented in this version")
        logging.warning("Consider tracking topics separately or using index stats")
        
        return {
            "note": ["Topics are now stored in metadata", "Use query filters instead"]
        }
    
    def get_stats(self) -> Dict:
        """Get detailed statistics."""
        stats = self.index.describe_index_stats()
        tracker_stats = self.tracker.get_stats()
        
        return {
            'index_name': self.index_name,
            'total_vectors': stats.total_vector_count,
            'total_namespaces': len(stats.namespaces),
            'namespace_strategy': self.namespace_strategy,
            'namespaces': list(stats.namespaces.keys()),
            'classification_enabled': self.use_classification,
            'uploaded_files': tracker_stats['total_files'],
            'uploaded_chunks': tracker_stats['total_chunks'],
            'note': 'Topics are stored in metadata, use query filters'
        }
    
    #  NEW: Management methods
    def reset_upload_tracker(self):
        """Clear upload tracker (force re-upload all files next time)."""
        self.tracker.clear()
        logging.info(" Upload tracker reset - all files will be re-uploaded on next run")
    
    def remove_file_from_tracker(self, filename: str):
        """Remove specific file from tracker (will be re-uploaded next time)."""
        filepath = os.path.join(DATA_PATH, filename)
        self.tracker.remove_file(filepath)
    
    def show_uploaded_files(self):
        """Show all uploaded files."""
        stats = self.tracker.get_stats()
        logging.info("="*60)
        logging.info(f" UPLOADED FILES ({stats['total_files']} files)")
        logging.info("="*60)
        for filepath in stats['files']:
            info = self.tracker.uploaded_files[filepath]
            logging.info(f" {os.path.basename(filepath)}")
            logging.info(f"   Chunks: {info['chunks']}")
            logging.info(f"   Namespace: {info['namespace']}")
            logging.info(f"   Uploaded: {info['uploaded_at']}")


# ===========================
# Unified Interface
# ===========================
def load_memory_system(use_smart_split: bool = True, 
                      namespace_strategy: str = "per_book",
                      force_reindex: bool = False):
    """
    Load Pinecone with optional smart content splitting.
     NEW: Resume upload - only uploads new files
    
    Args:
        use_smart_split: Enable intelligent content-based splitting
        namespace_strategy: "single" or "per_book"
        force_reindex: If True, re-upload all files (ignore previous uploads)
    """
    logging.info("="*60)
    logging.info(" Loading Smart Pinecone Memory")
    logging.info("="*60)
    
    manager = SmartPineconeManager(
        use_content_classification=use_smart_split,
        namespace_strategy=namespace_strategy
    )
    
    # Index documents (with resume support)
    manager.index_documents(force_reindex=force_reindex)
    
    # Stats
    stats = manager.get_stats()
    logging.info(f"\n Final Stats:")
    logging.info(f"   Namespaces: {stats['total_namespaces']}")
    logging.info(f"   Vectors: {stats['total_vectors']}")
    logging.info(f"   Uploaded files: {stats['uploaded_files']}")
    logging.info(f"   Strategy: {stats['namespace_strategy']}")
    
    logging.info("="*60)
    logging.info(" Smart Memory System Ready")
    logging.info("="*60 + "\n")
    
    return manager, manager.reranker


# ===========================
# Test
# ===========================
if __name__ == "__main__":
    print("\n Testing Smart Pinecone System with Resume\n")
    
    # Load with smart splitting and per-book namespaces
    manager, reranker = load_memory_system(
        use_smart_split=True,
        namespace_strategy="per_book",
        force_reindex=False  
    )
    
    # Show what's uploaded
    manager.show_uploaded_files()
    
    # Test smart query with metadata filtering
    print("\n" + "="*60)
    print(" Smart Query Test with Metadata Filtering")
    print("="*60)
    
    query = "pyramid construction techniques"
    print(f"\nQuery: {query}")
    print("Filtering: topic='pyramids'\n")
    
    results = manager.query(query, top_k=3, topic_filter='pyramids')
    
    for i, match in enumerate(results, 1):
        print(f"{i}. Book: {match.metadata.get('book')}")
        print(f"   Topic: {match.metadata.get('topic')}")
        print(f"   Score: {match.score:.4f}\n")
"""
Vector Database RAG Engine for WHALE-RE

Supports ChromaDB and FAISS with sentence embeddings for semantic search.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Any, Literal
import hashlib

# Optional imports for vector databases
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    import faiss
    import numpy as np
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# PDF extraction
try:
    from PyPDF2 import PdfReader
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    print("PyPDF2 not available. Install with: pip install PyPDF2")

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False


class VectorRAGEngine:
    """
    Vector-based RAG engine using embeddings and vector databases.
    
    Supports:
    - ChromaDB: Persistent vector store with metadata filtering
    - FAISS: High-performance similarity search
    - SentenceTransformers: State-of-the-art embeddings
    """
    
    def __init__(
        self,
        rag_dir: str = "data/RAG",
        backend: Literal["chromadb", "faiss"] = "chromadb",
        embedding_model: str = "all-MiniLM-L6-v2",
        persist_dir: str = "data/vector_store",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize vector RAG engine.
        
        Args:
            rag_dir: Directory containing source documents
            backend: Vector database backend ('chromadb' or 'faiss')
            embedding_model: SentenceTransformer model name
            persist_dir: Directory for vector store persistence
            chunk_size: Characters per chunk
            chunk_overlap: Overlap between chunks
        """
        self.rag_dir = Path(rag_dir)
        self.backend = backend
        self.persist_dir = Path(persist_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Validate backend availability
        if backend == "chromadb" and not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB not available. Install: pip install chromadb")
        if backend == "faiss" and not FAISS_AVAILABLE:
            raise ImportError("FAISS not available. Install: pip install faiss-cpu")
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("SentenceTransformers not available. Install: pip install sentence-transformers")
        
        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize vector database
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        if backend == "chromadb":
            self._init_chromadb()
        elif backend == "faiss":
            self._init_faiss()
        
        # Load and index documents
        self._load_and_index_documents()
    
    def _init_chromadb(self):
        """Initialize ChromaDB client and collection."""
        print("Initializing ChromaDB...")
        self.chroma_client = chromadb.PersistentClient(path=str(self.persist_dir / "chromadb"))
        
        # Create or get collection
        try:
            self.collection = self.chroma_client.get_or_create_collection(
                name="whale_re_standards",
                metadata={"hnsw:space": "cosine"}
            )
            print(f"ChromaDB collection loaded with {self.collection.count()} documents")
        except Exception as e:
            print(f"Error initializing ChromaDB: {e}")
            raise
    
    def _init_faiss(self):
        """Initialize FAISS index."""
        print("Initializing FAISS...")
        self.faiss_index = None
        self.faiss_metadata: List[Dict[str, Any]] = []
        
        # Try to load existing index
        index_file = self.persist_dir / "faiss.index"
        metadata_file = self.persist_dir / "faiss_metadata.json"
        
        if index_file.exists() and metadata_file.exists():
            try:
                self.faiss_index = faiss.read_index(str(index_file))
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    self.faiss_metadata = json.load(f)
                print(f"FAISS index loaded with {self.faiss_index.ntotal} vectors")
                return
            except Exception as e:
                print(f"Could not load existing FAISS index: {e}")
        
        # Create new index
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product (cosine with normalized vectors)
        self.faiss_metadata = []
    
    def _load_and_index_documents(self):
        """Load documents from RAG directory and create embeddings."""
        if not self.rag_dir.exists():
            print(f"RAG directory not found: {self.rag_dir}")
            return
        
        # Check if already indexed
        if self.backend == "chromadb" and self.collection.count() > 0:
            print("Documents already indexed in ChromaDB")
            return
        
        if self.backend == "faiss" and self.faiss_index.ntotal > 0:
            print("Documents already indexed in FAISS")
            return
        
        print("Loading and indexing documents...")
        documents = self._load_documents()
        
        if not documents:
            print("No documents found to index")
            return
        
        chunks = self._chunk_documents(documents)
        print(f"Created {len(chunks)} chunks from {len(documents)} documents")
        
        # Create embeddings and index
        self._index_chunks(chunks)
        print("Indexing complete")
    
    def _load_documents(self) -> List[Dict[str, str]]:
        """Load all text and PDF files from RAG directory."""
        documents = []
        supported_exts = {'.txt', '.md', '.pdf'}
        
        for root, _, files in os.walk(self.rag_dir):
            for file in files:
                if Path(file).suffix.lower() in supported_exts:
                    filepath = Path(root) / file
                    try:
                        # Extract category from folder structure
                        rel_path = filepath.relative_to(self.rag_dir)
                        category = rel_path.parts[0] if len(rel_path.parts) > 1 else "General"
                        
                        # Load content based on file type
                        if filepath.suffix.lower() == '.pdf':
                            content = self._extract_pdf_text(filepath)
                        else:
                            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                        
                        if content.strip():
                            documents.append({
                                "path": str(filepath),
                                "filename": file,
                                "category": category,
                                "content": content
                            })
                            print(f"Loaded: {file} ({len(content)} chars, category: {category})")
                    except Exception as e:
                        print(f"Error loading {filepath}: {e}")
        
        return documents
    
    def _extract_pdf_text(self, filepath: Path) -> str:
        """
        Extract text from PDF file using available libraries.
        Tries pdfplumber first (better quality), falls back to PyPDF2.
        """
        text = ""
        
        # Try pdfplumber first (better quality, handles complex layouts)
        if PDFPLUMBER_AVAILABLE:
            try:
                import pdfplumber
                with pdfplumber.open(filepath) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n\n"
                if text.strip():
                    print(f"  Extracted {len(text)} chars from PDF using pdfplumber")
                    return text
            except Exception as e:
                print(f"  pdfplumber extraction failed: {e}, trying PyPDF2...")
        
        # Fallback to PyPDF2
        if PYPDF2_AVAILABLE:
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(str(filepath))
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
                if text.strip():
                    print(f"  Extracted {len(text)} chars from PDF using PyPDF2")
                    return text
            except Exception as e:
                print(f"  PyPDF2 extraction failed: {e}")
        
        # If both fail
        if not text.strip():
            print(f"  Warning: Could not extract text from PDF {filepath.name}")
            print(f"  Install pdfplumber: pip install pdfplumber")
        
        return text
    
    def _chunk_documents(self, documents: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Split documents into overlapping chunks."""
        chunks = []
        
        for doc in documents:
            content = doc["content"]
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            
            current_chunk = []
            current_size = 0
            
            for para in paragraphs:
                para_size = len(para)
                
                if current_size + para_size > self.chunk_size and current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    chunks.append({
                        "text": chunk_text,
                        "category": doc["category"],
                        "source": doc["filename"],
                        "path": doc["path"]
                    })
                    
                    # Keep last paragraph for overlap
                    if self.chunk_overlap > 0 and current_chunk:
                        current_chunk = [current_chunk[-1]]
                        current_size = len(current_chunk[0])
                    else:
                        current_chunk = []
                        current_size = 0
                
                current_chunk.append(para)
                current_size += para_size
            
            # Add remaining chunk
            if current_chunk:
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "category": doc["category"],
                    "source": doc["filename"],
                    "path": doc["path"]
                })
        
        return chunks
    
    def _index_chunks(self, chunks: List[Dict[str, Any]]):
        """Create embeddings and index chunks in vector database."""
        if not chunks:
            return
        
        # Extract texts and create embeddings
        texts = [chunk["text"] for chunk in chunks]
        print(f"Creating embeddings for {len(texts)} chunks...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        
        if self.backend == "chromadb":
            self._index_chromadb(chunks, embeddings)
        elif self.backend == "faiss":
            self._index_faiss(chunks, embeddings)
    
    def _index_chromadb(self, chunks: List[Dict[str, Any]], embeddings):
        """Index chunks in ChromaDB."""
        ids = [hashlib.md5(chunk["text"].encode()).hexdigest() for chunk in chunks]
        metadatas = [
            {
                "category": chunk["category"],
                "source": chunk["source"],
                "path": chunk["path"]
            }
            for chunk in chunks
        ]
        documents = [chunk["text"] for chunk in chunks]
        
        # Add to collection in batches (ChromaDB has limits)
        batch_size = 5000
        for i in range(0, len(chunks), batch_size):
            end_idx = min(i + batch_size, len(chunks))
            self.collection.add(
                ids=ids[i:end_idx],
                embeddings=embeddings[i:end_idx].tolist(),
                metadatas=metadatas[i:end_idx],
                documents=documents[i:end_idx]
            )
    
    def _index_faiss(self, chunks: List[Dict[str, Any]], embeddings):
        """Index chunks in FAISS."""
        # Normalize embeddings for cosine similarity
        embeddings = embeddings.astype('float32')
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.faiss_index.add(embeddings)
        
        # Store metadata separately
        for chunk in chunks:
            self.faiss_metadata.append({
                "text": chunk["text"],
                "category": chunk["category"],
                "source": chunk["source"],
                "path": chunk["path"]
            })
        
        # Save index
        faiss.write_index(self.faiss_index, str(self.persist_dir / "faiss.index"))
        with open(self.persist_dir / "faiss_metadata.json", 'w', encoding='utf-8') as f:
            json.dump(self.faiss_metadata, f)
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        category_filter: Optional[str] = None,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks using vector similarity.
        
        Args:
            query: Search query
            top_k: Number of results to return
            category_filter: Filter by document category (e.g., "ISO-26262")
            score_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of dicts with 'text', 'category', 'source', 'score'
        """
        # Create query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        
        if self.backend == "chromadb":
            return self._retrieve_chromadb(query_embedding, top_k, category_filter, score_threshold)
        elif self.backend == "faiss":
            return self._retrieve_faiss(query_embedding, top_k, category_filter, score_threshold)
        
        return []
    
    def _retrieve_chromadb(
        self,
        query_embedding,
        top_k: int,
        category_filter: Optional[str],
        score_threshold: float
    ) -> List[Dict[str, Any]]:
        """Retrieve from ChromaDB."""
        where_filter = None
        if category_filter:
            where_filter = {"category": {"$eq": category_filter}}
        
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k,
            where=where_filter
        )
        
        retrieved = []
        if results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                score = 1 - results["distances"][0][i]  # Convert distance to similarity
                if score >= score_threshold:
                    retrieved.append({
                        "text": doc,
                        "category": results["metadatas"][0][i]["category"],
                        "source": results["metadatas"][0][i]["source"],
                        "score": score
                    })
        
        return retrieved
    
    def _retrieve_faiss(
        self,
        query_embedding,
        top_k: int,
        category_filter: Optional[str],
        score_threshold: float
    ) -> List[Dict[str, Any]]:
        """Retrieve from FAISS."""
        # Normalize query
        query_embedding = query_embedding.astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search
        search_k = min(top_k * 10, self.faiss_index.ntotal) if category_filter else top_k
        scores, indices = self.faiss_index.search(query_embedding, search_k)
        
        retrieved = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.faiss_metadata):
                continue
            
            metadata = self.faiss_metadata[idx]
            
            # Apply category filter
            if category_filter and category_filter.lower() not in metadata["category"].lower():
                continue
            
            # Apply score threshold
            if score < score_threshold:
                continue
            
            retrieved.append({
                "text": metadata["text"],
                "category": metadata["category"],
                "source": metadata["source"],
                "score": float(score)
            })
            
            if len(retrieved) >= top_k:
                break
        
        return retrieved
    
    def get_context_for_agent(
        self,
        agent_key: str,
        input_text: str,
        max_chunks: int = 5
    ) -> str:
        """
        Get relevant context for a specific agent.
        
        Args:
            agent_key: Agent identifier (SYS1, SYS2, Review, SYS5)
            input_text: Input text to find relevant context for
            max_chunks: Maximum number of context chunks
            
        Returns:
            Formatted context string
        """
        # Category hints per agent
        category_hints = {
            "SYS1": ["ISO-IEC-IEEE-29148", "ISO-26262"],
            "SYS2": ["ISO-26262", "AUTOSAR"],
            "Review": ["ISO-IEC-IEEE-29148", "ISO-26262"],
            "SYS5": ["ISO-26262", "ASPICE"]
        }
        
        query = input_text[:500]
        results = []
        
        # Retrieve with category hints
        for category in category_hints.get(agent_key, []):
            results.extend(self.retrieve(query, top_k=2, category_filter=category, score_threshold=0.3))
        
        # If no category results, do general search
        if not results:
            results = self.retrieve(query, top_k=max_chunks, score_threshold=0.3)
        
        # Deduplicate
        seen_texts = set()
        unique_results = []
        for r in results:
            text_hash = hashlib.md5(r["text"].encode()).hexdigest()
            if text_hash not in seen_texts:
                seen_texts.add(text_hash)
                unique_results.append(r)
                if len(unique_results) >= max_chunks:
                    break
        
        # Format context
        if not unique_results:
            return ""
        
        context_parts = ["=== RELEVANT STANDARDS CONTEXT ===\n"]
        for i, result in enumerate(unique_results, 1):
            context_parts.append(
                f"[Source {i}: {result['source']} - {result['category']} (similarity: {result['score']:.2f})]"
            )
            context_parts.append(result['text'][:800])
            context_parts.append("")
        
        context_parts.append("=== END CONTEXT ===\n")
        return "\n".join(context_parts)


# Global instances
_vector_rag_engine: Optional[VectorRAGEngine] = None
_rag_backend: str = "chromadb"  # Default backend


def get_vector_rag_engine(
    backend: Literal["chromadb", "faiss"] = None,
    force_reload: bool = False
) -> Optional[VectorRAGEngine]:
    """
    Get or create global vector RAG engine instance.
    
    Args:
        backend: Vector database backend (defaults to 'chromadb')
        force_reload: Force recreation of engine
        
    Returns:
        VectorRAGEngine instance or None if dependencies unavailable
    """
    global _vector_rag_engine, _rag_backend
    
    if backend is None:
        backend = _rag_backend
    
    if _vector_rag_engine is None or force_reload or backend != _rag_backend:
        try:
            print(f"Initializing Vector RAG Engine with backend: {backend}")
            _vector_rag_engine = VectorRAGEngine(backend=backend)
            _rag_backend = backend
            return _vector_rag_engine
        except Exception as e:
            print(f"Failed to initialize Vector RAG Engine: {e}")
            return None
    
    return _vector_rag_engine


def augment_prompt_with_vector_rag(
    agent_key: str,
    prompt: str,
    input_text: str,
    backend: Literal["chromadb", "faiss"] = "chromadb"
) -> str:
    """
    Augment agent prompt with relevant context from vector RAG.
    
    Args:
        agent_key: Agent identifier
        prompt: Original prompt template
        input_text: Input text for context retrieval
        backend: Vector database backend
        
    Returns:
        Augmented prompt with context
    """
    engine = get_vector_rag_engine(backend=backend)
    if engine is None:
        print("Vector RAG engine not available, using original prompt")
        return prompt
    
    context = engine.get_context_for_agent(agent_key, input_text)
    if context:
        return f"{context}\n\n{prompt}"
    
    return prompt

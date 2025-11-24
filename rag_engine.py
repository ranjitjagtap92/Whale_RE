"""
RAG (Retrieval-Augmented Generation) Engine for WHALE-RE

Provides context-aware retrieval from automotive standards documents in data/RAG/
to enhance agent outputs with relevant compliance information.

Supports multiple backends:
- 'simple': Keyword-based similarity (no dependencies)
- 'chromadb': ChromaDB vector database (persistent, easy to use)
- 'faiss': FAISS vector database (high performance, Facebook Research)
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Any, Literal
import hashlib
import json

# Optional imports for vector databases
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("ChromaDB not available. Install with: pip install chromadb")

try:
    import faiss
    import numpy as np
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("FAISS not available. Install with: pip install faiss-cpu (or faiss-gpu)")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("SentenceTransformers not available. Install with: pip install sentence-transformers")

# PDF extraction
try:
    from PyPDF2 import PdfReader
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

class SimpleRAGEngine:
    """Lightweight RAG engine using text similarity without external vector DB."""
    
    def __init__(self, rag_dir: str = "data/RAG"):
        self.rag_dir = Path(rag_dir)
        self.documents: List[Dict[str, str]] = []
        self.chunks: List[Dict[str, Any]] = []
        self._load_documents()
    
    def _load_documents(self):
        """Load all text and PDF files from RAG directory and subdirectories."""
        if not self.rag_dir.exists():
            print(f"RAG directory not found: {self.rag_dir}")
            return
        
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
                            self.documents.append({
                                "path": str(filepath),
                                "filename": file,
                                "category": category,
                                "content": content
                            })
                    except Exception as e:
                        print(f"Error loading {filepath}: {e}")
        
        print(f"Loaded {len(self.documents)} documents from RAG directory")
        self._chunk_documents()
    
    def _extract_pdf_text(self, filepath: Path) -> str:
        """Extract text from PDF file using available libraries."""
        text = ""
        
        # Try pdfplumber first (better quality)
        if PDFPLUMBER_AVAILABLE:
            try:
                import pdfplumber
                with pdfplumber.open(filepath) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n\n"
                if text.strip():
                    return text
            except Exception as e:
                print(f"  pdfplumber failed for {filepath.name}: {e}, trying PyPDF2...")
        
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
                    return text
            except Exception as e:
                print(f"  PyPDF2 failed for {filepath.name}: {e}")
        
        if not text.strip():
            print(f"  Warning: Could not extract text from {filepath.name}")
            print(f"  Install: pip install pdfplumber PyPDF2")
        
        return text
    
    def _chunk_documents(self, chunk_size: int = 1000, overlap: int = 200):
        """Split documents into overlapping chunks for better retrieval."""
        for doc in self.documents:
            content = doc["content"]
            # Split by paragraphs first
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            
            current_chunk = []
            current_size = 0
            
            for para in paragraphs:
                para_size = len(para)
                
                if current_size + para_size > chunk_size and current_chunk:
                    # Save current chunk
                    chunk_text = '\n\n'.join(current_chunk)
                    self.chunks.append({
                        "text": chunk_text,
                        "source": doc["filename"],
                        "category": doc["category"],
                        "path": doc["path"]
                    })
                    
                    # Start new chunk with overlap
                    if overlap > 0 and current_chunk:
                        # Keep last paragraph for overlap
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
                self.chunks.append({
                    "text": chunk_text,
                    "source": doc["filename"],
                    "category": doc["category"],
                    "path": doc["path"]
                })
        
        print(f"Created {len(self.chunks)} chunks from documents")
    
    def _simple_similarity(self, query: str, text: str) -> float:
        """Calculate simple keyword-based similarity score."""
        query_lower = query.lower()
        text_lower = text.lower()
        
        # Extract important terms (longer than 3 chars, not common words)
        common_words = {'the', 'and', 'for', 'with', 'that', 'this', 'from', 'shall', 'must', 'will'}
        query_terms = [w for w in re.findall(r'\b\w+\b', query_lower) 
                      if len(w) > 3 and w not in common_words]
        
        if not query_terms:
            return 0.0
        
        score = 0.0
        text_words = set(re.findall(r'\b\w+\b', text_lower))
        
        for term in query_terms:
            if term in text_words:
                # Exact match
                score += 2.0
            else:
                # Partial match (substring)
                if any(term in word or word in term for word in text_words if len(word) > 3):
                    score += 0.5
        
        # Normalize by query length
        return score / len(query_terms)
    
    def retrieve(self, query: str, top_k: int = 3, category_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve most relevant chunks for the query."""
        if not self.chunks:
            return []
        
        # Filter by category if specified
        candidates = self.chunks
        if category_filter:
            candidates = [c for c in self.chunks if category_filter.lower() in c["category"].lower()]
        
        # Score all chunks
        scored_chunks = []
        for chunk in candidates:
            score = self._simple_similarity(query, chunk["text"])
            if score > 0:
                scored_chunks.append((score, chunk))
        
        # Sort by score and return top_k
        scored_chunks.sort(reverse=True, key=lambda x: x[0])
        return [{"score": score, **chunk} for score, chunk in scored_chunks[:top_k]]
    
    def get_context_for_agent(self, agent_key: str, input_text: str, max_chunks: int = 3) -> str:
        """Get relevant context for a specific agent based on input text."""
        # Determine relevant categories per agent (matched to actual folder names in data/RAG/)

        category_hints = {
            "SYS1": [
                # Core System & Requirements Engineering Standards
                "ISO-26262",  # Functional Safety
                "ISO 26262",  # Functional Safety (alternative naming)
                "ISO-21434",  # Cybersecurity Engineering
                "ISO 21434",  # Cybersecurity (alternative naming)
                "ISO-IEC-IEEE-29148",  # Requirements Engineering
                "ISO 29148",  # Requirements Engineering (alternative naming)
                "ISOIECIEEE 15288",  # System Lifecycle Processes
                "ISO 15288",  # System Lifecycle (alternative naming)
                "ISO/IEC/IEEE 15288",  # System Lifecycle (ISO format)
                "ISO-12207",  # Software Lifecycle Processes
                "ISO 12207",  # Software Lifecycle (alternative naming)
                "ISO/IEC/IEEE 12207",  # Software Lifecycle (ISO format)
                "IREB",  # IREB CPRE Requirement Engineering Best Practice
                "IREB CPRE",  # IREB CPRE Guidelines
                "INCOSE",  # INCOSE SE Handbook - System Engineering Framework
                "INCOSE SE",  # INCOSE System Engineering
                "ISO-21448",  # SOTIF - Safety of Intended Functionality (ADAS/AI)
                "ISO 21448",  # SOTIF (alternative naming)
                "SOTIF"  # Safety of Intended Functionality
            ],
            "SYS2": [
                # Technical Architecture & Communication Standards
                "ISO-26262",  # Functional Safety
                "ISO 26262",  # Functional Safety (alternative naming)
                "ISO-21434",  # Cybersecurity Engineering
                "ISO 21434",  # Cybersecurity (alternative naming)
                "AUTOSAR",  # Automotive Open System Architecture (Classic/Adaptive)
                "AUTOSAR CP",  # AUTOSAR Classic Platform
                "AUTOSAR AP",  # AUTOSAR Adaptive Platform
                "ISO11898",  # ISO 11898 - CAN (Controller Area Network) Protocol
                "ISO-11898",  # CAN Protocol (alternative naming)
                "ISO 11898",  # CAN Standard
                "CAN",  # CAN Protocol
                "ISO-17987",  # ISO 17987 - LIN (Local Interconnect Network)
                "ISO 17987",  # LIN (alternative naming)
                "LIN",  # LIN Protocol
                "ISO-17458",  # ISO 17458 - FlexRay
                "ISO 17458",  # FlexRay (alternative naming)
                "FlexRay",  # FlexRay Protocol
                "IEEE 802.3",  # IEEE 802.3 Automotive Ethernet
                "Ethernet",  # Automotive Ethernet
                "SOME/IP",  # SOME/IP (Autosar) Specification
                "SOME-IP",  # SOME/IP (alternative naming)
                "ECE R100",  # ECE R100 - Electric Vehicle Safety (High Voltage)
                "ECE-R100",  # ECE R100 (alternative naming)
                "ISO-6469",  # ISO 6469 - EV Functional Safety
                "ISO 6469",  # EV Functional Safety (alternative naming)
                "ISO-21780",  # ISO 21780 - LV DC Distribution
                "ISO 21780"  # LV DC Distribution (alternative naming)
            ],
            "Review": [
                # Quality, Compliance & Requirements Review Standards
                "ISO-IEC-IEEE-29148",  # Requirements Engineering
                "ISO 29148",  # Requirements Engineering (alternative naming)
                "ISO/IEC/IEEE 29148",  # Requirements Engineering (ISO format)
                "ISO-26262",  # Functional Safety
                "ISO 26262",  # Functional Safety (alternative naming)
                "ISO-21434",  # Cybersecurity Engineering
                "ISO 21434",  # Cybersecurity (alternative naming)
                "ISOIECIEEE 15288",  # System Lifecycle Processes
                "ISO 15288",  # System Lifecycle (alternative naming)
                "ISO/IEC/IEEE 15288",  # System Lifecycle (ISO format)
                "ISO-12207",  # Software Lifecycle Processes
                "ISO 12207",  # Software Lifecycle (alternative naming)
                "ISO/IEC/IEEE 12207",  # Software Lifecycle (ISO format)
                "IREB",  # IREB CPRE Requirement Engineering Best Practice
                "IREB CPRE",  # IREB CPRE Guidelines
                "INCOSE",  # INCOSE SE Handbook - System Engineering Framework
                "INCOSE SE",  # INCOSE System Engineering
                "ISO-14300",  # ISO 14300 - Project Management & Risk in Automotive
                "ISO 14300"  # Project Management (alternative naming)
            ],
            "SYS5": [
                # Test, Validation & Diagnostics Standards
                "ISO-26262",  # Functional Safety
                "ISO 26262",  # Functional Safety (alternative naming)
                "ASPICE",  # Automotive SPICE Process Assessment
                "ISO16750",  # ISO 16750 - Environmental & Electrical Tests
                "ISO-16750",  # Environmental Tests (alternative naming)
                "ISO 16750",  # Environmental Tests
                "LV124",  # LV124 (VDA) - German OEM Low Voltage Tests
                "LV148",  # LV148 (VDA) - German OEM Low Voltage Tests
                "LV-124",  # LV124 (alternative naming)
                "LV-148",  # LV148 (alternative naming)
                "LV_VDA",  # VDA (German Automotive Industry Association) OEM Standards
                "IEC-60068",  # IEC 60068 - Environmental Tests
                "IEC 60068",  # Environmental Tests (alternative naming)
                "ISO-7637",  # ISO 7637 - Electrical Disturbances in Vehicles
                "ISO 7637",  # Electrical Disturbances (alternative naming)
                "ISO-16735",  # ISO 16735 - EMC for Vehicle Components
                "ISO 16735",  # EMC for Vehicle Components (alternative naming)
                "ISO14229",  # ISO 14229 - UDS (Unified Diagnostic Services)
                "ISO-14229",  # UDS Diagnostics (alternative naming)
                "ISO 14229",  # UDS Diagnostics
                "UDS",  # Unified Diagnostic Services
                "ISO15031",  # ISO 15031 - OBD-II (On-Board Diagnostics) Standards
                "ISO-15031",  # OBD-II (alternative naming)
                "ISO 15031",  # OBD-II Standards
                "OBD-II",  # OBD-II Diagnostics
                "SAE J1979",  # SAE J1979 - OBD PIDs
                "SAE-J1979",  # SAE J1979 (alternative naming)
                "J1979",  # OBD PIDs
                "SAE J2012",  # SAE J2012 - Diagnostic Trouble Codes
                "SAE-J2012",  # SAE J2012 (alternative naming)
                "J2012",  # Diagnostic Trouble Codes
                "ISO-11898",  # CAN Protocol (for diagnostics)
                "ISO 11898",  # CAN Standard
                "CAN",  # CAN Protocol
                "ISO-21434",  # Cybersecurity Engineering
                "ISO 21434",  # Cybersecurity (alternative naming)
                "ECE R100",  # Electric Vehicle Safety (High Voltage)
                "ECE-R100",  # ECE R100 (alternative naming)
                "ISO-6469",  # EV Functional Safety
                "ISO 6469",  # EV Safety (alternative naming)
                "ISO-21448",  # SOTIF - Safety of Intended Functionality
                "ISO 21448",  # SOTIF (alternative naming)
                "SOTIF"  # Safety of Intended Functionality
            ]
        }
        
        # Build query from input
        query = input_text[:500]  # Use first 500 chars as query
        
        # Retrieve with category hints
        results = []
        for category in category_hints.get(agent_key, []):
            results.extend(self.retrieve(query, top_k=2, category_filter=category))
        
        # If no category results, do general search
        if not results:
            results = self.retrieve(query, top_k=max_chunks)
        
        # Deduplicate and limit
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
            context_parts.append(f"[Source {i}: {result['source']} - {result['category']}]")
            context_parts.append(result['text'][:800])  # Limit chunk size
            context_parts.append("")
        
        context_parts.append("=== END CONTEXT ===\n")
        return "\n".join(context_parts)


# Global RAG engine instance
_rag_engine: Optional[SimpleRAGEngine] = None

def get_rag_engine() -> SimpleRAGEngine:
    """Get or create the global RAG engine instance."""
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = SimpleRAGEngine()
    return _rag_engine

def augment_prompt_with_rag(agent_key: str, original_prompt: str, input_text: str, enabled: bool = True) -> str:
    """Augment agent prompt with RAG context if enabled."""
    if not enabled:
        return original_prompt
    
    try:
        engine = get_rag_engine()
        context = engine.get_context_for_agent(agent_key, input_text, max_chunks=3)
        
        if context:
            # Insert context before the input section
            augmented = original_prompt + "\n\n" + context + "\n\nUSER INPUT:\n"
            return augmented
        else:
            return original_prompt
    except Exception as e:
        print(f"RAG augmentation failed: {e}")
        return original_prompt

"""
Test script for PDF loading in vector RAG engine
"""

import sys
from pathlib import Path

# Test 1: Check PDF library availability
print("=== Testing PDF Library Availability ===\n")

try:
    import pdfplumber
    print("✓ pdfplumber is available")
    PDFPLUMBER_OK = True
except ImportError:
    print("✗ pdfplumber not found - Install: pip install pdfplumber")
    PDFPLUMBER_OK = False

try:
    from PyPDF2 import PdfReader
    print("✓ PyPDF2 is available")
    PYPDF2_OK = True
except ImportError:
    print("✗ PyPDF2 not found - Install: pip install PyPDF2")
    PYPDF2_OK = False

if not PDFPLUMBER_OK and not PYPDF2_OK:
    print("\n❌ No PDF libraries available. Install at least one:")
    print("   pip install pdfplumber PyPDF2")
    sys.exit(1)

print()

# Test 2: Check for PDF files in RAG directory
print("=== Scanning for PDF files ===\n")

rag_dir = Path("data/RAG")
if not rag_dir.exists():
    print(f"❌ RAG directory not found: {rag_dir}")
    sys.exit(1)

pdf_files = list(rag_dir.rglob("*.pdf"))
if not pdf_files:
    print("⚠️  No PDF files found in data/RAG/")
    print("    Add PDF standards to data/RAG/<category>/ to test")
    sys.exit(0)

print(f"Found {len(pdf_files)} PDF files:\n")
for pdf in pdf_files[:10]:  # Show first 10
    rel_path = pdf.relative_to(rag_dir)
    print(f"  • {rel_path}")
if len(pdf_files) > 10:
    print(f"  ... and {len(pdf_files) - 10} more")

print()

# Test 3: Extract text from first PDF
print("=== Testing PDF Text Extraction ===\n")

test_pdf = pdf_files[0]
print(f"Testing with: {test_pdf.name}\n")

# Try pdfplumber
if PDFPLUMBER_OK:
    try:
        import pdfplumber
        with pdfplumber.open(test_pdf) as pdf:
            first_page = pdf.pages[0]
            text = first_page.extract_text()
            if text:
                print(f"✓ pdfplumber: Extracted {len(text)} chars from first page")
                print(f"  Preview: {text[:200]}...\n")
            else:
                print("✗ pdfplumber: No text extracted\n")
    except Exception as e:
        print(f"✗ pdfplumber error: {e}\n")

# Try PyPDF2
if PYPDF2_OK:
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(str(test_pdf))
        first_page = reader.pages[0]
        text = first_page.extract_text()
        if text:
            print(f"✓ PyPDF2: Extracted {len(text)} chars from first page")
            print(f"  Preview: {text[:200]}...\n")
        else:
            print("✗ PyPDF2: No text extracted\n")
    except Exception as e:
        print(f"✗ PyPDF2 error: {e}\n")

# Test 4: Load documents using RAG engine
print("=== Testing RAG Engine PDF Loading ===\n")

try:
    # Try vector RAG first
    try:
        from rag_vector import VectorRAGEngine, CHROMADB_AVAILABLE, FAISS_AVAILABLE
        
        if CHROMADB_AVAILABLE or FAISS_AVAILABLE:
            backend = "chromadb" if CHROMADB_AVAILABLE else "faiss"
            print(f"Testing with VectorRAGEngine (backend: {backend})...\n")
            
            # Note: First run will take time to build embeddings
            print("⏳ Building vector index (this may take 2-5 minutes on first run)...")
            engine = VectorRAGEngine(backend=backend)
            
            # Count PDF documents
            pdf_docs = [d for d in engine._load_documents() if d['filename'].endswith('.pdf')]
            print(f"\n✓ Loaded {len(pdf_docs)} PDF documents via VectorRAGEngine")
            
            if pdf_docs:
                sample = pdf_docs[0]
                print(f"\nSample PDF document:")
                print(f"  File: {sample['filename']}")
                print(f"  Category: {sample['category']}")
                print(f"  Content length: {len(sample['content'])} chars")
                print(f"  Preview: {sample['content'][:200]}...")
        else:
            raise ImportError("No vector backends available")
            
    except ImportError:
        # Fallback to simple RAG
        from rag_engine import SimpleRAGEngine
        
        print("Testing with SimpleRAGEngine...\n")
        engine = SimpleRAGEngine()
        
        # Count PDF documents
        pdf_docs = [d for d in engine.documents if d['filename'].endswith('.pdf')]
        print(f"✓ Loaded {len(pdf_docs)} PDF documents via SimpleRAGEngine")
        
        if pdf_docs:
            sample = pdf_docs[0]
            print(f"\nSample PDF document:")
            print(f"  File: {sample['filename']}")
            print(f"  Category: {sample['category']}")
            print(f"  Content length: {len(sample['content'])} chars")
            print(f"  Preview: {sample['content'][:200]}...")
    
    print("\n✅ PDF loading test successful!")
    
except Exception as e:
    print(f"❌ Error testing RAG engine: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Test Complete ===")

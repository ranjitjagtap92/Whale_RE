# PDF Standards Integration Guide

## Overview

Your WHALE-RE system now supports loading PDF standards documents directly. All PDFs in your `data/RAG/` directory will be automatically indexed and used for context-aware requirement generation.

## ✅ Your Current Setup

Based on your RAG folder structure, you have these standards available:

- **AUTOSAR** - Automotive Open System Architecture
- **ASPICE** - Automotive SPICE Process Assessment
- **ISO-26262** - Functional Safety
- **ISO-IEC-IEEE-29148** - Requirements Engineering
- **ISO11898** - CAN Protocol
- **ISO14229** - UDS Diagnostic Protocol
- **ISO15031** - OBD-II Standards
- **ISO16750** - Environmental Tests
- **ISOIECIEEE 15288** - System Lifecycle Processes
- **ISO-26262 - Functional Safety** (additional)
- **ECE R100** - Electric Vehicle Safety
- **ISO-21434** - Cybersecurity
- **LV_VDA** - VDA OEM Standards

## 🚀 Quick Start

### 1. Install PDF Processing Libraries

```powershell
pip install pdfplumber PyPDF2
```

**Why both?**
- `pdfplumber`: Best quality text extraction (recommended)
- `PyPDF2`: Fallback for complex PDFs

### 2. Verify PDF Detection

Run the test script:

```powershell
python test_pdf_loading.py
```

This will:
- Check if PDF libraries are installed
- Scan for PDF files in `data/RAG/`
- Test extraction from a sample PDF
- Verify RAG engine can load PDFs

### 3. Build Vector Index (One-Time)

For semantic search with embeddings:

```powershell
# Install vector RAG dependencies
pip install -r requirements_rag_vector.txt

# Build index (takes 2-5 minutes for ~50 PDFs)
python -c "from rag_vector import VectorRAGEngine; VectorRAGEngine(backend='chromadb')"
```

### 4. Use in App

```powershell
streamlit run app.py
```

Enable RAG on any agent page - PDFs are automatically included!

## 📊 How PDF Loading Works

### Automatic Processing

1. **Discovery**: RAG engine scans `data/RAG/**/*.pdf`
2. **Extraction**: Converts PDF pages to text
   - Tries `pdfplumber` first (better quality)
   - Falls back to `PyPDF2` if needed
3. **Chunking**: Splits text into 1000-char chunks with 200-char overlap
4. **Indexing**: Creates vector embeddings (for vector RAG) or keyword index (simple RAG)
5. **Retrieval**: Finds relevant chunks based on query similarity

### Category Detection

PDFs inherit category from their folder:

```
data/RAG/
├── ISO-26262/
│   └── ISO_26262_Part6.pdf  → Category: "ISO-26262"
├── AUTOSAR/
│   └── AUTOSAR_ClassicPlatform.pdf  → Category: "AUTOSAR"
└── ASPICE/
    └── ASPICE_PAM_31.pdf  → Category: "ASPICE"
```

### Supported PDF Types

✅ **Works Well:**
- Text-based PDFs (ISO standards, technical documents)
- Machine-readable PDFs
- Standards with clear text layout

⚠️ **May Have Issues:**
- Scanned documents (images, no OCR)
- Complex multi-column layouts
- PDFs with heavy formatting/tables

## 🔍 Testing PDF Extraction Quality

### Quick Test

```python
from rag_engine import SimpleRAGEngine

engine = SimpleRAGEngine()

# Find a PDF document
pdf_doc = next(d for d in engine.documents if d['filename'].endswith('.pdf'))

print(f"File: {pdf_doc['filename']}")
print(f"Category: {pdf_doc['category']}")
print(f"Extracted: {len(pdf_doc['content'])} characters")
print(f"\nFirst 500 chars:\n{pdf_doc['content'][:500]}")
```

### Check Extraction Quality

Good extraction should have:
- ✅ Complete sentences
- ✅ Proper spacing
- ✅ Section headings preserved
- ✅ Minimal garbled text

Poor extraction signs:
- ❌ Random characters
- ❌ Missing spaces between words
- ❌ Jumbled text order
- ❌ Empty content

## 🛠️ Troubleshooting

### Problem: "No text extracted from PDF"

**Solution 1**: Install better PDF library
```powershell
pip install pdfplumber
```

**Solution 2**: PDF might be scanned image
- Use OCR: `pip install pytesseract`
- Or convert PDF to text manually

**Solution 3**: Check PDF is not encrypted/protected
```python
from PyPDF2 import PdfReader
reader = PdfReader("your_file.pdf")
print(f"Encrypted: {reader.is_encrypted}")
```

### Problem: Garbled or incomplete text

**Cause**: Complex PDF layout confuses extraction

**Solutions**:
1. Try different extraction library (pdfplumber vs PyPDF2)
2. Manually convert to text: Copy-paste from PDF viewer → Save as .txt
3. Use dedicated PDF-to-text converter

### Problem: Very slow first run

**Cause**: Building embeddings for many large PDFs

**Solutions**:
1. **Expected behavior** for 50+ PDFs with vector RAG
2. Reduce chunk size: `VectorRAGEngine(chunk_size=500)`
3. Use simple RAG instead (no embeddings needed)
4. Process PDFs in smaller batches

### Problem: Out of memory during indexing

**Solutions**:
1. Use smaller embedding model:
   ```python
   VectorRAGEngine(embedding_model="all-MiniLM-L6-v2")
   ```
2. Reduce chunk size to process fewer chunks
3. Close other applications
4. Process one category at a time

## 📈 Performance Tips

### Optimize for Speed

1. **Use pdfplumber** (faster and better quality):
   ```powershell
   pip install pdfplumber
   ```

2. **Cache extracted text** (optional enhancement):
   - Save extracted text as `.txt` alongside `.pdf`
   - Skip PDF extraction if `.txt` exists

3. **Selective indexing**:
   - Only include relevant standards per agent
   - Remove duplicate/obsolete PDFs

### Optimize for Quality

1. **Review extraction quality**:
   ```powershell
   python test_pdf_loading.py
   ```

2. **Manual text files for critical standards**:
   - Copy important sections from PDF
   - Save as `.txt` for guaranteed quality
   - Place alongside PDF in same folder

3. **Clean up text**:
   - Remove page numbers, headers, footers
   - Fix common OCR errors
   - Improve formatting

## 📚 Best Practices

### Organizing Standards

```
data/RAG/
├── ISO-26262/               # Functional Safety
│   ├── ISO26262_Part1.pdf
│   ├── ISO26262_Part6.pdf
│   └── key_sections.txt     # Manual extract of critical parts
│
├── ASPICE/                  # Process Assessment
│   ├── ASPICE_PAM_31.pdf
│   └── test_guidelines.txt  # Extracted test-specific content
│
└── AUTOSAR/                 # Architecture
    ├── AUTOSAR_CP_R2011.pdf
    └── requirements_patterns.txt
```

### Hybrid Approach (Recommended)

**PDFs for:**
- Complete standard reference
- Comprehensive documentation
- Background context

**Text files (.txt/.md) for:**
- Frequently-used sections
- Agent-specific guidelines
- High-quality extracts
- Custom examples

### Per-Agent Optimization

**Agent 1 (SYS.1)**:
- ISO 29148: Requirements engineering principles
- ISO 26262 Part 3: Concept phase requirements

**Agent 2 (SYS.2)**:
- ISO 26262 Part 4: System design requirements
- AUTOSAR: Architecture patterns
- Standards-specific requirement templates

**Agent 3 (Review)**:
- ISO 29148: Quality criteria
- ISO 26262: Safety requirements review checklists

**Agent 5 (SYS.5)**:
- ASPICE SYS.5: System qualification test process
- ISO 26262 Part 4: Verification requirements
- ISO 16750: Test conditions and procedures

## 🎯 Expected Results

### Without RAG (baseline)

```
SYS.1-001: The system shall display vehicle speed
```

### With PDF-based RAG (enhanced)

```
SYS.1-001: The system shall display vehicle speed with accuracy ±2 km/h
  Rationale: Supports driver situational awareness per ISO 26262-3:2018 clause 6.4.2
  ASIL: ASIL B (moderate risk, controllable by driver)
  Verification Method: Test per ISO 16750-2 with calibrated reference speed sensor
  Acceptance: Speed display error <2 km/h across temperature range -40°C to +85°C
```

### Quality Indicators

Good RAG integration shows:
- ✅ Standard citations (ISO 26262, ASPICE)
- ✅ Specific clauses referenced
- ✅ Correct terminology usage
- ✅ ASIL levels included
- ✅ Verification methods specified
- ✅ Compliance-aware language

## 🔄 Updating Standards

### Add New PDFs

1. Copy PDF to appropriate category folder:
   ```powershell
   Copy-Item "new_standard.pdf" "data\RAG\ISO-26262\"
   ```

2. Rebuild vector index:
   ```powershell
   # Delete existing index
   Remove-Item -Recurse data\vector_store\chromadb
   
   # Rebuild (will include new PDF)
   python -c "from rag_vector import VectorRAGEngine; VectorRAGEngine(backend='chromadb')"
   ```

3. Restart app:
   ```powershell
   streamlit run app.py
   ```

### Update Existing PDFs

Same process as adding new PDFs - delete index and rebuild.

### Monitor Index Size

```powershell
# Check total documents
python -c "from rag_vector import get_vector_rag_engine; print(f'Docs: {get_vector_rag_engine().collection.count()}')"

# Check storage size
Get-ChildItem -Recurse data\vector_store | Measure-Object -Property Length -Sum
```

## 📦 Complete Installation

All-in-one setup:

```powershell
# PDF extraction
pip install pdfplumber PyPDF2

# Vector RAG (recommended)
pip install chromadb sentence-transformers

# Or use requirements file
pip install -r requirements_rag_vector.txt
```

## ✅ Verification Checklist

- [ ] PDF libraries installed (`pdfplumber`, `PyPDF2`)
- [ ] Test script runs successfully (`python test_pdf_loading.py`)
- [ ] PDFs detected in `data/RAG/` folders
- [ ] Sample text extracted from at least one PDF
- [ ] Vector index built (if using vector RAG)
- [ ] App starts without errors
- [ ] RAG checkbox available on agent pages
- [ ] Generated requirements include standard references

## 🆘 Need Help?

### Check Console Output

The app logs PDF processing details:

```
Loaded: ISO26262_Part6.pdf (125000 chars, category: ISO-26262)
  Extracted 125000 chars from PDF using pdfplumber
Created 127 chunks from 15 documents
```

### Common Issues

| Issue | Check | Fix |
|-------|-------|-----|
| No PDFs found | Run `test_pdf_loading.py` | Verify PDFs in `data/RAG/` |
| Empty extraction | Check console for errors | Install `pdfplumber` |
| Slow indexing | Normal for first run | Wait or use simple RAG |
| OOM error | Too many large PDFs | Reduce chunk size |
| No context in output | RAG not enabled | Check RAG checkbox |

### Test Retrieval

```python
from rag_vector import VectorRAGEngine

engine = VectorRAGEngine(backend="chromadb")
results = engine.retrieve("ASIL D requirements", top_k=3)

for r in results:
    print(f"\n{r['source']} (score: {r['score']:.2f})")
    print(f"Category: {r['category']}")
    print(f"Text: {r['text'][:200]}...")
```

Your system is now ready to leverage all your PDF standards for intelligent, compliance-aware requirement generation! 🎉

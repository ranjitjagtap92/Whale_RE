# Vector RAG Setup Guide

## Overview

WHALE-RE now supports three RAG backends for retrieving relevant automotive standards context:

1. **Simple** (keyword-based): Lightweight, no extra dependencies
2. **ChromaDB** (recommended): Semantic search with persistent storage
3. **FAISS**: High-performance vector similarity search

## Installation

### Option 1: ChromaDB (Recommended)

ChromaDB is the easiest to set up and provides excellent performance with persistent storage:

```powershell
pip install chromadb sentence-transformers
```

**Pros:**
- Easy to use
- Persistent vector storage
- Metadata filtering support
- No manual index management

**Cons:**
- Slightly slower than FAISS for very large datasets

### Option 2: FAISS

FAISS (Facebook AI Similarity Search) offers maximum performance:

```powershell
# CPU version (most compatible)
pip install faiss-cpu sentence-transformers

# OR GPU version (requires CUDA)
pip install faiss-gpu sentence-transformers
```

**Pros:**
- Extremely fast similarity search
- Optimized for large-scale datasets
- GPU acceleration available

**Cons:**
- Requires manual index saving/loading
- No built-in metadata filtering

### Complete Installation

Install all dependencies at once:

```powershell
pip install -r requirements_rag_vector.txt
```

## First-Time Setup

### 1. Verify Installation

Run this command to check if vector RAG is available:

```powershell
python -c "from rag_vector import VectorRAGEngine, CHROMADB_AVAILABLE, FAISS_AVAILABLE; print(f'ChromaDB: {CHROMADB_AVAILABLE}, FAISS: {FAISS_AVAILABLE}')"
```

Expected output:
```
ChromaDB: True, FAISS: True
```

### 2. Build Vector Index

The first time you use vector RAG, it will automatically:
1. Load documents from `data/RAG/` directory
2. Download the embedding model (`all-MiniLM-L6-v2` - ~80MB)
3. Create embeddings for all text chunks
4. Build and persist the vector index

**First run takes ~2-5 minutes** depending on:
- Number of documents
- Embedding model download speed
- CPU speed

Subsequent runs use the persisted index (instant startup).

### 3. Test Vector RAG

Create a test script `test_vector_rag.py`:

```python
from rag_vector import VectorRAGEngine

# Initialize with ChromaDB
engine = VectorRAGEngine(backend="chromadb")

# Test query
results = engine.retrieve("functional safety requirements", top_k=3)

# Display results
for i, result in enumerate(results, 1):
    print(f"\n--- Result {i} (score: {result['score']:.3f}) ---")
    print(f"Source: {result['source']} ({result['category']})")
    print(f"Text: {result['text'][:200]}...")
```

Run:
```powershell
python test_vector_rag.py
```

## Usage in WHALE-RE

### UI Usage

1. **Start the app**:
   ```powershell
   streamlit run app.py
   ```

2. **Enable RAG** on any agent page (SYS.1, SYS.2, Review, SYS.5, or Manager)

3. **Select backend** from dropdown (currently needs to be added to UI):
   - Simple: Keyword matching (default, no vector DB needed)
   - ChromaDB: Semantic search with persistence
   - FAISS: High-performance search

### Programmatic Usage

```python
from orchestrator import run_single

# Use ChromaDB
result = run_single(
    "SYS1",
    "The vehicle shall maintain cruise control speed",
    use_rag=True,
    rag_backend="chromadb"
)

# Use FAISS
result = run_single(
    "SYS2",
    sys1_output,
    use_rag=True,
    rag_backend="faiss"
)

# Use simple keyword-based RAG
result = run_single(
    "Review",
    sys2_output,
    use_rag=True,
    rag_backend="simple"
)
```

## Configuration

### Embedding Models

Change the embedding model by modifying `rag_vector.py`:

```python
engine = VectorRAGEngine(
    backend="chromadb",
    embedding_model="all-MiniLM-L6-v2"  # Default: fast, good quality
)
```

**Alternative models:**
- `all-mpnet-base-v2`: Higher quality, slower (420MB)
- `paraphrase-MiniLM-L6-v2`: Similar to all-MiniLM, optimized for paraphrases
- `multi-qa-MiniLM-L6-cos-v1`: Optimized for question-answering

### Chunk Size

Adjust in `VectorRAGEngine` initialization:

```python
engine = VectorRAGEngine(
    chunk_size=1000,    # Characters per chunk (default: 1000)
    chunk_overlap=200   # Overlap between chunks (default: 200)
)
```

### Storage Location

Vector databases are stored in `data/vector_store/`:
- ChromaDB: `data/vector_store/chromadb/`
- FAISS: `data/vector_store/faiss.index` + `faiss_metadata.json`

## Performance Comparison

| Backend | Index Creation | Query Speed | Storage | Setup Complexity |
|---------|---------------|-------------|---------|------------------|
| Simple  | Instant       | ~50ms       | None    | Easy             |
| ChromaDB| 2-5 min       | ~20ms       | ~5-10MB | Easy             |
| FAISS   | 2-5 min       | ~5ms        | ~3-5MB  | Medium           |

**Recommendation**: Start with **ChromaDB** for best balance of ease-of-use and performance.

## Troubleshooting

### ChromaDB Installation Issues

If you see errors about SQLite version:

```powershell
pip install --upgrade pysqlite3-binary
```

### FAISS Import Errors

On Windows, if FAISS won't import:

```powershell
# Install Microsoft Visual C++ Redistributable
# Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe
```

### Out of Memory

If embedding creation fails with OOM:

1. Reduce chunk size:
   ```python
   engine = VectorRAGEngine(chunk_size=500)
   ```

2. Process documents in batches (modify `_index_chunks()` in `rag_vector.py`)

3. Use a smaller embedding model:
   ```python
   engine = VectorRAGEngine(embedding_model="all-MiniLM-L6-v2")  # Smallest, fastest
   ```

### Slow First Query

First query after startup is slow (~2-5 seconds) due to model loading. Subsequent queries are fast (<50ms).

To warm up the model on startup, add to `rag_vector.py`:

```python
# After engine initialization
engine.retrieve("warm up query", top_k=1)  # Dummy query
```

## Rebuilding Index

### Full Rebuild

Delete the vector store and restart:

```powershell
# ChromaDB
Remove-Item -Recurse -Force data\vector_store\chromadb

# FAISS
Remove-Item data\vector_store\faiss.index
Remove-Item data\vector_store\faiss_metadata.json
```

Next run will rebuild from scratch.

### Add New Documents

1. Add `.txt` or `.md` files to `data/RAG/<category>/`
2. Rebuild index (see above)
3. Restart app

## Advanced: Custom Similarity Metrics

### ChromaDB

ChromaDB uses cosine similarity by default. To change:

```python
self.collection = self.chroma_client.get_or_create_collection(
    name="whale_re_standards",
    metadata={"hnsw:space": "l2"}  # Options: cosine, l2, ip
)
```

### FAISS

Change index type in `_init_faiss()`:

```python
# Inner product (current default)
self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)

# L2 distance
self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)

# HNSW (faster, approximate)
self.faiss_index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
```

## Migration from Simple to Vector RAG

1. Install dependencies: `pip install chromadb sentence-transformers`
2. Wait for first index build (2-5 minutes)
3. Update UI calls to pass `rag_backend="chromadb"`
4. Test with sample requirements

**No changes to documents needed** - vector RAG reads from same `data/RAG/` directory.

## Best Practices

1. **Use ChromaDB for most cases**: Best balance of ease and performance
2. **Use FAISS for large datasets**: >10k chunks or latency-critical applications
3. **Keep Simple RAG as fallback**: Works without any dependencies
4. **Warm up model on startup**: First query is slow, warm up in background
5. **Monitor index size**: Rebuild if documents change significantly
6. **Filter by category**: Use `category_filter` for targeted retrieval
7. **Tune chunk size**: Larger chunks = more context, slower retrieval

## Monitoring

### Check Index Size

```python
from rag_vector import get_vector_rag_engine

engine = get_vector_rag_engine(backend="chromadb")
print(f"Total chunks: {engine.collection.count()}")
```

### Measure Query Performance

```python
import time

start = time.time()
results = engine.retrieve("test query", top_k=5)
elapsed = time.time() - start
print(f"Query took {elapsed*1000:.1f}ms")
```

### View Storage Size

```powershell
# ChromaDB
Get-ChildItem -Recurse data\vector_store\chromadb | Measure-Object -Property Length -Sum

# FAISS
Get-ChildItem data\vector_store\faiss* | Measure-Object -Property Length -Sum
```

## FAQ

**Q: Which backend should I choose?**
A: Start with ChromaDB. It's easiest to use and performs well for typical workloads.

**Q: Can I use both backends simultaneously?**
A: No, select one backend per session. You can switch between runs.

**Q: How much does vector RAG improve results?**
A: Semantic search (ChromaDB/FAISS) finds conceptually similar content, while simple RAG finds exact keyword matches. Vector RAG typically retrieves more relevant context for complex queries.

**Q: Do I need a GPU?**
A: No, CPU is sufficient. GPU accelerates embedding creation but isn't necessary for <10k documents.

**Q: What happens if vector DB dependencies aren't installed?**
A: App falls back to simple keyword-based RAG automatically. No errors.

**Q: Can I use my own embedding model?**
A: Yes, pass any SentenceTransformers model name to `embedding_model` parameter.

**Q: How do I know which backend is active?**
A: Check console output when RAG initializes: "Initializing Vector RAG Engine with backend: chromadb"

## Next Steps

1. Install dependencies: `pip install -r requirements_rag_vector.txt`
2. Run app: `streamlit run app.py`
3. Enable RAG on any agent page
4. Compare results between simple and vector RAG backends
5. Add more standards documents to `data/RAG/` for richer context

For issues or questions, check the console output for detailed error messages.

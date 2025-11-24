# WHALE-RE – AI Requirements Engineering Co‑pilot

WHALE-RE is a Streamlit application that turns high‑level customer requirements into a fully traced engineering chain: SYS.1 requirements → SYS.2 requirements → Review feedback → SYS.5 test cases. It provides on‑page traceability dashboards, styled exports (CSV/XLSX/DOCX/PDF/REQIF), autosave, and a “Manager” page to run the end‑to‑end pipeline.

The UI is locked to a clean light theme with black text for readability. Per‑agent pages include compact tables, zebra striping, and inline export bars. Traceability visualizations are rendered on the same page (no new tabs).

## Highlights

- Multi‑agent flow with bidirectional traceability:
  - Agent 1: Customer → SYS.1 (SMART, domains, prioritization)
  - Agent 2: SYS.1 → SYS.2 (splitting, technical detailing, verification)
  - Agent 3: Review of SYS.2 (compliance and rewrite hints)
  - Agent 4: SYS.2 → SYS.5 (structured test cases)
- **RAG (Retrieval-Augmented Generation)**: Context-aware generation powered by automotive standards
  - Semantic search through ISO 26262, ASPICE, AUTOSAR, ISO 29148, and 10+ other standards
  - PDF extraction with automatic text parsing and chunking
  - Multiple backends: Simple (keyword), ChromaDB (persistent), FAISS (high-performance)
  - Agent-specific category filtering for targeted context retrieval
  - Improves accuracy, compliance, and standard citations in outputs
  - **UI**: Clear "🔍 Enable RAG Context (Standards-based)" labels on all agent pages
- On‑page traceability dashboards (pies + compact mappings) for Agents 1, 2, and 4
- Uniform, always‑visible export bar: DOCX, XLSX, CSV, PDF; Manager also bundles ZIP and REQIF
- Styled Excel/Word/PDF exports with readable formatting for long text
- Robust pipeline: chunking, deduplication, per‑item retries, autosave to disk
- **Visible save notifications**: ✅ success messages with file paths, ⚠️ warnings on failure
- Broad input support: .txt, .docx, .pdf, .xlsx, .reqif, .json

## Architecture

- UI and Orchestration
  - `app.py`: Streamlit UI, navigation, per‑agent pages, dashboards, exports, autosave
  - `orchestrator.py`: Executes a single agent (`run_single`) or the full pipeline (`run_pipeline`)
    - Uses OpenAI (v1) when `OPENAI_API_KEY` is set
    - Optionally integrates with CrewAI via `crew_pipeline.py` (best‑effort fallback when unavailable)
    - Strategies: input chunking, JSON parsing resilience, deduplication across chunks, per‑item retries for missing outputs
    - **RAG integration**: Augments prompts with context from standards when enabled
  - `agents.py`: Agent prompt templates and JSON output schemas (see "Prompts" below)
  - `utils.py`: Parsing, normalization, exports (CSV/XLSX/DOCX/PDF/REQIF), RTM builder, file loaders

- RAG (Retrieval-Augmented Generation)
  - `rag_engine.py`: Simple keyword-based RAG with PDF extraction support
  - `rag_vector.py`: Vector RAG with ChromaDB/FAISS backends and semantic search
  - `data/RAG/`: Standards library (ISO 26262, ASPICE, AUTOSAR, ISO 29148, ISO 16750, etc.)
  - `data/vector_store/`: Persisted vector indices (created on first run)
  - Supports .txt, .md, and .pdf documents with automatic text extraction

- Data flow (end‑to‑end)
  1) Customer text → Agent 1 → SYS.1 list
  2) SYS.1 list → Agent 2 → SYS.2 list (atomic splitting allowed)
  3) SYS.2 list → Agent 3 → Review entries (compliance, suggestions)
  4) SYS.2 list → Agent 4 → SYS.5 test cases
  5) Manager builds an RTM and exposes consolidated exports

- Dashboards and Traceability
  - Agent 1: Customer ↔ SYS.1 (two pies + compact mapping table)
  - Agent 2: SYS.1 ↔ SYS.2 (traceability pie, optional status pie, coverage badges, compact mapping)
  - Agent 4: SYS.2 ↔ SYS.5 (traceability pie, optional status/priority pie, coverage badges, compact mapping)

## Agents and responsibilities

Prompts live in `agents.py` as `AGENT_PROMPTS` with explicit JSON array schemas per agent.

- Agent 1 – SYS.1 Elicitation (Customer → SYS.1)
  - Ensures SMART requirements (“The system shall …”), domain, priority, rationale, and status
  - Output fields (typical):
    - Traceability: Parent ID (CUST_REQ‑xxx) → Current ID (SYS.1‑xxx) → Next ID (SYS.2‑xxx)
    - Customer Req. ID, Customer Requirement
    - SYS.1 Req. ID, SYS.1 Requirement, Domain, Priority, Rationale, Requirement Status

- Agent 2 – SYS.2 Analysis (SYS.1 → SYS.2)
  - Splits multi‑behavior SYS.1 into atomic SYS.2 items; adds technical metadata
  - Output fields (typical):
    - Traceability: Parent ID (SYS.1‑xxx) → Current ID (SYS.2‑xxx) → Next ID (SYS.5‑xxx)
    - SYS.1 Req. ID/Requirement; SYS.2 Req. ID/Requirement
  - TYPE (Functional/Non-Functional/Information)
  - Interfaces, Timing, Fault Handling, ASIL
  - Verification Level (System Qualification Test (SYS.5) / System Integration Test (SYS.4))
  - Verification Criteria (flat map or sentences), Domain, and Requirement Status

- Agent 3 – SYS.2 Review
  - Reviews each SYS.2 against ISO/IEC/IEEE 29148, ISO 26262, SMART
  - Provides feedback, SMART check, proposed rewrite, compliance check, severity, suggestions
  - App focuses on: SYS.2 Req ID, Review Feedback, Compliance Check, Suggested Improvement (formatted as sentences)

- Agent 4 – SYS.5 Test Case Generation (SYS.2 → SYS.5)
  - Generates 1+ test cases per SYS.2 when multiple acceptance modes/conditions exist
  - Output fields (typical): Traceability (Parent SYS.2 ID), Test Case ID, Description, Preconditions, Test Steps, Expected Result, Pass/Fail Criteria, Test Level, Safety Goal Link

## Prompts (schemas and rules)

See `agents.py` for the full texts; each prompt defines rules and a JSON array schema. At a glance:

- SYS1: SMART phrasing (“The system shall …”), domain from a fixed list, priority (High/Medium/Low), rationale with standard references
- SYS2: Atomic splitting, interfaces/timing/fault handling/ASIL, verification level & criteria, type category; maintain bidirectional traceability
- Review: SMART check dict, compliance (“Yes/Partial/No”), severity, suggested rewrite/improvement
- SYS5: At least one test per SYS.2, number steps, include pass/fail criteria; link Safety Goals if ASIL present

## Tech stack

- Python 3.10+
- Streamlit (UI), Plotly (pies) with Matplotlib fallback
- pandas (tables), openpyxl (Excel), python‑docx (Word), fpdf (PDF), PyPDF2 (PDF text read)
- OpenAI (v1) for LLM calls, optional CrewAI/LangChain orchestration
- **RAG**: pdfplumber/PyPDF2 (PDF extraction), ChromaDB (vector store), FAISS (similarity search), SentenceTransformers (embeddings)

Dependencies are listed in `requirements.txt` (core) and `requirements_rag_vector.txt` (RAG features).

## File structure (key paths)

Top‑level

- `app.py` – main Streamlit app (UI, dashboards, exports, autosave)
- `agents.py` – `AGENT_PROMPTS` for Agents 1–4
- `orchestrator.py` – `run_single`, `run_pipeline`, OpenAI/CrewAI glue, chunking/dedup/retry, RAG integration
- `utils.py` – exports (CSV/XLSX/DOCX/PDF/REQIF), RTM, file importers, normalization helpers
- `crew_pipeline.py` – optional CrewAI pipeline (if available)
- `rag_engine.py` – Simple keyword-based RAG with PDF extraction
- `rag_vector.py` – Vector RAG with ChromaDB/FAISS and semantic search
- `test_pdf_loading.py` – Test script for PDF extraction verification
- `PDF_STANDARDS_GUIDE.md` – Comprehensive guide for using PDF standards with RAG
- `VECTOR_RAG_SETUP.md` – Setup instructions for vector databases and embeddings
- Optional: `.streamlit/config.toml` – Streamlit settings if present (defaults are 8501 unless overridden)
- `requirements.txt`, `.env`

Mirrored package (optional): `whale_re/` contains the same logical modules for packaging.

Runtime outputs and samples

- `data/inputs/` – sample customer requirement files
- `data/outputs/` – autosaved tables for each page
- `data/outputs/agent5/` – Manager page autosaves (SYS1, SYS2, Review, SYS5, RTM)
- `data/RAG/` – Standards library for RAG (ISO 26262, ASPICE, AUTOSAR, ISO 29148, ISO 16750, ISO 14229, ISO 11898, ISO 21434, ECE R100, LV_VDA, etc.)
- `data/vector_store/` – Persisted vector indices for ChromaDB/FAISS (created on first run)
- `static/logo_otl/` – optional logo image displayed in the nav bar

## Setup (Windows PowerShell)

1) Create and activate a virtual environment, then install deps

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2) Configure OpenAI (optional but needed for generation)

Create `.env` in the repo root:

```
OPENAI_API_KEY=sk-...your_key...
```

3) (Optional) Install RAG dependencies for enhanced context-aware generation

```powershell
# For PDF extraction
pip install pdfplumber PyPDF2

# For vector RAG (semantic search)
pip install -r requirements_rag_vector.txt
```

**Note**: RAG is optional but highly recommended for compliance-aware outputs. See `PDF_STANDARDS_GUIDE.md` and `VECTOR_RAG_SETUP.md` for details.

4) Run the app

```powershell
streamlit run app.py
```

## Authentication (Optional Login System)

The app now supports optional user authentication via `streamlit-authenticator`.

### Enable
1. Ensure dependency installed (already in `requirements.txt`).
2. Keep `auth_config.yaml` in the project root (created by default with sample users).

If the file or dependency is missing, the app falls back to guest mode (no authentication) and displays a banner in the sidebar.

### Default Users (Sample Only)
| Username | Password    | Role     |
|----------|------------|----------|
| `alice`  | `alice123` | analyst  |
| `bob`    | `bob12345` | reviewer |
| `manager`| `manage!`  | manager  |

Passwords are stored as bcrypt hashes in `auth_config.yaml`. **Replace these with your own secure credentnials before production usage.**

### Customizing
Edit `auth_config.yaml`:
```yaml
credentials:
  users:
    - username: jane
      name: Jane Doe
      password: <bcrypt-hash>
      email: jane@example.com
      role: analyst
cookie:
  name: whale_auth
  key: change-this-secret
  expiry_days: 7
```
Generate bcrypt password hashes:
```python
import bcrypt
print(bcrypt.hashpw(b"your_password", bcrypt.gensalt()).decode())
```

### Session Behavior
- Successful login shows user badge + Logout button in sidebar.
- Invalid credentials stop page rendering until corrected.
- When auth disabled (missing config/dependency), a guest label is shown.

### Notes
- Roles are informational; extend by gating pages (e.g., restrict Manager to `role == 'manager'`).
- Never commit real production credentials; exclude `auth_config.yaml` from version control if sensitive.

By default, Streamlit serves on port 8501 unless overridden by config or CLI flags. If the port is busy or you prefer another, run:

```powershell
streamlit run app.py --server.port 8510
```

Then open the shown Local URL.

## Usage (per page)

General tips

- Uploaders accept: .txt, .docx, .pdf, .xlsx, .reqif, .json
- Large inputs are chunked automatically; outputs are deduplicated and retried per item when needed
- Every results table has a white background, black text, sticky header, zebra rows, and an inline "Export Options" bar (DOCX/XLSX/CSV/PDF)
- **Enable RAG**: Check "Enable RAG" on agent pages to inject context from automotive standards (ISO 26262, ASPICE, AUTOSAR, etc.) for more accurate, compliance-aware outputs

Home

- Overview cards for each agent with statuses and a Manager workflow hint

Agent 1 – SYS.1

1) Upload customer requirements (or paste text) and click “Generate SYS.1 Requirements”
2) View the SYS.1 table and export as needed
3) At the bottom, see “Traceability and Status Dashboard”
   - Pie 1: Traceability Status (Traced vs Not Traced)
   - Pie 2: Overall Status (Approved/Draft/Rejected)
   - Compact mapping: CUST_REQ → SYS.1 (zebra striping; status tints)

Agent 2 – SYS.2

1) Upload or paste a JSON array of SYS.1 items; click “Generate SYS.2 Requirements”
2) The table shows formatted “Verification Criteria” sentences and exports
3) Below the table: SYS.1 → SYS.2 traceability dashboard
   - Traceability pie (Traced vs Not Traced)
   - Optional overall status pie (when a status‑like column exists)
   - Coverage badges (Traced, Total, Coverage %)
   - Compact mapping: SYS.1 Req → SYS.2 Req(s)

Agent 3 – Review

1) Upload or paste SYS.2 items; click “Generate Review Feedback”
2) Feedback, compliance, and suggestions are converted into readable sentences
3) Export via the inline bar; autosave is performed

Agent 4 – SYS.5

1) Upload or paste a JSON array of SYS.2 items; click “Generate Test Cases”
2) The table enumerates and numbers test steps automatically
3) Below the table: SYS.2 → SYS.5 traceability dashboard
   - Traceability pie and optional overall status/priority pie
   - Coverage badges (Traced, Total, Coverage %)
   - Compact mapping: SYS.2 Req → Test Case ID(s)

Manager – Orchestration & RTM

- Paste customer requirements and run the full pipeline (SYS.1 → SYS.2 → Review → SYS.5)
- Manager shows metrics, merged SYS.2+Review, conflict detection for duplicate SYS.2 IDs, optional approval gating, and RTM
- Sidebar exports: individual CSVs, styled RTM (Excel/Word/PDF), REQIF, and a ZIP bundle of all tables

## Screenshots

Add screenshots under `docs/screenshots/` and update the paths below.

![Home Dashboard](docs/screenshots/home.png)
![Agent 1 - SYS.1](docs/screenshots/agent1_sys1.png)
![Agent 2 - SYS.2](docs/screenshots/agent2_sys2.png)
![Agent 3 - Review](docs/screenshots/agent3_review.png)
![Agent 4 - SYS.5](docs/screenshots/agent4_sys5.png)
![Manager - RTM](docs/screenshots/manager_rtm.png)

Tips:
- Use 1366×768 or higher for consistent layout
- Prefer light theme captures (the app enforces a white theme)
- Crop to focus on the table, export bar, and traceability pies

## Architecture Diagram

Simple ASCII sketch for quick orientation:

```
┌──────────────────────────┐
│        Customer          │
│   Requirements (text)    │
└────────────┬─────────────┘
       │ Agent 1 (SYS.1)
       ▼
       ┌───────────────┐
       │   SYS.1 list  │
       └───────┬───────┘
         │ Agent 2 (SYS.2)
         ▼
       ┌───────────────┐           ┌──────────────────┐
       │   SYS.2 list  │──────────▶│  Agent 3 Review  │
       └───────┬───────┘           └──────────────────┘
         │ Agent 4 (SYS.5)
         ▼
       ┌──────────────────┐
       │  SYS.5 testcases │
       └─────────┬────────┘
     │
     ▼
  ┌───────────────────┐
  │   Manager (RTM)   │
  │  Exports & ZIP    │
  └───────────────────┘
```

Optionally include a draw.io diagram and export a PNG alongside it:

- Source: `docs/whale-re-architecture.drawio`
- Exported: `docs/screenshots/architecture.png`

Reference it in the README:

![Architecture](docs/screenshots/architecture.png)

## Exports and autosave

- Per‑page export bar: DOCX, XLSX, CSV, PDF
- Manager sidebar: CSVs, styled RTM (XLSX/DOCX/PDF), REQIF, ZIP bundle
- Autosave locations (created if needed):
  - SYS.1 → `data/outputs/sys1_requirements.xlsx`
  - SYS.2 → `data/outputs/sys2_requirements.xlsx`
  - Review → `data/outputs/sys2_requirements_reviewed.xlsx`
  - SYS.5 → `data/outputs/sys.5_test_cases.xlsx`
  - Manager → `data/outputs/agent5/` (SYS1, SYS2, Review, SYS5, RTM)

## Troubleshooting

- Port already in use
  - Start on another port:
    ```powershell
    streamlit run app.py --server.port 8510
    ```

- No outputs generated / “No LLM backend available”
  - Set `OPENAI_API_KEY` in `.env` and restart

- Long documents produce partial outputs
  - The orchestrator chunks inputs and deduplicates; if some items are still missing, it retries per item (especially for SYS.2 and SYS.5)

- PDF/DOCX import yields weak text
  - PDF text extraction depends on document quality; consider providing .txt/.xlsx for best fidelity

- Charts don't show
  - Plotly is preferred; Matplotlib is a fallback. Ensure both are installed from `requirements.txt`

- RAG not improving outputs / "No text extracted from PDF"
  - Install PDF libraries: `pip install pdfplumber PyPDF2`
  - Test extraction: `python test_pdf_loading.py`
  - Verify PDFs in `data/RAG/` are text-based (not scanned images)
  - For vector RAG: Install dependencies `pip install -r requirements_rag_vector.txt`
  - First run builds embeddings (2-5 minutes); subsequent runs load cached index
  - See `PDF_STANDARDS_GUIDE.md` for detailed troubleshooting

- Slow first run with RAG enabled
  - Normal behavior: building vector embeddings from PDFs takes time
  - Use simple RAG (keyword-based) for faster initialization
  - Or process PDFs in smaller batches

## Privacy & networking

- When `OPENAI_API_KEY` is configured, the app will call the OpenAI API to generate outputs
- Without an API key, the app still runs but shows a placeholder error object for agent calls
- No other outbound calls are made by the app itself

## Development notes

- The UI enforces a permanent light theme with black text
- Traceability dashboards render on the same page below each results table; pies have white backgrounds and inside‑percent labels
- Tables are static HTML for guaranteed white background and sticky headers
- Verification criteria and review sentences are auto‑formatted for readability

## License

Internal/Project use. If you intend to open‑source, add your preferred license here.


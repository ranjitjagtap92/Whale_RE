## WHALE-RE: Copilot Instructions for AI Coding Agents

Purpose: Make you productive fast in this codebase. Focus on real patterns, concrete commands, and how components work together.

### Big Picture
- UI: `app.py` is a Streamlit app with pages for Agent 1 (SYS.1), Agent 2 (SYS.2), Agent 3 (Review), Agent 4 (SYS.5), and Manager (pipeline + RTM). Tables render as static white HTML for visual consistency; each page has an inline export bar (DOCX/XLSX/CSV/PDF) and a traceability dashboard rendered on the same page.
- Orchestration: `orchestrator.py` exposes `run_single(agent_key, input_text)` and `run_pipeline(customer_requirements)`.
  - Uses OpenAI v1 (`openai.OpenAI`) when `OPENAI_API_KEY` is set; otherwise returns an error object but keeps the app running.
  - Optional CrewAI path in `crew_pipeline.py` (preferred when key is present). Both paths expect/produce JSON arrays per agent.
- Prompts & Schemas: `agents.py` defines `AGENT_PROMPTS` for `SYS1`, `SYS2`, `Review`, `SYS5`. Each prompt requires a JSON array with specific fields and traceability links.
- Utilities: `utils.py` handles parsing, field normalization, exports (CSV/XLSX/DOCX/PDF/REQIF/ZIP), and RTM composition.

### Data Flow and Conventions
- End-to-end: Customer text → SYS.1 list → SYS.2 list → Review entries (on SYS.2) → SYS.5 test cases. Manager merges these into an RTM.
- Canonical fields (used across parsing, dedup, merges):
  - `Customer Req. ID`, `Customer Requirement`
  - `SYS.1 Req. ID`, `SYS.1 Requirement`
  - `SYS.2 Req. ID`, `SYS.2 Requirement`
  - `Test Case ID`, `Test Steps`, `Expected Result`, `Pass/Fail Criteria`
- Traceability object (typical): `{"Parent ID": "...", "Current ID": "...", "Next ID": "..."}`.
- Normalization: `utils.normalize_df_for_agent/normalize_records_for_agent` map varied headers to canonical names. Prefer these before merging or exporting.
- Chunking & Dedup:
  - `run_single` splits large inputs into batches (size 10). SYS.1 uses text blocks; others use JSON array chunks.
  - Per-agent retries for missing items (SYS.2 and SYS.5) ensure coverage when the LLM skips rows.
  - Dedup keys favor IDs: for SYS.5 rows it uses (`SYS.2 Req. ID`, `Test Case ID`/Description); otherwise `SYS.2 Req. ID` → `SYS.1 Req. ID` → `Customer Req. ID`.
- File inputs: `utils.load_file_for_agent(...)` adapts per agent:
  - SYS.1 → returns concatenated text (from .txt/.docx/.pdf/.xlsx/.reqif).
  - SYS.2/Review/SYS.5 → return records (list of dicts) when possible; otherwise text. Headers are normalized.
  - For SYS.5, `run_single` also accepts wrappers like `{ "SYS2": [ ... ] }` and will unwrap.

### UI Patterns (app.py)
- Always-on light theme with white backgrounds; pies via Plotly if available (Matplotlib fallback). Traceability dashboard lives below each table.
- Export bar helper: `render_export_buttons(df, base_name)` produces DOCX/XLSX/CSV/PDF bytes via `utils`.
- Static table renderer: `render_static_white_table(df, ...)` uses `components.html` to enforce white, sticky headers, and vertical scroll.
- RAG UI: Each agent page has `st.write("🔍 **Enable RAG Context** (Standards-based)")` label followed by checkbox with `label_visibility="visible"`.
- Save notifications: `st.success(f"✅ Auto-saved to: {path}")` after successful save, `st.warning(f"⚠️ Auto-save failed: {e}")` on errors.
- Session state keys commonly used: `sys1_table_df` (and analogous per page) to drive the dashboards; `sys1_use_rag` etc. for RAG toggle state.

### Developer Workflows
- Setup (Windows PowerShell):
  - `python -m venv .venv`
  - `. .\.venv\Scripts\Activate.ps1`
  - `pip install -r requirements.txt`
  - Optional: create `.env` with `OPENAI_API_KEY=sk-...` and optionally `OPENAI_MODEL=gpt-4o`.
- Run the app:
  - VS Code task: `Run Streamlit app` (uses `.venv` interpreter, headless, port 8501).
  - Or from shell: `streamlit run app.py` (use `--server.port 8510` to change port).
- Quick validation: `python smoke_test.py` checks exporters and orchestrator behavior without an API key.

### Extending/Modifying Agents
- Add/adjust a schema: edit `AGENT_PROMPTS` in `agents.py`. Keep output as a JSON array with canonical fields to play well with normalization and dedup.
- New agent or field changes that affect joins:
  - Update dedup logic in `orchestrator.run_single` if the new rows need custom keys.
  - Update `utils._canonical_map_for_agent` to normalize new/aliased columns.
  - Update export/RTM builders if new relationships are introduced.
- Example: a stricter SYS.2 split rule → ensure unique `SYS.2 Req. ID` (e.g., `SYS.2-001a`, `SYS.2-001b`) and update any mapping in `utils` if new fields are added.

### Exports, Autosave, and Manager
- Exports: per page export bar for DOCX/XLSX/CSV/PDF; Manager sidebar can export individual CSVs, styled RTM (XLSX/DOCX/PDF), REQIF, and a ZIP bundle via `utils.export_all_as_zip`.
- Autosave locations (created on demand):
  - SYS.1 → `data/outputs/sys1_requirements.xlsx`
  - SYS.2 → `data/outputs/sys2_requirements.xlsx`
  - Review → `data/outputs/sys2_requirements_reviewed.xlsx`
  - SYS.5 → `data/outputs/sys.5_test_cases.xlsx`
  - Manager bundle → `data/outputs/agent5/`
- Autosave notifications: Each agent displays `st.success()` with ✅ and file path after successful save; Manager shows expandable list with all saved files. Failed saves trigger `st.warning()` with error details.

### External Integrations
- OpenAI v1 client in `orchestrator.py` (`model="gpt-4o"` default). Returns JSON-as-text; robust parsing will extract fenced JSON if needed.
- CrewAI path (`crew_pipeline.py`) using `langchain_openai.ChatOpenAI` and `Crew/Agent/Task`. Enabled only when `OPENAI_API_KEY` is set.

### Pointers to Key Files
- `app.py` – UI, dashboards, page export bars, autosave to `data/outputs`.
- `orchestrator.py` – `run_single` (chunk, retry, dedup), `run_pipeline`.
- `agents.py` – Prompt templates and required JSON schemas.
- `utils.py` – Parsing, normalization, exports, RTM, file importers.
- `crew_pipeline.py` – Optional CrewAI execution.
- `smoke_test.py` – Minimal end-to-end exporter and orchestrator checks.

If anything here doesn’t match what you see in the code (e.g., field names or page behaviors), point it out and I’ll reconcile the instructions with the current implementation.

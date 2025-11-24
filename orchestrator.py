import os
import json
import re
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
from agents import AGENT_PROMPTS
from crew_pipeline import run_crewai_single, run_crewai_pipeline

# RAG integration - Support both simple and vector backends
try:
    from rag_engine import augment_prompt_with_rag
    RAG_SIMPLE_AVAILABLE = True
except Exception:
    RAG_SIMPLE_AVAILABLE = False

try:
    from rag_vector import augment_prompt_with_vector_rag, CHROMADB_AVAILABLE, FAISS_AVAILABLE
    RAG_VECTOR_AVAILABLE = True
except Exception:
    RAG_VECTOR_AVAILABLE = False
    CHROMADB_AVAILABLE = False
    FAISS_AVAILABLE = False

# Fallback if no RAG available
if not RAG_SIMPLE_AVAILABLE and not RAG_VECTOR_AVAILABLE:
    def augment_prompt_with_rag(agent_key, prompt, input_text, enabled=True):
        return prompt
    def augment_prompt_with_vector_rag(agent_key, prompt, input_text, backend="chromadb"):
        return prompt

# OpenAI v1 client (preferred)
try:
    from openai import OpenAI
    _openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception:
    _openai_client = None

def call_llm(prompt: str, system_message: Optional[str] = None, model: str = "gpt-4o", temperature: float = 0.0, max_tokens: int = 3500) -> str:
    """Call an LLM using OpenAI if configured. Returns a string (expected to be JSON by callers)."""
    if _openai_client is not None:
        try:
            resp = _openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message or "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or "[]"
        except Exception as e:
            return json.dumps([{"error": f"OpenAI call failed: {e}"}])

    return json.dumps([{"error": "No LLM backend available. Set OPENAI_API_KEY in a .env file."}])


def run_single(
    agent_key: str,
    input_text: str,
    use_rag: bool = False,
    rag_backend: str = "simple"
) -> Dict[str, Any]:
    """Run a single agent using CrewAI if configured; fallback to direct OpenAI client else error-wrapped.

    Also supports chunked processing to avoid output truncation for large inputs (no row limits).
    
    Args:
        agent_key: Agent identifier (SYS1, SYS2, Review, SYS5)
        input_text: Input text or JSON for the agent
        use_rag: Enable RAG context augmentation from standards documents
        rag_backend: RAG backend to use ('simple', 'chromadb', 'faiss')
    """

    # Helpers ---------------------------------------------------------------
    def _try_parse(s: str):
        try:
            return json.loads(s)
        except Exception:
            return None

    def _parse_out_to_list(raw: str) -> List[Dict[str, Any]]:
        # 1) direct
        parsed = _try_parse(raw)
        if parsed is None:
            # 2) from code fences
            fence_match = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.DOTALL | re.IGNORECASE)
            if fence_match:
                inner = fence_match.group(1).strip()
                parsed = _try_parse(inner)
            if parsed is None:
                # 3) slice first JSON span
                first_obj = raw.find('{'); last_obj = raw.rfind('}')
                first_arr = raw.find('['); last_arr = raw.rfind(']')
                candidates: List[str] = []
                if first_obj != -1 and last_obj > first_obj:
                    candidates.append(raw[first_obj:last_obj+1])
                if first_arr != -1 and last_arr > first_arr:
                    candidates.append(raw[first_arr:last_arr+1])
                for cand in candidates:
                    parsed = _try_parse(cand)
                    if parsed is not None:
                        break
        if parsed is None:
            return [{"Raw": raw}]
        if isinstance(parsed, list):
            # ensure list of dicts
            return [x if isinstance(x, dict) else {"Raw": str(x)} for x in parsed]
        if isinstance(parsed, dict):
            return [parsed]
        return [{"Raw": str(parsed)}]

    def _split_cust_reqs(text: str) -> List[str]:
        # Split by requirement IDs like CUST_REQ-001 or CUST-REQ-001
        lines = text.splitlines()
        blocks: List[str] = []
        current: List[str] = []
        id_re = re.compile(r"^\s*(CUST[\-_ ]?REQ[\-_ ]?\d+)", flags=re.IGNORECASE)
        for ln in lines:
            if id_re.match(ln) and current:
                blocks.append("\n".join(current).strip())
                current = [ln]
            else:
                current.append(ln)
        if current:
            blocks.append("\n".join(current).strip())
        # Fallback if only one block detected
        if len(blocks) <= 1:
            # Split on double newlines as paragraphs
            paras = [p.strip() for p in text.split("\n\n") if p.strip()]
            return paras if paras else [text]
        return blocks

    def _chunk_list(lst: List[Any], n: int) -> List[List[Any]]:
        return [lst[i:i+n] for i in range(0, len(lst), n)]

    def _dedup(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        out: List[Dict[str, Any]] = []
        def _get_any(d: Dict[str, Any], keys: List[str]) -> Optional[str]:
            for k in keys:
                if k in d and d[k] not in (None, ""):
                    try:
                        return str(d[k]).strip()
                    except Exception:
                        return str(d[k])
            return None

        def _key(d: Dict[str, Any]):
            # Detect SYS.5-style rows (test case fields present)
            is_tc_row = any(k in d for k in (
                "Test Case ID", "TC ID", "Test_ID", "Test Steps", "Preconditions", "Expected Result", "Pass/Fail Criteria"
            ))

            # Alias sets for robustness across agents
            sys1_id = _get_any(d, ["SYS.1 Req. ID", "SYS1 Req ID", "SYS_1 Req ID", "SYS.1 Requirement ID"]) or None
            sys1_text = _get_any(d, ["SYS.1 Requirement", "SYS1 Requirement"]) or ""

            sys2_id = _get_any(d, ["SYS.2 Req. ID", "SYS.2 Req ID", "SYS2 Req ID", "SYS_2 Req ID"]) or None
            sys2_text = _get_any(d, ["SYS.2 Requirement", "SYS2 Requirement"]) or ""

            tc_id = _get_any(d, ["Test Case ID", "TC ID", "Test_ID"]) or None
            tc_desc = _get_any(d, ["Description", "Test Description"]) or ""

            # Composite preferences to avoid cross-chunk collisions:
            # - For SYS.5 rows (usually have SYS.2 Req ID + Test Case info)
            if is_tc_row and sys2_id and (tc_id or tc_desc):
                return ("SYS5", sys2_id, tc_id or "", tc_desc)
            # If looks like TC row but lacks SYS.2 ID, fall back to TC info to keep multiple TCs
            if is_tc_row and tc_id:
                return ("TC", tc_id, tc_desc)

            # Prefer source IDs for non-TC rows first to avoid collisions when generated IDs reset per chunk
            cust_id = _get_any(d, ["Customer Req. ID", "Customer Req_ID", "Customer Requirement ID", "Customer ID"])  # alias-friendly
            if cust_id and not is_tc_row:
                return ("CUST", cust_id)

            # - For SYS.2 rows
            if sys2_id:
                return ("SYS2", sys2_id, sys2_text)

            # - For SYS.1 rows
            if sys1_id:
                return ("SYS1", sys1_id, sys1_text)

            # - Fallback on Test Case info alone (still include description to reduce collisions) for non-detected TC rows
            if not is_tc_row and tc_id:
                return ("TC", tc_id, tc_desc)
            # Compound fallback using a hash of sorted items (stable-ish)
            try:
                return ("hash", json.dumps(d, sort_keys=True))
            except Exception:
                return ("idx", str(len(out)))
        for d in items:
            if not isinstance(d, dict):
                out.append({"Raw": str(d)})
                continue
            k = _key(d)
            if k in seen:
                continue
            seen.add(k)
            out.append(d)
        return out

    # Chunking strategy -----------------------------------------------------
    CHUNK_SIZE = 10
    aggregated: List[Dict[str, Any]] = []

    # Augment prompt with RAG context if enabled
    base_prompt = AGENT_PROMPTS[agent_key]
    if use_rag:
        if rag_backend in ("chromadb", "faiss") and RAG_VECTOR_AVAILABLE:
            # Use vector RAG
            if rag_backend == "chromadb" and not CHROMADB_AVAILABLE:
                print("ChromaDB not available, falling back to simple RAG")
                base_prompt = augment_prompt_with_rag(agent_key, base_prompt, input_text) if RAG_SIMPLE_AVAILABLE else base_prompt
            elif rag_backend == "faiss" and not FAISS_AVAILABLE:
                print("FAISS not available, falling back to simple RAG")
                base_prompt = augment_prompt_with_rag(agent_key, base_prompt, input_text) if RAG_SIMPLE_AVAILABLE else base_prompt
            else:
                base_prompt = augment_prompt_with_vector_rag(agent_key, base_prompt, input_text, backend=rag_backend)
        elif rag_backend == "simple" and RAG_SIMPLE_AVAILABLE:
            # Use simple keyword-based RAG
            base_prompt = augment_prompt_with_rag(agent_key, base_prompt, input_text)
        else:
            print(f"RAG backend '{rag_backend}' not available")

    # For SYS1 with raw text input, split into per-requirement blocks
    if agent_key == "SYS1" and isinstance(input_text, str):
        blocks = _split_cust_reqs(input_text)
        batches = _chunk_list(blocks, CHUNK_SIZE)
        for batch in batches:
            batch_text = "\n\n".join(batch)
            if OPENAI_API_KEY:
                try:
                    part = run_crewai_single(agent_key, batch_text).get(agent_key, [])
                except Exception:
                    part = _parse_out_to_list(call_llm(base_prompt + "\n\nINPUT:\n" + batch_text))
            else:
                part = _parse_out_to_list(call_llm(base_prompt + "\n\nINPUT:\n" + batch_text))
            if isinstance(part, list):
                aggregated.extend(part)
        return {agent_key: _dedup(aggregated)}

    # For other agents, if input is a large JSON array, chunk it
    try:
        maybe_json = json.loads(input_text)
    except Exception:
        maybe_json = None

    # If a wrapped object is provided (e.g., {"SYS2": [...]}) for SYS5, unwrap to the list
    if agent_key == "SYS5" and isinstance(maybe_json, dict):
        inner = None
        for k in ("SYS2", "SYS.2", "sys2", "sys_2"):
            if k in maybe_json and isinstance(maybe_json[k], list):
                inner = maybe_json[k]
                break
        if inner is not None:
            maybe_json = inner

    if isinstance(maybe_json, list) and agent_key in ("SYS2", "Review", "SYS5"):
        # Process in chunks for all three agents. For SYS5 specifically, ensure coverage per input item
        # by retrying missing SYS.2 IDs individually if the batch output skips any.
        for chunk in _chunk_list(maybe_json, CHUNK_SIZE if len(maybe_json) > CHUNK_SIZE else len(maybe_json)):
            chunk_text = json.dumps(chunk)
            if OPENAI_API_KEY:
                try:
                    part = run_crewai_single(agent_key, chunk_text).get(agent_key, [])
                except Exception:
                    part = _parse_out_to_list(call_llm(base_prompt + "\n\nINPUT:\n" + chunk_text))
            else:
                part = _parse_out_to_list(call_llm(base_prompt + "\n\nINPUT:\n" + chunk_text))
            if isinstance(part, list):
                aggregated.extend(part)

            # Additional coverage for SYS2: retry any missing SYS.1 items per item
            if agent_key == "SYS2":
                def _get_any_local(d: Dict[str, Any], keys: List[str]) -> Optional[str]:
                    for k in keys:
                        if isinstance(d, dict) and k in d and d[k] not in (None, ""):
                            try:
                                return str(d[k]).strip()
                            except Exception:
                                return str(d[k])
                    return None

                out_ref_ids = set()
                if isinstance(part, list):
                    for r in part:
                        if isinstance(r, dict):
                            ref = _get_any_local(r, ["SYS.1 Req. ID", "SYS1 Req ID", "SYS_1 Req ID", "SYS.1 Requirement ID"]) or _get_any_local(r, ["SYS.1 Requirement", "SYS1 Requirement"]) or None
                            if ref:
                                out_ref_ids.add(ref)
                for item in chunk:
                    in_ref = _get_any_local(item if isinstance(item, dict) else {}, ["SYS.1 Req. ID", "SYS1 Req ID", "SYS_1 Req ID", "SYS.1 Requirement ID"]) or _get_any_local(item if isinstance(item, dict) else {}, ["SYS.1 Requirement", "SYS1 Requirement"]) or None
                    if in_ref and in_ref not in out_ref_ids:
                        single_text = json.dumps([item]) if isinstance(item, dict) else json.dumps([item])
                        if OPENAI_API_KEY:
                            try:
                                single_part = run_crewai_single(agent_key, single_text).get(agent_key, [])
                            except Exception:
                                single_part = _parse_out_to_list(call_llm(base_prompt + "\n\nINPUT:\n" + single_text))
                        else:
                            single_part = _parse_out_to_list(call_llm(base_prompt + "\n\nINPUT:\n" + single_text))
                        if isinstance(single_part, list):
                            aggregated.extend(single_part)

            # Additional coverage for SYS5: retry any missing SYS.2 IDs per item
            if agent_key == "SYS5":
                def _get_any_local(d: Dict[str, Any], keys: List[str]) -> Optional[str]:
                    for k in keys:
                        if isinstance(d, dict) and k in d and d[k] not in (None, ""):
                            try:
                                return str(d[k]).strip()
                            except Exception:
                                return str(d[k])
                    return None

                out_sys2_ids = set()
                if isinstance(part, list):
                    for r in part:
                        if isinstance(r, dict):
                            sid = _get_any_local(r, ["SYS.2 Req. ID", "SYS.2 Req ID", "SYS2 Req ID", "SYS_2 Req ID"])
                            if sid:
                                out_sys2_ids.add(sid)
                for item in chunk:
                    in_sid = _get_any_local(item if isinstance(item, dict) else {}, ["SYS.2 Req. ID", "SYS.2 Req ID", "SYS2 Req ID", "SYS_2 Req ID"])
                    if in_sid and in_sid not in out_sys2_ids:
                        single_text = json.dumps([item]) if isinstance(item, dict) else json.dumps([item])
                        if OPENAI_API_KEY:
                            try:
                                single_part = run_crewai_single(agent_key, single_text).get(agent_key, [])
                            except Exception:
                                single_part = _parse_out_to_list(call_llm(base_prompt + "\n\nINPUT:\n" + single_text))
                        else:
                            single_part = _parse_out_to_list(call_llm(base_prompt + "\n\nINPUT:\n" + single_text))
                        if isinstance(single_part, list):
                            aggregated.extend(single_part)
        return {agent_key: _dedup(aggregated)}
        
    # SYS5 post-processing when input was list but bypassed chunk path (unlikely) or after single-shot
    if agent_key == "SYS5" and isinstance(maybe_json, list):
        def _get_any_local(d: Dict[str, Any], keys: List[str]) -> Optional[str]:
            for k in keys:
                if isinstance(d, dict) and k in d and d[k] not in (None, ""):
                    try:
                        return str(d[k]).strip()
                    except Exception:
                        return str(d[k])
            return None
        input_sys2_ids = []
        for item in maybe_json:
            sid = _get_any_local(item if isinstance(item, dict) else {}, ["SYS.2 Req. ID", "SYS.2 Req ID", "SYS2 Req ID", "SYS_2 Req ID"])
            if sid:
                input_sys2_ids.append(sid)
        output_by_sid: Dict[str, List[Dict[str, Any]]] = {}
        for row in aggregated:
            sid = _get_any_local(row, ["SYS.2 Req. ID", "SYS.2 Req ID", "SYS2 Req ID", "SYS_2 Req ID"]) or ""
            if sid:
                output_by_sid.setdefault(sid, []).append(row)
        # Ensure at least one test case per SYS.2 ID
        for sid in input_sys2_ids:
            if sid not in output_by_sid:
                aggregated.append({
                    "Traceability": {"Parent ID": sid, "Current ID": f"{sid}-TC-PLACEHOLDER"},
                    "SYS.2 Req. ID": sid,
                    "Test Case ID": f"TC-{sid}-01",
                    "Description": "Placeholder test case - generation failed, manual authoring required.",
                    "Preconditions": "TBD",
                    "Test Steps": ["TBD"],
                    "Expected Result": "TBD",
                    "Pass/Fail Criteria": "TBD",
                    "Test Level": "System",
                    "Safety Goal Link": f"SG-{sid}",
                    "Generation Status": "Placeholder"
                })
        # Normalize Test Case IDs and coerce steps / safety goal
        counters: Dict[str, int] = {}
        seen_ids: set = set()
        for row in aggregated:
            sid = _get_any_local(row, ["SYS.2 Req. ID", "SYS.2 Req ID", "SYS2 Req ID", "SYS_2 Req ID"]) or "UNASSIGNED"
            # Coerce Test Steps to list
            steps = row.get("Test Steps")
            if isinstance(steps, str):
                parts = [p.strip() for p in re.split(r"[\n;]+", steps) if p.strip()]
                row["Test Steps"] = parts if parts else [steps.strip() or "TBD"]
            elif not isinstance(steps, list):
                row["Test Steps"] = ["TBD"]
            # Safety Goal Link auto-fill if ASIL present
            if row.get("ASIL") and not row.get("Safety Goal Link"):
                row["Safety Goal Link"] = f"SG-{sid}"
            # Assign deterministic TC IDs
            tcid = row.get("Test Case ID") or ""
            if tcid in ("", None) or tcid in seen_ids:
                counters[sid] = counters.get(sid, 0) + 1
                new_id = f"TC-{sid}-{counters[sid]:02d}"
                row["Test Case ID"] = new_id
                seen_ids.add(new_id)
            else:
                seen_ids.add(tcid)
            # Pass/Fail Criteria fallback
            if not row.get("Pass/Fail Criteria"):
                row["Pass/Fail Criteria"] = "TBD"
        return {agent_key: _dedup(aggregated)}

    # Default single-shot path ---------------------------------------------
    if OPENAI_API_KEY:
        try:
            return run_crewai_single(agent_key, input_text)
        except Exception:
            pass
    prompt_template = AGENT_PROMPTS.get(agent_key)
    if not prompt_template:
        return {agent_key: json.dumps([{"error": "Unknown agent key"}])}
    prompt = prompt_template + "\n\nINPUT:\n" + input_text
    out = call_llm(prompt)
    return {agent_key: _parse_out_to_list(out)}


def run_pipeline(
    customer_requirements: str,
    use_rag: bool = False,
    rag_backend: str = "simple"
) -> Dict[str, Any]:
    """Run all agents in sequence via CrewAI when available; fallback to direct orchestrator otherwise.
    
    Args:
        customer_requirements: Initial customer requirements text
        use_rag: Enable RAG context augmentation for all agents
        rag_backend: RAG backend to use ('simple', 'chromadb', 'faiss')
    """
    if OPENAI_API_KEY:
        try:
            return run_crewai_pipeline(customer_requirements)
        except Exception:
            pass
    outputs: Dict[str, Any] = {}
    # Agent 1: SYS1
    o1 = run_single("SYS1", customer_requirements, use_rag=use_rag, rag_backend=rag_backend)
    outputs.update(o1)
    # Prepare SYS1 output text for SYS2 agent input (stringified JSON)
    sys1_text = json.dumps(o1.get("SYS1", []))
    o2 = run_single("SYS2", sys1_text, use_rag=use_rag, rag_backend=rag_backend)
    outputs.update(o2)
    # SYS2 -> Review
    sys2_text = json.dumps(o2.get("SYS2", []))
    o3 = run_single("Review", sys2_text, use_rag=use_rag, rag_backend=rag_backend)
    outputs.update(o3)
    # SYS2 + Review -> SYS5 (use SYS2 text)
    o4 = run_single("SYS5", sys2_text, use_rag=use_rag, rag_backend=rag_backend)
    outputs.update(o4)
    return outputs

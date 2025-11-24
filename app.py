import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # matplotlib optional
    plt = None  # fallback to disable charts when not installed
import base64, os, json, re
import yaml
try:
    import streamlit_authenticator as stauth
except Exception:
    stauth = None  # allow app to run without auth dependency
from datetime import datetime
# Optional interactive chart support (Plotly)
try:
    import plotly.express as px  # type: ignore
except Exception:
    px = None  # noqa: F401
try:
    from streamlit_plotly_events import plotly_events  # type: ignore
except Exception:
    plotly_events = None  # noqa: F401
from pathlib import Path
from orchestrator import run_single, run_pipeline
from utils import (
    parse_agent_output,
    build_traceability_matrix,
    export_csv,
    export_excel_styled,
    export_word_styled,
    export_pdf_styled,
    export_all_as_zip,
    export_reqif,
    save_excel_styled_to_path,
    load_file_for_agent,
)

# Absolute base directory for saving outputs regardless of working dir
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------- Authentication Gate ----------------
# Load auth configuration and enforce login before exposing pages.
def _init_auth():
    if stauth is None:
        return None, True  # auth disabled (dependency missing)
    
    # Try loading from Streamlit secrets first (for cloud deployment)
    config = None
    if hasattr(st, 'secrets') and 'credentials' in st.secrets:
        try:
            config = {
                'credentials': dict(st.secrets['credentials']),
                'cookie': dict(st.secrets['cookie'])
            }
        except Exception as e:
            st.warning(f'Failed to load from Streamlit secrets: {e}')
    
    # Fall back to local auth_config.yaml for development
    if config is None:
        cfg_path = os.path.join(BASE_DIR, 'auth_config.yaml')
        if not os.path.exists(cfg_path):
            st.warning('Auth config missing; running without login.')
            return None, True
        try:
            with open(cfg_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            st.warning(f'Failed to load auth config: {e}. Continuing without auth.')
            return None, True
    
    try:
        authenticator = stauth.Authenticate(
            config['credentials'],
            config['cookie']['name'],
            config['cookie']['key'],
            config['cookie']['expiry_days']
        )
    except Exception as e:
        st.warning(f'Auth init failed: {e}.')
        return None, True
    return authenticator, False

# Check for logout query parameter FIRST before anything else
query_params = st.query_params
if query_params.get('action') == 'logout':
    # Clear all session state keys
    keys_to_clear = list(st.session_state.keys())
    for key in keys_to_clear:
        del st.session_state[key]
    # Clear query params
    st.query_params.clear()
    # Force rerun to show login page
    st.rerun()

authenticator, auth_disabled = _init_auth()

if not auth_disabled and authenticator is not None:
    try:
        authenticator.login(location='main')
    except Exception as e:
        st.error(f'Login failed: {e}')
        st.stop()
    
    # Check authentication status from session state
    auth_status = st.session_state.get('authentication_status')
    auth_name = st.session_state.get('name')
    auth_username = st.session_state.get('username')
    
    if auth_status is False:
        st.error('Invalid username or password')
        st.stop()
    elif auth_status is None:
        st.info('Please enter your credentials.')
        st.stop()
    # Store auth info for display in navbar
    st.session_state['auth_name'] = auth_name
    st.session_state['auth_username'] = auth_username
    st.session_state['authenticator'] = authenticator
    
    # Store login timestamp if not already set
    if 'login_timestamp' not in st.session_state:
        import time
        st.session_state['login_timestamp'] = time.time()
    
    # Check session timeout (1 minute = 60 seconds)
    import time
    SESSION_TIMEOUT = 60  # seconds
    if 'login_timestamp' in st.session_state:
        elapsed = time.time() - st.session_state['login_timestamp']
        if elapsed > SESSION_TIMEOUT:
            # Session expired - force logout
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
else:
    # If auth disabled, indicate
    st.session_state['auth_name'] = None
    st.session_state['auth_username'] = None
    st.session_state['authenticator'] = None
# --- Domain inference for SYS.1 (SW, HW, System, Mechanical)
def infer_domain(text: str) -> str:
    if not isinstance(text, str):
        return "System"
    t = text.lower()
    # Heuristics
    sw_hits = any(k in t for k in (
        "software", "algorithm", "code", "api", "interface", "protocol", "can ", "ethernet", "signal processing", "diagnostic", "dtc", "calibration", "state machine", "firmware"
    ))
    hw_hits = any(k in t for k in (
        "hardware", "sensor", "actuator", "ecu", "processor", "microcontroller", "mcu", "fpga", "power", "voltage", "current", "thermal", "connector", "pcb"
    ))
    mech_hits = any(k in t for k in (
        "mechanical", "bracket", "housing", "mount", "torque", "force", "gear", "bearing", "chassis", "fastener", "screw", "vibration"
    ))
    # Decide with priority for strong signals
    if sw_hits and not (hw_hits or mech_hits):
        return "SW"
    if hw_hits and not mech_hits and not sw_hits:
        return "HW"
    if mech_hits and not (sw_hits or hw_hits):
        return "Mechanical"
    # Mixed: choose System if multiple aspects present
    if sum([sw_hits, hw_hits, mech_hits]) >= 2:
        return "System"
    # Default
    return "System"


# Theme + metadata constants (light gradient palette)
bg = "#ffffff"  # light base
fg = "#111827"  # near-black text
subfg = "#111827"  # muted/secondary text (near-black)
card_border = "#d7e2ec"  # soft border
project_name = "WHALE"
version = "1.0"
author = "System"

# -------- Export Bar Helper (re-implemented after earlier refactor corruption) --------
def render_export_buttons(df: pd.DataFrame, base_name: str, label: str = "Export Options"):
    if df is None or df.empty:
        return

    # Generate export bytes
    try:
        csv_bytes = export_csv(df)
    except Exception:
        csv_bytes = df.to_csv(index=False).encode("utf-8")
    try:
        xlsx_bytes = export_excel_styled(df, base_name)
    except Exception:
        xlsx_bytes = b""
    try:
        docx_bytes = export_word_styled(df, base_name)
    except Exception:
        docx_bytes = b""
    try:
        reqif_bytes = export_reqif(df, spec_name=base_name)
    except Exception:
        reqif_bytes = b""

    def _a(data: bytes, filename: str, text: str, mime: str):
        if not data:
            return f"<span class='seg-btn' style='opacity:.45;cursor:not-allowed;'>{text}</span>"
        b64 = base64.b64encode(data).decode()
        return (
            f"<a class='seg-btn' download='{filename}' href='data:{mime};base64,{b64}' "
            f"style='text-decoration:none;'>{text}</a>"
        )

    pdf_bytes = b""
    try:
        pdf_bytes = export_pdf_styled(df, base_name)
    except Exception:
        pass

    # Small inline SVG icon for download (uses currentColor for easy theming)
    download_svg = (
        "<svg class='icon-svg' viewBox='0 0 24 24' width='14' height='14' xmlns='http://www.w3.org/2000/svg' fill='currentColor' aria-hidden='true'>"
        "<path d='M5 20h14v-2H5v2zm7-18l-5 5h3v6h4V7h3l-5-5z'></path>"
        "</svg>"
    )

    bar_html = [
        "<div class='export-bar-new'>",
        f"<span class='export-label'><span class='export-icon'>{download_svg}</span>{label}</span>",
        f"<div class='seg-group'>",
        _a(docx_bytes, f"{base_name}.docx", "DOCX", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        _a(xlsx_bytes, f"{base_name}.xlsx", "XLSX", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        _a(csv_bytes, f"{base_name}.csv", "CSV", "text/csv"),
        _a(pdf_bytes, f"{base_name}.pdf", "PDF", "application/pdf"),
        _a(reqif_bytes, f"{base_name}.reqif", "REQIF", "application/xml"),
        "</div>",
        "</div>",
    ]
    st.markdown("".join(bar_html), unsafe_allow_html=True)

    # Rendering of the Traceability section happens at the bottom of each page
    # to meet the requirement "rendered at the bottom of the same page".
    # This toggle only controls the state via query params.

def render_static_white_table(df: pd.DataFrame, max_height: str = "95vh", table_id: str = "white-table", no_limit: bool = False):
    """Render a static HTML table with guaranteed white background & vertical scroll.

    This bypasses Streamlit's internal styling (which can force dark headers) by using
    pandas to_html + custom CSS. No interactivity (sorting/filter) but ensures visual consistency.
    """
    if df is None or df.empty:
        return
    safe_id = table_id.replace(" ", "_")
    # Build HTML table
    # Replace newline characters in dataframe (especially Test Steps) with <br> placeholders for HTML rendering
    _df = df.copy()
    for col in _df.columns:
        if _df[col].dtype == object:
            _df[col] = _df[col].apply(lambda v: v.replace("\n", "<br>") if isinstance(v, str) else v)
    html = _df.to_html(index=False, escape=False, border=0, classes=f"{safe_id}-cls")
    wrapper_height_rule = f"max-height:{max_height};" if not no_limit else "max-height:none;"
    css = f"""
        <style>
            /* Light gray theme for static tables */
            .{safe_id}-wrapper {{ {wrapper_height_rule} overflow-y:auto; overflow-x:hidden; border:1px solid #d9dee3; border-radius:6px; background:#ffffff !important; }}
            table.{safe_id}-cls {{ width:100%; border-collapse:collapse; background:#ffffff !important; color:#111827; font-size:13px; }}
            .{safe_id}-cls thead th {{ position:sticky; top:0; background:#f1f5f9; color:#111827; font-weight:600; border-bottom:2px solid #cbd5e1; z-index:5; }}
            .{safe_id}-cls th, .{safe_id}-cls td {{ padding:4px 6px; text-align:left; vertical-align:top; border:1px solid #d9dee3; }}
            .{safe_id}-cls tbody tr:nth-child(odd) td {{ background:#ffffff !important; }}
            .{safe_id}-cls tbody tr:nth-child(even) td {{ background:#f8fafc !important; }}
            .{safe_id}-cls tbody tr:hover td {{ background:#e2e8f0 !important; }}
            .{safe_id}-cls td {{ white-space:normal; word-break:break-word; }}
            .{safe_id}-wrapper::-webkit-scrollbar {{ width:10px; }}
            .{safe_id}-wrapper::-webkit-scrollbar-track {{ background:#f1f5f9; }}
            .{safe_id}-wrapper::-webkit-scrollbar-thumb {{ background:#c3c7cc; border-radius:6px; }}
            .{safe_id}-wrapper::-webkit-scrollbar-thumb:hover {{ background:#9aa0a6; }}
            /* Force white background on all containers */
            body, html {{ background:#ffffff !important; }}
            div[data-testid="stVerticalBlock"] {{ background:#ffffff !important; }}
            section[data-testid="stMain"] {{ background:#ffffff !important; }}
        </style>
        """
    # Use components.html to avoid Streamlit escaping and guarantee rendering
    if no_limit:
        # Expand height based on rows with a generous cap (effectively unlimited)
        row_px = 28
        header_px = 42
        max_px = 5000  # generous cap for long tables
        body_rows = len(df)
        est_height = min(max_px, header_px + body_rows * row_px + 20)
        est_height = max(est_height, 300)
        components.html(css + f"<div class='{safe_id}-wrapper'>{html}</div>", height=est_height, scrolling=True)
    else:
        # Dynamic pixel height: header + rows (approx 28px per row) capped by a max
        row_px = 28
        header_px = 42
        max_px = 1200  # increased safety cap for near full-viewport tables
        body_rows = len(df)
        est_height = min(max_px, header_px + body_rows * row_px + 20)
        est_height = max(est_height, 140)
        components.html(css + f"<div class='{safe_id}-wrapper'>{html}</div>", height=est_height, scrolling=True)


def notify_saved(saved_path: str, label: str = "File"):
    """Show a bottom-right toast only after verifying the file exists on disk."""
    try:
        abs_path = os.path.abspath(saved_path)
        exists = os.path.exists(abs_path) and os.path.getsize(abs_path) > 0
    except Exception:
        exists = False
        abs_path = saved_path
    if exists:
        try:
            st.toast(f"Saved successfully: {abs_path}", icon="✅")
        except Exception:
            pass
    else:
        st.error(f"{label} save reported but file not found on disk. Please use Export buttons or try again.")


def render_traceability_section():
    """Interactive traceability visualization and table (bottom of page, collapsible).

    - Pie shows distribution of: Mapped, Partially Mapped, Unmapped
    - Clicking a slice filters the table below to that status (requires plotly + streamlit-plotly-events)
    - Table columns: Customer Requirement ID, Linked SYS.1 Requirement ID, Traceability Status, Notes
    """
    sys1_df = st.session_state.get("sys1_table_df")
    if sys1_df is None or sys1_df.empty:
        st.info("Traceability data not available yet. Run SYS.1 agent first.")
        return

    # Identify key columns present in SYS.1 table
    cust_id_col = next((c for c in sys1_df.columns if c.lower().startswith("customer req")), None)
    sys1_id_col = next((c for c in sys1_df.columns if c.lower().startswith("sys.1 req")), None)
    cust_text_col = next((c for c in sys1_df.columns if c.lower().strip() == "customer requirement"), None)
    sys1_text_col = next((c for c in sys1_df.columns if c.lower().strip() == "sys.1 requirement"), None)
    status_col = next((c for c in sys1_df.columns if c.lower().startswith("requirement status")), None)
    if not cust_id_col or not sys1_id_col:
        st.warning("Required columns (Customer Req / SYS.1 Req) not found in SYS.1 output.")
        return

    # Build normalized mapping table
    work = sys1_df[[cust_id_col, sys1_id_col]].copy()
    COL_CUST = "Customer Requirement ID (CUST_REQ-00X)"
    COL_SYS1 = "Linked SYS.1 Requirement ID"
    COL_STATUS = "Traceability Status"
    COL_NOTES = "Notes"
    work.columns = [COL_CUST, COL_SYS1]

    # Derive mapping status rules
    def _status_for_row(i: int) -> str:
        sid = str(work.iloc[i][COL_SYS1]).strip()
        if not sid or sid.lower() in {"nan", "none", "null", "-", "tbd"}:
            return "Unmapped"
        # If we have a SYS.1 link, use requirement status if present to refine
        if status_col and i < len(sys1_df):
            sval = sys1_df.iloc[i].get(status_col, "")
            sval = str(sval).strip().lower()
            if "approved" in sval:
                return "Mapped"
            # Draft/Rejected/Other -> treat as partial mapping
            return "Partially Mapped"
        return "Mapped"

    statuses = [ _status_for_row(i) for i in range(len(work)) ]
    notes = []
    for i in range(len(work)):
        stat = statuses[i]
        note = ""
        if stat == "Unmapped":
            note = "No linked SYS.1 requirement yet."
        elif stat == "Partially Mapped":
            sval = str(sys1_df.iloc[i].get(status_col, "")).strip() if status_col else ""
            note = f"Linked status: {sval or 'Draft/Unknown'}"
        else:  # Mapped
            # Prefer short snippet of SYS.1 requirement text if available
            snippet_src = sys1_df.iloc[i].get(sys1_text_col) if sys1_text_col else None
            if not isinstance(snippet_src, str) or not snippet_src.strip():
                snippet_src = sys1_df.iloc[i].get(cust_text_col) if cust_text_col else ""
            sn = str(snippet_src).strip()
            note = (sn[:120] + ("..." if len(sn) > 120 else "")) if sn else ""
        notes.append(note)

    mapping_df = work.copy()
    mapping_df[COL_STATUS] = statuses
    mapping_df[COL_NOTES] = notes

    # Order statuses consistently
    status_order = ["Mapped", "Partially Mapped", "Unmapped"]
    counts = mapping_df[COL_STATUS].value_counts()
    counts = counts.reindex(status_order).fillna(0).astype(int)

    # Collapsible visualization section (open when toggle is on)
    with st.container():
        # Anchor target and non-collapsible section title
        st.markdown(
            "<div id='trace_dashboard'></div><h2 style='margin:0 0 12px; font-weight:800; color:#0f3355;'>Traceability and Status Dashboard</h2>",
            unsafe_allow_html=True,
        )
        # --- Top row: Two pies ---
        c_left, c_right = st.columns(2)

    # Left pie: Traceability Status (Customer Req.)
        traced_mask = mapping_df[COL_SYS1].apply(lambda v: str(v).strip().lower() not in {"", "nan", "none", "null", "-", "tbd"})
        traced_count = int(traced_mask.sum())
        untraced_count = int(len(mapping_df) - traced_count)

        if px is not None:
            try:
                with c_left:
                    _names_a = ["Traced", "Not Traced"]
                    _vals_a = [traced_count, untraced_count]
                    _pairs_a = [(n, v) for n, v in zip(_names_a, _vals_a) if v > 0]
                    if not _pairs_a:
                        _pairs_a = [("No Data", 1)]
                    names_a = [p[0] for p in _pairs_a]
                    values_a = [p[1] for p in _pairs_a]
                    fig_a = px.pie(
                        names=names_a,
                        values=values_a,
                        color=names_a,
                        color_discrete_map={
                            "Traced": "#16a34a",       # green
                            "Not Traced": "#d1d5db",   # gray
                            "No Data": "#e5e7eb",
                        },
                        hole=0.0,
                    )
                    # Show only percent inside to reduce overlap; clearer hover text; subtle slice border
                    fig_a.update_traces(
                        textinfo="percent",
                        textposition="inside",
                        insidetextorientation="radial",
                        textfont_color="#ffffff",
                        hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
                        marker=dict(line=dict(color="#ffffff", width=1)),
                        pull=[0.02, 0.02]
                    )
                    fig_a.update_layout(
                        title_text="Traceability Status (Customer Req.)",
                        height=380,
                        showlegend=True,
                        margin=dict(l=10, r=10, t=40, b=10),
                        paper_bgcolor="#ffffff",
                        plot_bgcolor="#ffffff",
                        font=dict(color="#111827"),
                        legend=dict(title_text="Traceability", orientation="v", yanchor="top", y=0.98, xanchor="right", x=0.98),
                    )
                    st.plotly_chart(fig_a, use_container_width=True, theme=None, config={"displayModeBar": False})
                    st.caption(f"Total Customer Requirements: {len(mapping_df)}")
            except Exception:
                with c_left:
                    st.write({"Traced": traced_count, "Not Traced": untraced_count})
                    st.caption(f"Total Customer Requirements: {len(mapping_df)}")
        else:
            with c_left:
                if plt is not None:
                    try:
                        fig1, ax1 = plt.subplots(figsize=(4.2,4.2))
                        colors = ["#16a34a", "#d1d5db"]
                        ax1.pie([traced_count, untraced_count], labels=["Traced", "Not Traced"], colors=colors, startangle=90, autopct='%1.0f%%', textprops={'fontsize':9})
                        ax1.axis('equal')
                        st.pyplot(fig1, clear_figure=True)
                    except Exception:
                        st.write({"Traced": traced_count, "Not Traced": untraced_count})
                else:
                    st.write({"Traced": traced_count, "Not Traced": untraced_count})
                st.caption(f"Total Customer Requirements: {len(mapping_df)}")

        # Right pie: Overall Status (SYS.1)
        def _norm_status(s: str) -> str:
            s = (s or "").strip().lower()
            if "approve" in s:
                return "Approved"
            if "reject" in s:
                return "Rejected"
            return "Draft"

        if status_col:
            statuses_norm = [ _norm_status(str(sys1_df.iloc[i].get(status_col, ""))) for i in range(len(sys1_df)) ]
        else:
            # If no explicit status, infer: traced -> Approved, else Draft
            statuses_norm = [ "Approved" if traced_mask.iloc[i] else "Draft" for i in range(len(mapping_df)) ]

        from collections import Counter
        scounts = Counter(statuses_norm)
        approved = int(scounts.get("Approved", 0))
        rejected = int(scounts.get("Rejected", 0))
        draft = int(scounts.get("Draft", 0))
        total_sys1 = int(len(sys1_df)) if isinstance(sys1_df, pd.DataFrame) else int(len(mapping_df))

        if px is not None:
            try:
                with c_right:
                    _names_b = ["Approved", "Rejected", "Draft"]
                    _vals_b = [approved, rejected, draft]
                    _pairs_b = [(n, v) for n, v in zip(_names_b, _vals_b) if v > 0]
                    if not _pairs_b:
                        _pairs_b = [("No Data", 1)]
                    names_b = [p[0] for p in _pairs_b]
                    values_b = [p[1] for p in _pairs_b]
                    fig_b = px.pie(
                        names=names_b,
                        values=values_b,
                        color=names_b,
                        color_discrete_map={
                            "Approved": "#16a34a",  # green
                            "Rejected": "#ef4444",  # red
                            "Draft": "#f59e0b",     # amber
                            "No Data": "#e5e7eb",
                        },
                        hole=0.0,
                    )
                    # Show only percent inside to reduce overlap; clearer hover text; subtle slice border
                    fig_b.update_traces(
                        textinfo="percent",
                        textposition="inside",
                        insidetextorientation="radial",
                        textfont_color="#ffffff",
                        hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
                        marker=dict(line=dict(color="#ffffff", width=1)),
                        pull=[0.02, 0.02, 0.02]
                    )
                    fig_b.update_layout(
                        title_text="Overall Status (SYS.1)",
                        height=380,
                        showlegend=True,
                        margin=dict(l=10, r=10, t=40, b=10),
                        paper_bgcolor="#ffffff",
                        plot_bgcolor="#ffffff",
                        font=dict(color="#111827"),
                        legend=dict(title_text="Status", orientation="v", yanchor="top", y=0.98, xanchor="right", x=0.98),
                    )
                    st.plotly_chart(fig_b, use_container_width=True, theme=None, config={"displayModeBar": False})
                    st.caption(f"Total SYS.1 Requirements: {total_sys1}")
            except Exception:
                with c_right:
                    st.write({"Approved": approved, "Rejected": rejected, "Draft": draft})
                    st.caption(f"Total SYS.1 Requirements: {total_sys1}")
        else:
            with c_right:
                if plt is not None:
                    try:
                        fig2, ax2 = plt.subplots(figsize=(4.2,4.2))
                        colors = ["#16a34a", "#ef4444", "#f59e0b"]
                        ax2.pie([approved, rejected, draft], labels=["Approved", "Rejected", "Draft"], colors=colors, startangle=90, autopct='%1.0f%%', textprops={'fontsize':9})
                        ax2.axis('equal')
                        st.pyplot(fig2, clear_figure=True)
                    except Exception:
                        st.write({"Approved": approved, "Rejected": rejected, "Draft": draft})
                else:
                    st.write({"Approved": approved, "Rejected": rejected, "Draft": draft})
                st.caption(f"Total SYS.1 Requirements: {total_sys1}")

    # Minimal spacing before the compact table
        # --- Bottom: Traceability table (CUST_REQ to SYS.1) ---
        st.markdown(
            "<h3 style=\"margin:6px 0 6px; font-weight:800; background:linear-gradient(90deg,#2564d6,#8b5cf6); -webkit-background-clip:text; color:transparent;\">Traceability (CUST_REQ to SYS.1)</h3>",
            unsafe_allow_html=True,
        )
        # Build simplified two-column mapping with row coloring (Approved=green, Rejected=red; else no tint)
        COL_CUST_SHORT = "Customer Req. ID"
        COL_SYS1_SHORT = "SYS.1 Req. ID"
        table_rows = []
        for i in range(len(work)):
            cust_id = str(work.iloc[i][COL_CUST])
            sys1_id = str(work.iloc[i][COL_SYS1])
            stat = _norm_status(str(sys1_df.iloc[i].get(status_col, ""))) if status_col else ("Approved" if str(sys1_id).strip() else "Draft")
            if stat == "Approved":
                bg = "#d1fae5"  # green tint
            elif stat == "Rejected":
                bg = "#fee2e2"  # red tint
            else:
                bg = "#ffffff"  # default
            table_rows.append((cust_id, sys1_id, bg))

        # Build compact HTML table with row-specific backgrounds
        compact_css = """
        <style>
            .trace-compact-wrapper { max-height:30vh; overflow-y:auto; border:1px solid #e2e8f0; border-radius:6px; background:#ffffff; }
            .trace-compact-wrapper::-webkit-scrollbar { width:10px; }
            .trace-compact-wrapper::-webkit-scrollbar-track { background:#f1f5f9; }
            .trace-compact-wrapper::-webkit-scrollbar-thumb { background:#cbd5e1; border-radius:6px; }
            .trace-compact-wrapper::-webkit-scrollbar-thumb:hover { background:#94a3b8; }
            table.trace-compact-table { width:100%; border-collapse:collapse; background:#ffffff; color:#111827; font-size:12px; }
            .trace-compact-table thead th { position:sticky; top:0; background:#ffffff; font-weight:700; border-bottom:2px solid #cbd5e1; z-index:5; }
            .trace-compact-table th, .trace-compact-table td { padding:4px 6px; text-align:left; vertical-align:top; border:1px solid #e2e8f0; line-height:1.1; }
        </style>
        """
        html = [
            compact_css,
            "<div class='trace-compact-wrapper'>",
            "<table class='trace-compact-table'>",
            "<thead><tr>",
            "<th>Customer Req. ID</th>",
            "<th>SYS.1 Req. ID</th>",
            "</tr></thead>",
            "<tbody>",
        ]
        for idx, (cust_id, sys1_id, bg) in enumerate(table_rows):
            # Preserve explicit status colors; otherwise apply blue zebra striping on alternate rows
            base_bg = str(bg).strip().lower()
            if base_bg in {"#d1fae5", "#fee2e2"}:  # green/red tints
                final_bg = bg
            else:
                final_bg = "#f0f7ff" if (idx % 2 == 1) else "#ffffff"
            html.append(
                f"<tr style='background:{final_bg};'><td>{cust_id}</td><td>{sys1_id}</td></tr>"
            )
        html.append("</tbody></table></div>")
        st.markdown("\n".join(html), unsafe_allow_html=True)


# -------- Agent 2 (SYS.1 -> SYS.2) Traceability --------
def render_sys2_traceability(sys2_df: pd.DataFrame, sys1_input_list=None):
    if sys2_df is None or sys2_df.empty:
        return
    # Identify columns
    sys1_id_col = next((c for c in sys2_df.columns if c.lower().startswith("sys.1 req")), None)
    sys2_id_col = next((c for c in sys2_df.columns if c.lower().startswith("sys.2 req")), None)
    if not sys1_id_col or not sys2_id_col:
        return
    # Build mapping SYS.1 -> set(SYS.2)
    mapping = {}
    for _, r in sys2_df.iterrows():
        a = str(r.get(sys1_id_col, "")).strip()
        b = str(r.get(sys2_id_col, "")).strip()
        if not a:
            continue
        mapping.setdefault(a, set())
        if b:
            mapping[a].add(b)
    traced_ids = set(k for k, v in mapping.items() if v)

    # Inputs (unique SYS.1 IDs provided to SYS.2)
    input_ids = set()
    if isinstance(sys1_input_list, list):
        for it in sys1_input_list:
            try:
                if isinstance(it, dict):
                    for k in ("SYS.1 Req. ID", "SYS.1 Req ID", "SYS1 Req ID"):
                        if k in it and it[k] not in (None, ""):
                            input_ids.add(str(it[k]).strip())
                            break
            except Exception:
                continue
    # Fallback: if no inputs captured, use traced IDs as baseline
    total_inputs = len(input_ids) if input_ids else len(traced_ids)
    traced_count = len(traced_ids)
    untraced_count = max(0, total_inputs - traced_count)

    with st.container():
        st.markdown(
            "<h3 style='margin:8px 0 8px; font-weight:800; color:#0f3355;'>SYS.1 → SYS.2 Traceability</h3>",
            unsafe_allow_html=True,
        )
        # Layout for two pies when available
        c1, c2 = st.columns(2)
        if px is not None:
            try:
                names = ["Traced", "Not Traced"]
                vals = [traced_count, untraced_count]
                pairs = [(n, v) for n, v in zip(names, vals) if v > 0]
                if not pairs:
                    pairs = [("No Data", 1)]
                n2 = [p[0] for p in pairs]
                v2 = [p[1] for p in pairs]
                fig = px.pie(names=n2, values=v2, color=n2, color_discrete_map={
                    "Traced": "#16a34a", "Not Traced": "#d1d5db", "No Data": "#e5e7eb"
                })
                fig.update_traces(textinfo="percent", textposition="inside", textfont_color="#ffffff", hovertemplate="%{label}: %{value} (%{percent})<extra></extra>", marker=dict(line=dict(color="#ffffff", width=1)))
                fig.update_layout(title_text="Traceability Status (SYS.1 → SYS.2)", height=320, margin=dict(l=10,r=10,t=40,b=10), paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", font=dict(color="#111827"), legend=dict(title_text="Traceability", orientation="v", yanchor="top", y=0.98, xanchor="right", x=0.98))
                with c1:
                    st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})
            except Exception:
                st.write({"Traced": traced_count, "Not Traced": untraced_count})
        else:
            try:
                if plt is not None:
                    fig, ax = plt.subplots(figsize=(4.0,4.0))
                    ax.pie([traced_count, untraced_count], labels=["Traced", "Not Traced"], colors=["#16a34a", "#d1d5db"], startangle=90, autopct='%1.0f%%')
                    ax.axis('equal')
                    with c1:
                        st.pyplot(fig, clear_figure=True)
                else:
                    st.write({"Traced": traced_count, "Not Traced": untraced_count})
            except Exception:
                st.write({"Traced": traced_count, "Not Traced": untraced_count})

        # Optional second pie: Status distribution if present (e.g., Approved/Draft/Rejected)
        try:
            status_col = next((
                c for c in sys2_df.columns
                if c.lower().strip() in {"status", "overall status", "approval status", "approval", "requirement status"}
            ), None)
            if status_col:
                counts = sys2_df[status_col].astype(str).str.strip().str.title().value_counts()
                # Map to preferred labels and colors
                label_map = {
                    "Approved": "Approved",
                    "Draft": "Draft",
                    "Rejected": "Rejected",
                }
                color_map = {"Approved": "#16a34a", "Draft": "#f59e0b", "Rejected": "#ef4444"}
                labels = []
                values = []
                colors = []
                for k, v in counts.items():
                    lbl = label_map.get(k, k)
                    labels.append(lbl)
                    values.append(int(v))
                    if lbl not in color_map:
                        # assign a pleasant blue for unknown classes
                        color_map[lbl] = "#60a5fa"
                    colors.append(color_map[lbl])
                if labels and any(v > 0 for v in values):
                    if px is not None:
                        try:
                            fig2 = px.pie(names=labels, values=values, color=labels, color_discrete_map=color_map)
                            fig2.update_traces(textinfo="percent", textposition="inside", textfont_color="#ffffff", hovertemplate="%{label}: %{value} (%{percent})<extra></extra>", marker=dict(line=dict(color="#ffffff", width=1)))
                            fig2.update_layout(title_text="Overall Status (SYS.2)", height=320, margin=dict(l=10,r=10,t=40,b=10), paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", font=dict(color="#111827"), legend=dict(title_text="Status", orientation="v", yanchor="top", y=0.98, xanchor="right", x=0.98))
                            with c2:
                                st.plotly_chart(fig2, use_container_width=True, theme=None, config={"displayModeBar": False})
                        except Exception:
                            with c2:
                                st.write({lbl: int(v) for lbl, v in zip(labels, values)})
                    elif plt is not None:
                        try:
                            fig2, ax2 = plt.subplots(figsize=(4.0,4.0))
                            ax2.pie(values, labels=labels, colors=[color_map[l] for l in labels], startangle=90, autopct='%1.0f%%')
                            ax2.axis('equal')
                            with c2:
                                st.pyplot(fig2, clear_figure=True)
                        except Exception:
                            with c2:
                                st.write({lbl: int(v) for lbl, v in zip(labels, values)})
        except Exception:
            pass

        # Mapping table (compact)
        st.markdown(
            "<h4 style=\"margin:6px 0 6px; font-weight:800; background:linear-gradient(90deg,#2564d6,#8b5cf6); -webkit-background-clip:text; color:transparent;\">Mapping (SYS.1 Req → SYS.2 Req)</h4>",
            unsafe_allow_html=True,
        )
        # Coverage badges above table
        try:
            pct = 0 if (total_inputs or 0) == 0 else round(100 * traced_count / max(1, total_inputs))
            st.markdown(
                f"<div style='margin:4px 0 6px;'>"
                f"<span style='display:inline-block; padding:4px 8px; border-radius:9999px; background:#ecfdf5; color:#065f46; font-weight:600; margin-right:6px;'>Traced: {traced_count}</span>"
                f"<span style='display:inline-block; padding:4px 8px; border-radius:9999px; background:#fef3c7; color:#92400e; font-weight:600; margin-right:6px;'>Total: {total_inputs}</span>"
                f"<span style='display:inline-block; padding:4px 8px; border-radius:9999px; background:#eff6ff; color:#1d4ed8; font-weight:700;'>Coverage: {pct}%</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        except Exception:
            pass
        rows = []
        for sid in sorted(mapping.keys()):
            lst = sorted(mapping[sid])
            rows.append((sid, ", ".join(lst) if lst else ""))
        css = """
        <style>
            .map-compact { max-height:28vh; overflow:auto; border:1px solid #e2e8f0; border-radius:6px; }
            .map-compact table { width:100%; border-collapse:collapse; background:#ffffff; font-size:12px; }
            .map-compact th, .map-compact td { padding:4px 6px; border:1px solid #e2e8f0; text-align:left; }
            .map-compact thead th { position:sticky; top:0; background:#ffffff; }
        </style>
        """
        html = [css, "<div class='map-compact'><table>", "<thead><tr><th>SYS.1 Req. ID</th><th>SYS.2 Req. ID(s)</th></tr></thead><tbody>"]
        for i, (a, b) in enumerate(rows):
            zebra = "#f0f7ff" if (i % 2 == 1) else "#ffffff"
            html.append(f"<tr style='background:{zebra};'><td>{a}</td><td>{b}</td></tr>")
        html.append("</tbody></table></div>")
        st.markdown("".join(html), unsafe_allow_html=True)


# -------- Agent 4 (SYS.2 -> SYS.5) Traceability --------
def render_sys5_traceability(sys5_df: pd.DataFrame, sys2_input_list=None):
    if sys5_df is None or sys5_df.empty:
        return
    sys2_id_col = next((c for c in sys5_df.columns if c.lower().startswith("sys.2 req")), None)
    tc_id_col = next((c for c in sys5_df.columns if c.lower().startswith("test case id")), None)
    if not sys2_id_col or not tc_id_col:
        return
    # Mapping SYS.2 -> set(Test Case IDs)
    mapping = {}
    for _, r in sys5_df.iterrows():
        a = str(r.get(sys2_id_col, "")).strip()
        b = str(r.get(tc_id_col, "")).strip()
        if not a:
            continue
        mapping.setdefault(a, set())
        if b:
            mapping[a].add(b)
    traced_ids = set(k for k, v in mapping.items() if v)

    input_ids = set()
    if isinstance(sys2_input_list, list):
        for it in sys2_input_list:
            try:
                if isinstance(it, dict):
                    for k in ("SYS.2 Req. ID", "SYS.2 Req ID", "SYS2 Req ID", "SYS_2 Req ID"):
                        if k in it and it[k] not in (None, ""):
                            input_ids.add(str(it[k]).strip())
                            break
            except Exception:
                continue
    total_inputs = len(input_ids) if input_ids else len(traced_ids)
    traced_count = len(traced_ids)
    untraced_count = max(0, total_inputs - traced_count)

    with st.container():
        st.markdown(
            "<h3 style='margin:8px 0 8px; font-weight:800; color:#0f3355;'>SYS.2 → SYS.5 Traceability</h3>",
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        if px is not None:
            try:
                names = ["Traced", "Not Traced"]
                vals = [traced_count, untraced_count]
                pairs = [(n, v) for n, v in zip(names, vals) if v > 0]
                if not pairs:
                    pairs = [("No Data", 1)]
                n2 = [p[0] for p in pairs]
                v2 = [p[1] for p in pairs]
                fig = px.pie(names=n2, values=v2, color=n2, color_discrete_map={
                    "Traced": "#16a34a", "Not Traced": "#d1d5db", "No Data": "#e5e7eb"
                })
                fig.update_traces(textinfo="percent", textposition="inside", textfont_color="#ffffff", hovertemplate="%{label}: %{value} (%{percent})<extra></extra>", marker=dict(line=dict(color="#ffffff", width=1)))
                fig.update_layout(title_text="Traceability Status (SYS.2 → SYS.5)", height=320, margin=dict(l=10,r=10,t=40,b=10), paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", font=dict(color="#111827"), legend=dict(title_text="Traceability", orientation="v", yanchor="top", y=0.98, xanchor="right", x=0.98))
                with c1:
                    st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})
            except Exception:
                st.write({"Traced": traced_count, "Not Traced": untraced_count})
        else:
            try:
                if plt is not None:
                    fig, ax = plt.subplots(figsize=(4.0,4.0))
                    ax.pie([traced_count, untraced_count], labels=["Traced", "Not Traced"], colors=["#16a34a", "#d1d5db"], startangle=90, autopct='%1.0f%%')
                    ax.axis('equal')
                    with c1:
                        st.pyplot(fig, clear_figure=True)
                else:
                    st.write({"Traced": traced_count, "Not Traced": untraced_count})
            except Exception:
                st.write({"Traced": traced_count, "Not Traced": untraced_count})

        # Optional second pie: Test Case Priority/Status distribution if present
        try:
            status_col = next((c for c in sys5_df.columns if c.lower().strip() in {"status", "priority", "test status", "overall status"}), None)
            if status_col:
                counts = sys5_df[status_col].astype(str).str.strip().str.title().value_counts()
                label_map = {
                    "High": "High Priority",
                    "Medium": "Medium Priority",
                    "Low": "Low Priority",
                    "Approved": "Approved",
                    "Draft": "Draft",
                    "Rejected": "Rejected",
                    "Passed": "Passed",
                    "Failed": "Failed",
                }
                color_map = {
                    "High Priority": "#ef4444",
                    "Medium Priority": "#f59e0b",
                    "Low Priority": "#3b82f6",
                    "Approved": "#16a34a",
                    "Draft": "#f59e0b",
                    "Rejected": "#ef4444",
                    "Passed": "#16a34a",
                    "Failed": "#ef4444",
                }
                labels, values = [], []
                for k, v in counts.items():
                    lbl = label_map.get(str(k), str(k))
                    labels.append(lbl)
                    values.append(int(v))
                    if lbl not in color_map:
                        color_map[lbl] = "#60a5fa"
                if labels and any(v > 0 for v in values):
                    if px is not None:
                        try:
                            fig2 = px.pie(names=labels, values=values, color=labels, color_discrete_map=color_map)
                            fig2.update_traces(textinfo="percent", textposition="inside", textfont_color="#ffffff", hovertemplate="%{label}: %{value} (%{percent})<extra></extra>", marker=dict(line=dict(color="#ffffff", width=1)))
                            fig2.update_layout(title_text="Overall Status (SYS.5)", height=320, margin=dict(l=10,r=10,t=40,b=10), paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", font=dict(color="#111827"), legend=dict(title_text="Status", orientation="v", yanchor="top", y=0.98, xanchor="right", x=0.98))
                            with c2:
                                st.plotly_chart(fig2, use_container_width=True, theme=None, config={"displayModeBar": False})
                        except Exception:
                            with c2:
                                st.write({lbl: int(v) for lbl, v in zip(labels, values)})
                    elif plt is not None:
                        try:
                            fig2, ax2 = plt.subplots(figsize=(4.0,4.0))
                            ax2.pie(values, labels=labels, colors=[color_map[l] for l in labels], startangle=90, autopct='%1.0f%%')
                            ax2.axis('equal')
                            with c2:
                                st.pyplot(fig2, clear_figure=True)
                        except Exception:
                            with c2:
                                st.write({lbl: int(v) for lbl, v in zip(labels, values)})
        except Exception:
            pass

        # Mapping table
        st.markdown(
            "<h4 style=\"margin:6px 0 6px; font-weight:800; background:linear-gradient(90deg,#2564d6,#8b5cf6); -webkit-background-clip:text; color:transparent;\">Mapping (SYS.2 Req → Test Case ID)</h4>",
            unsafe_allow_html=True,
        )
        # Coverage badges above table
        try:
            pct = 0 if (total_inputs or 0) == 0 else round(100 * traced_count / max(1, total_inputs))
            st.markdown(
                f"<div style='margin:4px 0 6px;'>"
                f"<span style='display:inline-block; padding:4px 8px; border-radius:9999px; background:#ecfdf5; color:#065f46; font-weight:600; margin-right:6px;'>Traced: {traced_count}</span>"
                f"<span style='display:inline-block; padding:4px 8px; border-radius:9999px; background:#fef3c7; color:#92400e; font-weight:600; margin-right:6px;'>Total: {total_inputs}</span>"
                f"<span style='display:inline-block; padding:4px 8px; border-radius:9999px; background:#eff6ff; color:#1d4ed8; font-weight:700;'>Coverage: {pct}%</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        except Exception:
            pass
        rows = []
        for sid in sorted(mapping.keys()):
            lst = sorted(mapping[sid])
            rows.append((sid, ", ".join(lst) if lst else ""))
        css = """
        <style>
            .map-compact { max-height:28vh; overflow:auto; border:1px solid #e2e8f0; border-radius:6px; }
            .map-compact table { width:100%; border-collapse:collapse; background:#ffffff; font-size:12px; }
            .map-compact th, .map-compact td { padding:4px 6px; border:1px solid #e2e8f0; text-align:left; }
            .map-compact thead th { position:sticky; top:0; background:#ffffff; }
        </style>
        """
        html = [css, "<div class='map-compact'><table>", "<thead><tr><th>SYS.2 Req. ID</th><th>Test Case ID(s)</th></tr></thead><tbody>"]
        for i, (a, b) in enumerate(rows):
            zebra = "#f0f7ff" if (i % 2 == 1) else "#ffffff"
            html.append(f"<tr style='background:{zebra};'><td>{a}</td><td>{b}</td></tr>")
        html.append("</tbody></table></div>")
        st.markdown("".join(html), unsafe_allow_html=True)

# --- Navigation session state helpers (restored) ---
st.set_page_config(page_title="WHALE-RE", layout="wide")
if "page" not in st.session_state:
    st.session_state.page = "Home"
# Dark mode permanently disabled (requirement: always white theme). Remove any previous state usage.

# Sync page from URL query parameter using stable API
try:
    qp = st.query_params
    if "page" in qp:
        _candidate = qp.get("page")
        valid_pages = {"Home","SYS1","SYS2","Review","SYS5","Manager"}
        if _candidate in valid_pages:
            st.session_state.page = _candidate
    # Sync traceability toggle from query param ?trace=1
    # Only set True when present; do not force-clear on absence so inline checkbox can control state.
    if "trace" in qp:
        st.session_state.show_traceability = True
except Exception:
    pass

def go_to(page: str):
    st.session_state.page = page
    try:
        # Update URL without deprecated experimental API
        st.query_params["page"] = page
    except Exception:
        pass


# Unified light theme constants (single source of truth)
# Background stays pure white; foreground palette refined for contrast harmony
bg = "#ffffff"
fg = "#111827"       # Primary text (near-black)
subfg = "#111827"    # Muted / secondary text (near-black)
card_border = "#d7e2ec"  # Soft cool border

main_css = """
<style>
                /* Global override: force body and root to always be light regardless of system theme */
                html, body, #root, .stApp, [data-testid="stAppViewContainer"] { 
                    background:#ffffff !important; 
                    color:#111827 !important; 
                    color-scheme: light !important;
                }
                /* Toast notification positioning at top-right */
                .stToast { 
                    position: fixed !important;
                    top: 80px !important;
                    right: 20px !important;
                    z-index: 9999 !important;
                    min-width: 300px !important;
                    max-width: 450px !important;
                }
                /* Neon animated arrow for workflow indicators */
                .neon-flow { color:#0f3355; font-weight:600; }
                .neon-flow .step { color:#000; }
                .neon-arrow { display:inline-block; margin:0 6px; color:#6BFFFB; text-shadow:0 0 6px #15F5FF, 0 0 14px #14B8FF; animation: pulseGlow 1.55s ease-in-out infinite; }
                .neon-arrow.alt { color:#B26BFF; text-shadow:0 0 6px #C084FC, 0 0 14px #9333EA; animation-delay:.35s; }
                @keyframes pulseGlow { 0%,100% { filter:drop-shadow(0 0 2px rgba(21,245,255,.75)); opacity:1; }
                    50% { filter:drop-shadow(0 0 8px rgba(21,245,255,.95)); opacity:.85; } }
    body, .stApp { background:radial-gradient(circle at 40% 20%, #ffffff 0%, #eef6ff 55%, #d9ecfb 100%) fixed; color:__FG__; }
    /* Expand to full viewport width while keeping comfortable side padding */
    .block-container { max-width:100% !important; width:100% !important; padding-left:1.25rem !important; padding-right:1.25rem !important; }
    /* Remove default Streamlit top padding / blank header space */
    .stApp header { display:none !important; }
    .block-container, .main .block-container { padding-top:0.35rem !important; }
    /* Also reset root element margin to compress space */
    .stApp { padding-top:0 !important; margin-top:0 !important; }
    .hero-wrapper { margin-top:24px; margin-bottom:34px; display:flex; align-items:center; justify-content:flex-start; gap:46px; background:linear-gradient(145deg, rgba(255,255,255,0.75), rgba(231,243,255,0.85)); padding:46px 54px; border:1px solid #d2e5f4; border-radius:28px; box-shadow:0 6px 30px -10px rgba(31,78,120,0.18), 0 2px 10px -4px rgba(31,78,120,0.12); }
    .hero-visual { flex:0 0 200px; height:170px; border-radius:22px; background:radial-gradient(circle at 40% 35%,#7bc4ff,#3d8bff 55%, #2564d6); position:relative; box-shadow:0 8px 28px -12px rgba(61,139,255,0.55),0 4px 16px -4px rgba(0,0,0,0.15); overflow:hidden; display:flex; align-items:center; justify-content:center; }
    .hero-visual:after { content:""; position:absolute; inset:0; background:radial-gradient(circle at 75% 75%, rgba(255,255,255,0.45), transparent 60%); mix-blend-mode:overlay; }
    .hero-copy { max-width:760px; }
    .hero-title { text-align:left; font-size:54px; font-weight:800; letter-spacing:1px; margin:0 0 10px 0; line-height:1.05; }
    /* Enhanced logo-style word */
    .hero-title span.logo-word { display:inline-block; background:linear-gradient(90deg,#2564d6,#58a6ff 55%,#8fd4ff); -webkit-background-clip:text; color:transparent; position:relative; padding:2px 6px 4px; }
    @keyframes underlineGlow {
        0%, 100% { width: 40%; opacity: 0.4; }
        50% { width: 60%; opacity: 0.7; }
    }
    .hero-title span.logo-word:after { 
        content:""; 
        position:absolute; 
        left: 50%; 
        transform: translateX(-50%);
        bottom:2px; 
        height:3px; 
        border-radius:4px; 
        background:linear-gradient(90deg, transparent, #58a6ff, #8fd4ff, transparent); 
        animation: underlineGlow 3s ease-in-out infinite;
    }
    .hero-title .mini-accent { font-size:15px; vertical-align:top; margin-left:4px; font-weight:700; color:#2564d6; letter-spacing:2px; }
    .hero-quote { text-align:left; color:__SUBFG__; font-size:18px; margin:0 0 16px 0; }
    .hero-quote.tagline { font-style:italic; font-weight:600; font-size:23px; letter-spacing:.4px; position:relative; padding:6px 0 6px 18px; margin-left:0; color:#0f3355; background:unset; -webkit-background-clip:border-box; }
    .hero-quote.tagline:before { content:""; position:absolute; left:0; top:8px; bottom:8px; width:6px; border-radius:4px; background:linear-gradient(180deg,#2564d6,#58a6ff); box-shadow:0 0 0 1px rgba(37,100,214,0.25), 0 4px 10px -3px rgba(37,100,214,0.45); }
    .hero-quote.tagline em { font-style:inherit; font-weight:700; background:linear-gradient(90deg,#163a59,#2564d6 55%,#58a6ff); -webkit-background-clip:text; color:transparent; }
    @media (max-width:1100px) { .hero-title { font-size:48px; } }
    @media (max-width:820px) { .hero-title { font-size:42px; } }
    .hero-sub { text-align:left; color:#0f3355; font-size:16px; font-weight:600; background:linear-gradient(135deg,rgba(255,255,255,0.9),rgba(214,236,255,0.85)); display:inline-block; padding:8px 16px 9px; border-radius:10px; box-shadow:0 4px 18px -8px rgba(31,78,120,0.28), 0 2px 8px -4px rgba(31,78,120,0.20); letter-spacing:.8px; border:1px solid #c1d7e8; position:relative; overflow:hidden; }
    .hero-sub:before { content:""; position:absolute; inset:0; background:linear-gradient(120deg,rgba(37,100,214,0.15),rgba(255,255,255,0)); mix-blend-mode:overlay; opacity:.55; pointer-events:none; }
    .card { border:1px solid rgba(0,0,0,0.06); border-radius:16px; padding:14px 20px 20px; min-height:140px; box-sizing:border-box; position:relative; overflow:hidden; transition:box-shadow .25s, transform .20s, border-color .25s, background .25s; margin-bottom:10px; background:transparent; box-shadow:0px 4px 12px rgba(0,0,0,0.15); color:#000000; }
    .card:before { content:none; }
    a.card-link:hover .card, a.card-link:focus .card { transform:translateY(-4px); box-shadow:0px 10px 24px rgba(0,0,0,0.18); border-color:rgba(0,0,0,0.10); }
    a.card-link:active .card { transform:translateY(-1px); box-shadow:0px 6px 14px rgba(0,0,0,0.16); }
    /* Stronger, smooth gradients per card type */
    .card.sys1 { background:linear-gradient(135deg,#bbdefb,#90caf9); }
    .card.sys2 { background:linear-gradient(135deg,#c8e6c9,#a5d6a7); }
    .card.review { background:linear-gradient(135deg,#ffe0b2,#ffcc80); }
    .card.sys5 { background:linear-gradient(135deg,#f8bbd0,#f48fb1); }
    .card.manager { background:linear-gradient(135deg,#d1c4e9,#b39ddb); }
    /* Icon pill inside card headings */
    .icon-pill { display:inline-flex; align-items:center; justify-content:center; width:30px; height:30px; border-radius:10px; background:linear-gradient(135deg,#ffffff,#e1f1ff); margin-right:8px; font-size:15px; box-shadow:0 2px 6px -2px rgba(31,78,120,0.25); }
    .status-row { position:absolute; left:20px; bottom:18px; }
    .status-pill { display:inline-block; padding:4px 12px 5px; font-size:11.5px; letter-spacing:.4px; font-weight:700; border-radius:999px; background:#e5e7eb; color:#111827; box-shadow:0 3px 10px -4px rgba(31,78,120,0.20); border:1px solid transparent; }
    .status-pill.neutral { background:#6b7280; color:#ffffff; }
    .status-pill.progress { background:#2564d6; color:#ffffff; }
    .status-pill.warning { background:#d97706; color:#ffffff; }
    .status-pill.success { background:#16a34a; color:#ffffff; }
    /* CTA primary action button (updated to light palette gradient) */
    .cta-btn-primary { display:inline-block; padding:14px 28px; font-weight:600; font-size:15px; letter-spacing:.5px; border-radius:14px; background:linear-gradient(120deg,#2564d6,#58a6ff); color:#ffffff !important; text-decoration:none; box-shadow:0 8px 26px -10px rgba(37,100,214,0.45), 0 3px 10px -4px rgba(37,100,214,0.35); position:relative; overflow:hidden; transition:box-shadow .3s, transform .25s; }
    .cta-btn-primary:before { content:""; position:absolute; inset:0; background:linear-gradient(150deg,rgba(255,255,255,0.35),rgba(255,255,255,0)); opacity:.55; mix-blend-mode:overlay; }
    .cta-btn-primary:hover { transform:translateY(-4px); box-shadow:0 14px 36px -12px rgba(37,100,214,0.55), 0 5px 14px -4px rgba(37,100,214,0.40); }
    .cta-btn-primary:active { transform:translateY(-1px); box-shadow:0 6px 18px -8px rgba(37,100,214,0.55); }
    /* Unified 3-column (auto) grid for home cards */
    .cards-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(250px,1fr)); gap:28px 26px; align-items:stretch; }
    .cards-row-single { margin-top:34px; }
    @media (max-width:1180px){ .cards-grid { gap:22px 20px; } }
    @media (max-width:980px){ .cards-grid { grid-template-columns:repeat(auto-fill,minmax(260px,1fr)); gap:18px; } }
    @media (max-width:820px){ .card { min-height:140px; } .hero-title { font-size:40px; } }
    .card h3 { margin:0 0 6px 0; font-size:17px; font-weight:700; letter-spacing:.3px; color:#000000; display:flex; align-items:center; gap:8px; line-height:1.3; }
    .card h3 .animated-icon { font-size:20px; flex-shrink:0; display:inline-flex; align-items:center; justify-content:center; width:24px; height:24px; }
    .muted { color:#374151; font-size:13px; line-height:1.45; margin:0; font-weight:400; }
    a.card-link { text-decoration:none; color:inherit; display:block; width:100%; }
    a.card-link:focus { outline:none; }
    a.card-link:focus .card { box-shadow:0 0 0 3px rgba(59,130,246,0.45), 0 6px 22px -6px rgba(0,0,0,0.25); }
    /* (Reverted) Removed modern glassmorphism variants */
    .nav-bar { background:rgba(255,255,255,0.95) !important; border:2px solid rgba(37,100,214,0.20); border-radius:28px; position:fixed; left:12px; right:12px; top:6px; z-index:999; backdrop-filter:blur(22px) saturate(1.15); margin-bottom:26px; padding:6px 32px; box-shadow:0 4px 12px -2px rgba(31,78,120,0.15), 0 12px 32px -8px rgba(31,78,120,0.25), 0 0 0 1px rgba(255,255,255,0.8) inset; overflow:hidden; }
    .nav-bar:after { content:""; position:absolute; left:0; right:0; bottom:0; height:5px; background:linear-gradient(90deg,rgba(37,100,214,0.0),rgba(37,100,214,0.35),rgba(88,166,255,0.0)); pointer-events:none; filter:blur(4px); opacity:.55; }
    .nav-bar .nav-link { color:#2b4b66 !important; text-decoration:none; margin:0 40px 0 0; font-weight:500; display:inline-block; padding:10px 14px 10px; border-radius:14px; position:relative; font-size:15px; letter-spacing:.3px; transition:color .25s, background .25s, box-shadow .25s; }
    .nav-bar .nav-link:hover { color:#111827 !important; background:rgba(37,100,214,0.08); }
    .nav-bar .nav-link.active { color:#0f3355 !important; cursor:default; pointer-events:none; font-weight:600; background:linear-gradient(135deg,rgba(37,100,214,0.22),rgba(88,166,255,0.18)); box-shadow:0 4px 12px -4px rgba(37,100,214,0.35) inset, 0 0 0 1px rgba(37,100,214,0.35); }
    .nav-bar .nav-link.active:after { content:""; position:absolute; left:10%; bottom:4px; height:3px; width:80%; background:linear-gradient(90deg,#2564d6,#58a6ff); border-radius:2px; opacity:.85; }
    .nav-bar .nav-link:last-child { margin-right:0; }
    div.stButton > button { background:#ffffff !important; color:#18324a !important; border:1px solid #c1d7e8 !important; border-radius:12px !important; font-weight:700 !important; padding:12px 18px !important; font-size:15px !important; transition:background .25s, border-color .25s, box-shadow .25s; }
    div.stButton > button:hover { background:#f1f8ff !important; border-color:#9ac0da !important; box-shadow:0 3px 14px -6px rgba(31,78,120,0.25); }
    div.stButton > button:active { background:#e2f1fb !important; box-shadow:0 2px 6px rgba(31,78,120,0.25) inset; }
    .stFileUploader > label { font-weight:600; color:__FG__; }
    .stFileUploader .uploadedFile { background:#ffffff !important; color:#111827 !important; }
    .stFileUploader .uploadedFile * { color:#111827 !important; }
    [data-testid="stFileUploaderFile"] { background:#ffffff !important; border:1px solid #d1d5db !important; border-radius:6px !important; padding:4px 8px !important; box-shadow:0 1px 2px rgba(0,0,0,0.04) !important; }
    [data-testid="stFileUploaderFileName"], [data-testid="stFileUploaderFileSize"], [data-testid="stFileUploaderFileName"] *, [data-testid="stFileUploaderFileSize"] * { color:#111827 !important; font-weight:500 !important; }
    .stFileUploader [data-testid="stFileUploaderDropzone"] { background:#ffffff !important; border:1px dashed #cbd5e1 !important; color:#111827 !important; padding:0.6rem 0.75rem !important; border-radius:10px !important; }
    .stFileUploader [data-testid="stFileUploaderDropzone"] * { color:#111827 !important; }
    .stFileUploader [data-testid="stFileUploaderDropzone"] button { background:#ffffff !important; color:#111827 !important; border:1px solid #d1d5db !important; border-radius:6px !important; font-weight:600 !important; box-shadow:none !important; }
    .stFileUploader [data-testid="stFileUploaderDropzone"] button:hover { background:#f3f4f6 !important; border-color:#9ca3af !important; }
    .stFileUploader [data-testid="stFileUploaderDropzone"] button:active { background:#e5e7eb !important; }
    .stTextArea textarea { background:#ffffff !important; color:#111827 !important; border:1px solid #c1d7e8 !important; border-radius:10px !important; font-family:inherit !important; font-size:14px !important; }
    .stTextArea textarea:focus { outline:none !important; border-color:#2564d6 !important; box-shadow:0 0 0 1px #2564d6 inset, 0 0 0 3px rgba(37,100,214,0.30) !important; }
    .stTextArea textarea:disabled { background:#f3f8fb !important; color:#5a6b7d !important; opacity:1 !important; }
    .stTextArea label { color:#111827 !important; font-weight:600 !important; }
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] div, .stMultiSelect div[data-baseweb="select"] div, .stDateInput input, .stTimeInput input { background:#ffffff !important; color:#111827 !important; border:1px solid #c1d7e8 !important; border-radius:10px !important; font-size:14px !important; }
    .stTextInput input:focus, .stNumberInput input:focus, .stSelectbox div[data-baseweb="select"]:focus-within div, .stMultiSelect div[data-baseweb="select"]:focus-within div, .stDateInput input:focus, .stTimeInput input:focus { outline:none !important; border-color:#2564d6 !important; box-shadow:0 0 0 1px #2564d6 inset, 0 0 0 3px rgba(37,100,214,0.30) !important; }
    .stTextInput input:disabled, .stNumberInput input:disabled, .stSelectbox div[data-baseweb="select"]:disabled div, .stMultiSelect div[data-baseweb="select"]:disabled div { background:#f8fafc !important; color:#374151 !important; opacity:1 !important; }
    /* Ensure all expanders keep the light theme (no dark bars) */
    [data-testid="stExpander"] details { background:#ffffff !important; border:1px solid #d1d5db !important; border-radius:10px !important; }
    [data-testid="stExpander"] summary { background:#ffffff !important; color:#111827 !important; border-radius:10px !important; padding:10px 12px !important; }
    [data-testid="stExpander"] summary:hover { background:#f8fafc !important; }
    [data-testid="stExpander"] div[role="region"] { background:#ffffff !important; padding:10px 12px !important; }
    div[data-baseweb="toggle"] { background:#ffffff !important; border:1px solid #d1d5db !important; border-radius:999px !important; padding:2px !important; }
    div[data-baseweb="toggle"] > div:first-child { background:#dc2626 !important; box-shadow:none !important; }
    div[data-baseweb="toggle"] input:checked + div { background:#16a34a !important; }
    div[data-baseweb="toggle"] label, div[data-baseweb="toggle"] span { color:#ffffff !important; }
    .export-row div.stButton > button { background:#ffffff !important; color:#111827 !important; border:1px solid #d1d5db !important; border-radius:8px !important; font-weight:600 !important; padding:4px 12px !important; min-width:70px; }
    .export-row div.stButton > button:hover { background:#f3f4f6 !important; border-color:#9ca3af !important; }
    .export-row div.stButton > button:active { background:#e5e7eb !important; }
    .export-row div[data-testid="stDownloadButton"] button,
    .export-row button[kind*="download"],
    .export-row div[data-testid="stButton"] button {
            background:#ffffff !important;
            color:#111827 !important;
            border:1px solid #d1d5db !important;
            border-radius:8px !important;
            font-weight:600 !important;
            padding:4px 12px !important;
            box-shadow:none !important;
    }
    .export-row div[data-testid="stDownloadButton"] button:hover,
    .export-row button[kind*="download"]:hover { background:#f1f8ff !important; border-color:#9ac0da !important; }
    .export-row div[data-testid="stDownloadButton"] button:active,
    .export-row button[kind*="download"]:active { background:#e2f1fb !important; }
    .export-row div[data-testid="stDownloadButton"] button:disabled,
    .export-row button[kind*="download"]:disabled { background:#ffffff !important; color:#9ca3af !important; opacity:1 !important; }
    .responsive-table-wrapper { width:100%; overflow-x:hidden !important; }
    .responsive-table-wrapper table { width:100% !important; }
    .stDataFrame div[data-testid="stHorizontalBlock"] { background:#ffffff !important; }
    .stDataFrame [data-testid="stHeader"] { background:#ffffff !important; color:#111827 !important; }
    .stDataFrame tbody tr td { background:#ffffff !important; color:#111827 !important; }
    .stDataFrame tbody tr:hover td { background:#f8fafc !important; }
    .stDataFrame table { background:#ffffff !important; table-layout:auto !important; width:100% !important; }
    .stDataFrame th, .stDataFrame td { width:auto !important; }
    .stDataFrame thead tr th { background:#ffffff !important; color:#111827 !important; font-weight:600; }
    .stDataFrame tbody tr th { background:#ffffff !important; color:#111827 !important; }
    .stDataFrame td, .stDataFrame th { word-wrap:break-word; white-space:pre-wrap !important; line-height:1.25rem; }
    .stDataFrame [data-testid="stDataFrameScrollable"],
    .stDataFrame [data-testid="stDataFrameContainer"] { width:100% !important; overflow-x:hidden !important; }
    .stDataFrame td, .stDataFrame th { word-break:break-word !important; hyphens:auto; }
    .stDataFrame [data-testid="stDataFrameScrollable"] { max-height:70vh; overflow-y:auto !important; }
    .stDataEditor [data-testid="stDataFrameContainer"],
    .stDataEditor [data-testid="stDataFrameScrollable"] { width:100% !important; }
    /* Ensure later duplicate rule keeps full width but can tighten padding on very small screens */
    @media (max-width:640px){ .block-container { padding-left:0.75rem !important; padding-right:0.75rem !important; } }
    .stDataFrame thead tr th { position:sticky; top:0; z-index:2; box-shadow:0 1px 0 #e5e7eb; }
    .stDataFrame tbody tr:nth-child(odd) td { background:#ffffff !important; }
    .stDataFrame tbody tr:nth-child(even) td { background:#f8fafc !important; }
    .stDataFrame tbody tr:hover td { background:#eef6ff !important; }
    .stDataFrame, .stDataFrame * { color:#111827 !important; }
    .stDataFrame tbody tr td, .stDataFrame thead tr th, .stDataFrame tbody tr th { background:#ffffff !important; }
    .stDataFrame [role="cell"], .stDataFrame [role="columnheader"] { border:1px solid #e2e8f0 !important; box-sizing:border-box; }
    .stDataFrame thead tr th { font-weight:600 !important; }
    .stDataFrame tbody tr:hover td { background:#f5f7fa !important; }
    [data-testid="stDataFrameScrollable"] { overflow-x:auto !important; }
    .stDataFrame [role="cell"] div { white-space:normal !important; word-break:break-word !important; }
    .stDataEditor, .stDataEditor * { color:#111827 !important; }
    .stDataEditor [data-testid="stDataFrameContainer"] { background:#f8fafc !important; }
    .stDataEditor [role="columnheader"],
    .stDataEditor [role="gridcell"],
    .stDataEditor [role="cell"] { background:#f8fafc !important; border:1px solid #e2e8f0 !important; box-sizing:border-box; }
    .stDataEditor [role="columnheader"] { font-weight:600 !important; background:#f1f5f9 !important; }
    /* Ensure select/dropdown editors in Data Editor are white with visible text */
    .stDataEditor [data-baseweb="select"] { background:#ffffff !important; }
    .stDataEditor [data-baseweb="select"] * { background:#ffffff !important; color:#111827 !important; }
    .stDataEditor [data-baseweb="select"] div { background:#ffffff !important; color:#111827 !important; border-color:#d1d5db !important; }
    .stDataEditor [data-baseweb="select"] [role="combobox"] { background:#ffffff !important; color:#111827 !important; }
    .stDataEditor [data-baseweb="select"] span { color:#111827 !important; }
    .stDataEditor [data-baseweb="select"] input { background:#ffffff !important; color:#111827 !important; }
    .stDataEditor [role="listbox"] { background:#ffffff !important; border:1px solid #d1d5db !important; }
    .stDataEditor [role="option"] { background:#ffffff !important; color:#111827 !important; padding:6px 12px !important; }
    .stDataEditor [role="option"]:hover { background:#f3f4f6 !important; color:#111827 !important; }
    .stDataEditor [role="option"][aria-selected="true"] { background:#eef6ff !important; color:#0f3355 !important; font-weight:600 !important; }
    .stDataEditor [role="gridcell"] div { white-space:normal !important; word-break:break-word !important; }
    .stDataEditor [data-testid="stDataFrameScrollable"] { overflow-x:auto !important; }
    .stDataEditor [role="row"]:nth-child(odd) [role="gridcell"],
    .stDataEditor [role="row"]:nth-child(even) [role="gridcell"] { background:#f8fafc !important; }
    .stDataEditor [role="row"]:hover [role="gridcell"] { background:#e2e8f0 !important; }
    /* Export bar harmonized with light theme */
    .export-row { background:rgba(255,255,255,0.70) !important; backdrop-filter:blur(10px); padding:10px 14px; border:1px solid #c1d7e8; border-radius:16px; margin:18px 0 26px; display:flex; align-items:center; gap:10px; flex-wrap:wrap; box-shadow:0 4px 18px -8px rgba(31,78,120,0.25), 0 2px 6px -2px rgba(31,78,120,0.18); }
    .loader-row { display:flex; align-items:center; gap:12px; margin:12px 0 20px; padding:12px 18px; background:rgba(255,215,113,0.12); border:1px solid #d6a736; border-radius:14px; box-shadow:0 4px 12px -4px rgba(0,0,0,0.45); }
    .spinner-small { width:22px; height:22px; border:3px solid #28426f; border-top-color:#4c92ff; border-radius:50%; animation:spin .7s linear infinite; box-sizing:border-box; }
    @keyframes spin { to { transform:rotate(360deg); } }
    .running-label { font-size:14px; font-weight:600; color:#111827; letter-spacing:.3px; }
    /* Alerts (warning/info/success/error): remove background, force black text */
    .stAlert, [data-testid="stAlert"] { background:transparent !important; border:none !important; box-shadow:none !important; }
    .stAlert [role="alert"], [data-testid="stAlert"] [role="alert"] { background:transparent !important; }
    .stAlert *, [data-testid="stAlert"] * { color:#000000 !important; }
    /* Force all BaseWeb select dropdown menus (rendered in portals) to be white with dark text */
    [data-baseweb="popover"] { background:#ffffff !important; border:1px solid #d1d5db !important; }
    [data-baseweb="popover"] * { background:#ffffff !important; color:#111827 !important; }
    [data-baseweb="menu"] { background:#ffffff !important; border:1px solid #d1d5db !important; box-shadow:0 4px 12px rgba(0,0,0,0.15) !important; }
    [data-baseweb="menu"] ul { background:#ffffff !important; }
    [data-baseweb="menu"] li { background:#ffffff !important; color:#111827 !important; padding:8px 12px !important; }
    [data-baseweb="menu"] li:hover { background:#f3f4f6 !important; color:#111827 !important; }
    [data-baseweb="menu"] [aria-selected="true"] { background:#eef6ff !important; color:#0f3355 !important; font-weight:600 !important; }
    /* Additional aggressive overrides for dropdown portal */
    div[role="listbox"] { background:#ffffff !important; border:1px solid #d1d5db !important; }
    div[role="listbox"] * { color:#111827 !important; }
    div[role="option"] { background:#ffffff !important; color:#111827 !important; }
    div[role="option"]:hover { background:#f3f4f6 !important; }
    div[role="option"][aria-selected="true"] { background:#eef6ff !important; color:#0f3355 !important; }
    ul[role="listbox"] { background:#ffffff !important; }
    ul[role="listbox"] li { background:#ffffff !important; color:#111827 !important; }
    /* Target any layered elements that might be dark */
    [class*="Popover"] { background:#ffffff !important; }
    [class*="popover"] { background:#ffffff !important; }
    [class*="Menu"] { background:#ffffff !important; }
    [class*="menu"] { background:#ffffff !important; }
    [class*="Dropdown"] { background:#ffffff !important; color:#111827 !important; }
    [class*="dropdown"] { background:#ffffff !important; color:#111827 !important; }

    /* Ensure select editor cells match Agent 1 Priority background */
    /* Match both Agent 1 Priority and Agent 2 Verification Level cells */
    .stDataEditor [role="gridcell"]:has([data-baseweb="select"]) { background:#ffffff !important; }
    .stDataEditor [role="gridcell"]:has([data-baseweb="select"]) * { background:#ffffff !important; }
</style>
"""
main_css = (main_css
            .replace("__BG__", bg)
            .replace("__FG__", fg)
            .replace("__SUBFG__", subfg)
            .replace("__CARD_BORDER__", card_border))
st.markdown(main_css, unsafe_allow_html=True)

# Final reinforcement for export buttons (ensures white background & black text in any theme/state)
st.markdown(
        """
        <style>
            .export-row div[data-testid="stDownloadButton"] button,
            .export-row button[kind*="download"],
            .export-row div.stButton > button {
                background:#ffffff !important; color:#111827 !important; border:1px solid #d1d5db !important;
                box-shadow:none !important; text-shadow:none !important; font-weight:600 !important;
            }
            .export-row div[data-testid="stDownloadButton"] button:hover,
            .export-row button[kind*="download"]:hover,
            .export-row div.stButton > button:hover { background:#f1f8ff !important; }
            .export-row div[data-testid="stDownloadButton"] button:active,
            .export-row button[kind*="download"]:active,
            .export-row div.stButton > button:active { background:#e2f1fb !important; }
            .export-row div[data-testid="stDownloadButton"] button:disabled,
            .export-row button[kind*="download"]:disabled,
            .export-row div.stButton > button:disabled { background:#ffffff !important; color:#9ca3af !important; opacity:1 !important; }
        </style>
        """,
        unsafe_allow_html=True,
)

# Final table styling override to ensure uniform white background & black text across all tables/editors
st.markdown(
        """
        <style id='final-table-override'>
            /* Force ALL dataframes & editors to light gray background + black text */
            .stDataFrame, .stDataEditor, .stDataFrame [data-testid="stDataFrameContainer"], .stDataFrame [data-testid="stDataFrameScrollable"],
            .stDataEditor [data-testid="stDataFrameContainer"], .stDataEditor [data-testid="stDataFrameScrollable"] { background:#f8fafc !important; }
            .stDataFrame *, .stDataEditor * { color:#111827 !important; }
            /* Headers (multiple markup patterns) */
            .stDataFrame thead th,
            .stDataFrame [role="columnheader"],
            .stDataFrame [data-testid*="header"],
            .stDataEditor [role="columnheader"],
            .stDataEditor [data-testid*="header"] { background:#f1f5f9 !important; color:#111827 !important; font-weight:600 !important; }
            /* Cells (table + virtualized grid) */
            .stDataFrame td, .stDataFrame th, .stDataFrame [role="cell"],
            .stDataEditor [role="gridcell"], .stDataEditor [role="cell"],
            .stDataEditor [data-testid*="cell"], .stDataFrame [data-testid*="cell"] {
                background:#f8fafc !important;
                border:1px solid #d3d8df !important;
                box-sizing:border-box;
            }
            /* Remove any theme-injected dark overlays */
            .stDataFrame [class*="overlay"], .stDataEditor [class*="overlay"] { background:transparent !important; }
            /* Light gray hover */
            .stDataFrame tbody tr:hover td,
            .stDataFrame [role="row"]:hover [role="cell"],
            .stDataEditor [role="row"]:hover [role*="cell"],
            .stDataEditor [data-testid*="row"] [data-testid*="cell"]:hover { background:#e2e8f0 !important; }
            /* Text wrapping & break long tokens */
            .stDataFrame [role="cell"] div,
            .stDataEditor [role="gridcell"] div,
            .stDataFrame td div, .stDataFrame th div { white-space:normal !important; word-break:break-word !important; }
            /* Ensure sticky header stays light gray */
            .stDataFrame thead tr th { background:#f1f5f9 !important; }
            /* Important: remove box-shadows that appear dark in some themes */
            .stDataFrame thead tr th { box-shadow:none !important; }
        </style>
        """,
        unsafe_allow_html=True,
)

# Compact & white table final override
st.markdown(
    """
    <style id='compact-white-table'>
        /* Force light gray background and compact density */
        .stDataFrame, .stDataEditor { background:#f8fafc !important; }
    .stDataFrame *, .stDataEditor * { color:#111827 !important; }
    .stDataFrame thead th, .stDataEditor [role="columnheader"] { background:#f1f5f9 !important; color:#111827 !important; font-weight:600 !important; }
        /* Compact cell sizing */
        .stDataFrame td, .stDataFrame th, .stDataFrame [role="cell"],
        .stDataEditor [role="gridcell"], .stDataEditor [role="cell"] {
             padding:4px 6px !important; font-size:13px !important; line-height:1.15 !important; background:#f8fafc !important;
        }
        /* Remove extra inner div padding */
        .stDataFrame td div, .stDataFrame th div, .stDataEditor [role="gridcell"] div { padding:0 !important; margin:0 !important; }
        /* Vertical scroll container */
        .stDataFrame [data-testid="stDataFrameScrollable"],
        .stDataEditor [data-testid="stDataFrameScrollable"] { max-height:65vh !important; overflow-y:auto !important; overflow-x:hidden !important; }
        /* Always wrap text */
        .stDataFrame td, .stDataFrame th, .stDataEditor [role="gridcell"] { white-space:normal !important; word-break:break-word !important; }
        /* Light gray hover */
        .stDataFrame tbody tr:hover td, .stDataEditor [role="row"]:hover [role="gridcell"] { background:#e2e8f0 !important; }
        /* Subtle grid lines */
        .stDataFrame td, .stDataFrame th, .stDataEditor [role="gridcell"], .stDataEditor [role="columnheader"] { border:1px solid #e2e5e9 !important; box-sizing:border-box; }
        /* Sticky header ensure shadowless */
        .stDataFrame thead tr th { position:sticky; top:0; z-index:20; box-shadow:0 1px 0 #e2e5e9; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
        """
        <style>
            .export-bar-new { display:flex; align-items:center; gap:10px; margin:8px 0 14px; justify-content:flex-start; }
            .export-bar-new .export-label { font-weight:650; color:#0d5ea8; display:inline-flex; align-items:center; gap:6px; }
            .export-bar-new .export-label .icon-svg, .export-bar-new .export-label .export-icon svg { margin-right:2px; position:relative; top:0.5px; }
            .export-bar-new .seg-group { display:inline-flex; background:#ffffff; border:1px solid #1d6fb8; border-radius:4px; overflow:hidden; }
            .export-bar-new .seg-btn { text-decoration:none; font-size:13px; font-weight:500; color:#0d5ea8; padding:6px 18px; background:#ffffff; border-right:1px solid #1d6fb8; line-height:1.1; display:inline-flex; align-items:center; justify-content:center; }
            .export-bar-new .seg-btn.last { border-right:none; }
            .export-bar-new .seg-btn:hover { background:#f0f7ff; }
            .export-bar-new .seg-btn:active { background:#e0effd; }
            .export-bar-new .seg-btn:focus { outline:2px solid #1d6fb8; outline-offset:-2px; }
            /* keep label and buttons nearby (no extra spacing) */
            .export-bar-new { justify-content:flex-start; }
            .export-bar-new .trace-toggle-wrapper { margin-left:auto; }
            .export-bar-new .trace-btn { border:1px solid #1d6fb8; border-radius:4px; margin-left:12px; }
            .export-bar-new .trace-btn { border-right:1px solid #1d6fb8 !important; }
            .export-bar-new .trace-btn:hover { background:#f0f7ff; }
            @media (max-width:640px){ .export-bar-new { flex-wrap:wrap; } }
            /* Hide old export-row based button layout */
            .export-row { display:none !important; }
        </style>
        """,
        unsafe_allow_html=True,
)

# Height override to ensure large tables remain visible with vertical scroll
st.markdown(
        """
        <style id='table-height-override'>
            .stDataFrame [data-testid="stDataFrameScrollable"],
            .stDataEditor [data-testid="stDataFrameScrollable"] { max-height:75vh !important; overflow-y:auto !important; }
            /* Visible custom scrollbar (WebKit) */
            .stDataFrame [data-testid="stDataFrameScrollable"]::-webkit-scrollbar,
            .stDataEditor [data-testid="stDataFrameScrollable"]::-webkit-scrollbar { width:10px; }
            .stDataFrame [data-testid="stDataFrameScrollable"]::-webkit-scrollbar-track,
            .stDataEditor [data-testid="stDataFrameScrollable"]::-webkit-scrollbar-track { background:#f1f1f1; }
            .stDataFrame [data-testid="stDataFrameScrollable"]::-webkit-scrollbar-thumb,
            .stDataEditor [data-testid="stDataFrameScrollable"]::-webkit-scrollbar-thumb { background:#c3c7cc; border-radius:6px; }
            .stDataFrame [data-testid="stDataFrameScrollable"]::-webkit-scrollbar-thumb:hover,
            .stDataEditor [data-testid="stDataFrameScrollable"]::-webkit-scrollbar-thumb:hover { background:#9aa0a6; }
        </style>
        """,
        unsafe_allow_html=True,
)

# Global: Force black text in all Text Areas (including disabled previews)
st.markdown(
        """
        <style id='textareas-black'>
            /* Text area text color and background */
            .stTextArea textarea, .stTextArea textarea:disabled {
                color:#000000 !important;
                -webkit-text-fill-color:#000000 !important; /* Safari/Chrome disabled text */
                caret-color:#000000 !important;
                background:#ffffff !important;
            }
            /* Label/title color */
            .stTextArea label, .stTextArea label span, .stTextArea label p {
                color:#000000 !important;
            }
            /* Placeholder color (keep very dark for readability) */
            .stTextArea textarea::placeholder { color:#111111 !important; opacity:1; }
        </style>
        """,
        unsafe_allow_html=True,
)

# (Removed old fixed overlay logo injection; logo now integrated inside nav bar)

# --- Navigation Bar (re-added simplified) ---
nav_pages = [
    ("Home", "Home"),
    ("Agent 1: SYS.1 Elicitation", "SYS1"),
    ("Agent 2: SYS.2 Analysis", "SYS2"),
    ("Agent 3: SYS.2 Review", "Review"),
    ("Agent 4: SYS.5 Testcase Generation", "SYS5"),
    ("Agent 5: Manager", "Manager"),
]
nav_parts: list[str] = []
for label, pid in nav_pages:
    if st.session_state.page == pid:
        nav_parts.append(f"<span class='nav-link active'>{label}</span>")
    else:
        nav_parts.append(f"<a class='nav-link' href='?page={pid}' target='_self'>{label}</a>")
bar_bg = "#ffffff"  # force white navbar
# Attempt to inline logo as base64 so it always loads (works with Streamlit static paths and avoids relative path breakage)
import os, base64
logo_html = "<div class='nav-logo-missing' style='font-size:13px; font-weight:600; padding:4px 10px; border:1px solid #d1d5db; border-radius:6px; background:#ffffff; color:#555;'>Logo</div>"
# Try multiple possible logo file names / extensions.
_logo_candidates = [
    os.path.join('static','logo_otl','otl_logo.png'),
    os.path.join('static','logo_otl','otl_logo.jpg'),
    os.path.join('static','logo_otl','otl_logo.jpeg'),
    os.path.join('static','logo_otl','otl_logo.svg'),
    os.path.join('static','logo_otl','logo.png'),
    os.path.join('static','logo_otl','logo.jpg'),
    os.path.join('static','logo_otl','logo.svg'),
]
_picked_path = None
for _cand in _logo_candidates:
    if os.path.exists(_cand):
        _picked_path = _cand
        break
if _picked_path:
    try:
        ext = os.path.splitext(_picked_path)[1].lower()
        mime = 'image/png'
        if ext in ['.jpg','.jpeg']:
            mime = 'image/jpeg'
        elif ext == '.svg':
            mime = 'image/svg+xml'
        with open(_picked_path,'rb') as _lf:
            _raw = _lf.read()
            if ext == '.svg':
                # SVG is text; keep as-is
                _b64 = base64.b64encode(_raw).decode('utf-8')
            else:
                _b64 = base64.b64encode(_raw).decode('utf-8')
        # Single-line HTML to avoid Streamlit formatting interpreting indentation as code
        logo_html = (
            f"<a href='?page=Home' target='_self' class='nav-logo-link' "
            f"style='display:inline-flex;align-items:center;text-decoration:none;'>"
            f"<img src='data:{mime};base64,{_b64}' alt='Logo' "
            f"style='height:42px;width:auto;display:block;object-fit:contain;'/>"
            f"</a>"
        )
    except Exception:
        pass
else:
    # Provide a subtle debug hint listing attempted paths (only visible if user inspects or no image found)
    attempted = '<br/>'.join(_logo_candidates[:4])
    logo_html = ("<div class='nav-logo-missing' title='Logo not found. Place one of: otl_logo.(png|jpg|jpeg|svg) in static/logo_otl/' "
                 "style='font-size:13px;font-weight:600;padding:4px 10px;border:1px dashed #d1d5db;border-radius:6px;background:#ffffff;color:#b91c1c;'>Logo</div>")

# Build user badge HTML for navbar with logout button
user_badge_html = ""
logout_button_html = ""
logout_key = f"logout_btn_{st.session_state.get('page', 'Home')}"

if st.session_state.get('auth_name'):
    user_name = st.session_state.get('auth_name', 'User')
    user_username = st.session_state.get('auth_username', '')
    
    # Check if logout was clicked
    if st.session_state.get(logout_key, False):
        # Clear all session state
        keys_to_clear = list(st.session_state.keys())
        for key in keys_to_clear:
            del st.session_state[key]
        st.rerun()
    
    user_badge_html = (
        f"<div style='display:flex;align-items:center;gap:6px;padding:6px 12px;background:#f3f4f6;border:1px solid #e5e7eb;border-radius:6px;font-size:13px;'>"
        f"<span style='color:#374151;font-weight:500;'>👤 {user_name}</span>"
        f"<span style='color:#6b7280;font-size:11px;'>({user_username})</span>"
        f"</div>"
    )
    
    # Add session timer
    timer_html = (
        f"<div id='session-timer' style='display:flex;align-items:center;gap:6px;padding:6px 12px;background:#eff6ff;border:1px solid #dbeafe;border-radius:6px;font-size:12px;'>"
        f"<span style='color:#1e40af;'>⏱️</span>"
        f"<span id='timer-display' style='color:#1e40af;font-weight:600;'>0:00</span>"
        f"</div>"
    )
    
    logout_button_html = (
        f"<div id='logout-trigger' style='display:flex;align-items:center;gap:6px;padding:6px 16px;background:#dc2626;color:white;border:1px solid #b91c1c;border-radius:6px;font-size:13px;font-weight:500;cursor:pointer;transition:background 0.2s;' "
        f"onmouseover='this.style.background=\"#b91c1c\"' onmouseout='this.style.background=\"#dc2626\"'>"
        f"<span>🚪</span><span>Logout</span>"
        f"</div>"
    )
    
elif st.session_state.get('auth_username') is None and stauth is not None:
    user_badge_html = "<div style='padding:6px 12px;background:#fef3c7;border:1px solid #fde047;border-radius:6px;font-size:12px;color:#92400e;'>🔒 Guest</div>"
    timer_html = ""

nav_html = (
    f"<div class='nav-bar' style=\"background:{bar_bg};padding:4px 12px;border-radius:28px;display:flex;align-items:center;justify-content:space-between;gap:16px;\">"
    f"<div class='nav-links' style='display:flex;align-items:center;flex-wrap:wrap;gap:0;'>{''.join(nav_parts)}</div>"
    f"<div style='flex:0 0 auto;display:flex;align-items:center;gap:8px;'>{user_badge_html}{timer_html}{logout_button_html}<div class='nav-logo' style='display:flex;align-items:center;'>{logo_html}</div></div>"
    f"</div><div style='height:78px'></div>"
)
st.markdown("""
<style>
  .nav-bar img { image-rendering:auto; }
  .nav-logo-link:hover img { filter:brightness(0.95); }
  /* Hide Streamlit's default user menu */
  [data-testid="stSidebarUserContent"] { display: none !important; }
  button[kind="header"] { display: none !important; }
</style>
""", unsafe_allow_html=True)
st.markdown(nav_html, unsafe_allow_html=True)

# Add JavaScript to wire up the navbar logout button and live timer
if st.session_state.get('auth_name'):
    import streamlit.components.v1 as components
    import time
    login_timestamp = st.session_state.get('login_timestamp', time.time())
    
    components.html(f"""
    <script>
    // Wire up logout button
    (function wireLogout() {{
        const navLogout = parent.document.getElementById('logout-trigger');
        if (navLogout && !navLogout.dataset.wired) {{
            navLogout.dataset.wired = 'true';
            navLogout.addEventListener('click', function(e) {{
                e.preventDefault();
                e.stopPropagation();
                // Clear session by reloading with logout parameter
                parent.window.location.href = parent.window.location.origin + parent.window.location.pathname + '?action=logout';
            }});
            console.log('Logout button wired successfully');
        }} else if (!navLogout) {{
            // Retry after a short delay if element not found
            setTimeout(wireLogout, 100);
        }}
    }})();
    
    // Live session timer
    (function updateTimer() {{
        const loginTime = {login_timestamp};
        const timerDisplay = parent.document.getElementById('timer-display');
        
        function formatTime(seconds) {{
            const hrs = Math.floor(seconds / 3600);
            const mins = Math.floor((seconds % 3600) / 60);
            const secs = seconds % 60;
            
            if (hrs > 0) {{
                return hrs + ':' + String(mins).padStart(2, '0') + ':' + String(secs).padStart(2, '0');
            }} else {{
                return mins + ':' + String(secs).padStart(2, '0');
            }}
        }}
        
        function tick() {{
            if (timerDisplay) {{
                const now = Date.now() / 1000;
                const elapsed = Math.floor(now - loginTime);
                timerDisplay.textContent = formatTime(elapsed);
            }}
        }}
        
        // Update immediately and then every second
        tick();
        setInterval(tick, 1000);
    }})();
    </script>
    """, height=0)

# --- Helper: responsive dataframe using our static white table ---
def responsive_dataframe(df: pd.DataFrame, hide_index: bool = True):
    if df is None:
        st.info("No data available.")
        return
    _df = df.copy()
    if hide_index and hasattr(_df, 'reset_index'):
        _df = _df.reset_index(drop=True)
    render_static_white_table(_df, max_height="70vh", table_id="responsive_df")

# --- Helper: Render user dropdown in sidebar ---
def render_user_sidebar():
    """Display user info and logout in sidebar if authenticated"""
    pass  # Removed sidebar user account display

# --- Page specific content ---
if st.session_state.page == "Home":

    # Centered Hero Section (logo, italic tagline, one-liner pill)
    st.markdown(
        """
<style>
/* Display font (falls back gracefully if blocked) */
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&display=swap');
/* Hero container */
.hero-wrap { text-align:center; padding:60px 12px 54px; position:relative; }
.hero-logo-title { font-family:'Bebas Neue', Arial, sans-serif; font-size:78px; font-weight:800; letter-spacing:4px; color:#000000; position:relative; display:inline-block; line-height:0.94; }
/* Soft background bloom behind the word */
/* Remove previous bloom background for a cleaner look */
.hero-logo-title::after { content:""; display:none; }
/* Neon single-tube look with white core and cyan halo */
/* Clean wordmark with subtle neon underline */
.hero-logo-title .hero-word { position:relative; z-index:1; color:#000000 !important; text-shadow: 0 1px 0 rgba(0,0,0,0.04); }
@keyframes underlineGlow {
    0%, 100% { width: 82%; opacity: 0.5; }
    50% { width: 92%; opacity: 0.85; }
}
.hero-logo-title .hero-underline { position:absolute; left:50%; transform:translateX(-50%); bottom:-8px; height:10px; border-radius:999px; background:
    linear-gradient(90deg, rgba(0,0,0,0) 0%, rgba(0,231,255,0.85) 18%, rgba(160,245,255,0.9) 50%, rgba(0,231,255,0.85) 82%, rgba(0,0,0,0) 100%);
    filter: blur(6px); box-shadow: 0 0 14px rgba(0,231,255,0.55), 0 0 26px rgba(0,231,255,0.35);
    animation: underlineGlow 3s ease-in-out infinite;
}
/* Removed pulsing for a calmer, professional look */
.hero-tagline { margin-top:14px; font-size:20px; font-style:italic; color:#000; }
.hero-one { margin:26px auto 0; display:inline-block; padding:14px 18px; border-radius:16px;
                        background: linear-gradient(180deg, #f0f7ff 0%, #ffffff 100%);
                        border:1px solid #dbeafe; color:#000; font-size:16px; box-shadow:0 6px 18px rgba(0,0,0,0.10); }
.hero-spacer { height:22px; }
</style>
<div class="hero-wrap">
    <div class="hero-logo-title">
        <span class="hero-word">WHALE</span>
        <span class="hero-underline" aria-hidden="true"></span>
    </div>
    <div class="hero-tagline">“We have to look into Everything”</div>
    <div class="hero-one">An AI-powered systems engineering co-pilot for the Automotive Industry</div>
</div>
    """,
    unsafe_allow_html=True,
    )

    # Compute dynamic SYS.1 status if available
    sys1_status = ""
    if "sys1_table_df" in st.session_state:
        try:
            _df = st.session_state["sys1_table_df"]
            if hasattr(_df, "empty") and not _df.empty:
                total = len(_df)
                approved = int((_df.get("Requirement Status") == "Approved").sum()) if "Requirement Status" in _df.columns else 0
                rejected = int((_df.get("Requirement Status") == "Rejected").sum()) if "Requirement Status" in _df.columns else 0
                drafts = max(total - approved - rejected, 0)
                sys1_status = f"{total} requirements extracted. ({approved} Approved, {rejected} Rejected, {drafts} Draft)"
        except Exception:
            pass

    # Single row card layout
    # Derive SYS.1 status classification (class & short label)
    sys1_status_lower = sys1_status.lower() if sys1_status else ""
    if "approved" in sys1_status_lower:
        sys1_status_class = "ready"
        sys1_status_label = "Approved"
    elif "requirements extracted" in sys1_status_lower:
        # Show number extracted if present at beginning (e.g., '12 requirements extracted')
        first_token = sys1_status.split(" requirements extracted")[0]
        sys1_status_class = "progress"
        sys1_status_label = first_token.strip()
    else:
        sys1_status_class = "pending"
        sys1_status_label = ""
    # If currently running, show Running…
    try:
        if st.session_state.get("running_sys1"):
            sys1_status_class = "progress"
            sys1_status_label = "Running…"
    except Exception:
        pass
    # Compute dynamic statuses for other cards
    # SYS.2 status
    sys2_status_class = "pending"
    sys2_status_label = ""
    try:
        if "sys2_table_df" in st.session_state:
            _df2 = st.session_state["sys2_table_df"]
            if hasattr(_df2, "empty") and not _df2.empty:
                n2 = len(_df2)
                # If we have a status column, prefer summarizing approvals
                if "Requirement Status" in _df2.columns:
                    _vals = _df2["Requirement Status"].astype(str).str.strip()
                    approved2 = int(_vals.str.contains("approve", case=False, na=False).sum())
                    rejected2 = int(_vals.str.contains("reject", case=False, na=False).sum())
                    drafts2 = max(n2 - approved2 - rejected2, 0)
                    if n2 > 0 and approved2 == n2:
                        sys2_status_class = "ready"
                        sys2_status_label = f"{approved2} Approved"
                    else:
                        sys2_status_class = "progress"
                        sys2_status_label = f"{n2} generated"
                else:
                    sys2_status_class = "progress"
                    sys2_status_label = f"{n2} generated"
    except Exception:
        pass
    # Running state override
    try:
        if st.session_state.get("running_sys2"):
            sys2_status_class = "progress"
            sys2_status_label = "Running…"
    except Exception:
        pass

    # Review status (Compliance Check)
    review_status_class = "pending"
    review_status_label = ""
    try:
        if "review_table_df" in st.session_state:
            _dfr = st.session_state["review_table_df"]
            if hasattr(_dfr, "empty") and not _dfr.empty:
                need_refine = 0
                if "Compliance Check" in _dfr.columns:
                    vals = _dfr["Compliance Check"].astype(str).str.strip().str.lower()
                    need_refine = int((vals != "yes").sum())
                if need_refine > 0:
                    review_status_class = "warning"
                    review_status_label = f"{need_refine} need refinement"
                else:
                    review_status_class = "ready"
                    review_status_label = "All compliant"
    except Exception:
        pass
    # Running state override
    try:
        if st.session_state.get("running_review"):
            review_status_class = "progress"
            review_status_label = "Running…"
    except Exception:
        pass

    # SYS.5 status (count tests)
    sys5_status_class = "pending"
    sys5_status_label = ""
    try:
        if "sys5_table_df" in st.session_state:
            _df5 = st.session_state["sys5_table_df"]
            if hasattr(_df5, "empty") and not _df5.empty:
                n5 = len(_df5)
                sys5_status_class = "progress" if n5 > 0 else "pending"
                sys5_status_label = f"{n5} tests" if n5 > 0 else ""
    except Exception:
        pass
    # Running state override
    try:
        if st.session_state.get("running_sys5"):
            sys5_status_class = "progress"
            sys5_status_label = "Running…"
    except Exception:
        pass

    # Manager badge (dynamic)
    manager_status_class = "pending"
    manager_status_label = ""
    try:
        if st.session_state.get("running_pipeline"):
            manager_status_class = "progress"
            manager_status_label = "Running…"
        else:
            have: list[bool] = []
            for key in ("sys1_table_df", "sys2_table_df", "review_table_df", "sys5_table_df"):
                dfv = st.session_state.get(key)
                try:
                    have.append(bool(getattr(dfv, "empty", True) is False))
                except Exception:
                    have.append(False)
            if any(have) and not all(have):
                manager_status_class = "warning"
                manager_status_label = "Partial Outputs"
            elif all(have):
                manager_status_class = "ready"
                manager_status_label = "Pipeline Complete"
    except Exception:
        pass

    st.markdown(
        f"""
        <style>
          .home-cards-row {{ display:flex; gap:22px; align-items:stretch; justify-content:space-between; flex-wrap:nowrap; }}
          .home-cards-row .card {{ flex:1 1 0; min-width:0; }}
          @media (max-width:1400px) {{ .home-cards-row {{ gap:18px; }} }}
          @media (max-width:1250px) {{ .home-cards-row {{ flex-wrap:wrap; }} }}
          .home-cards-row h3, .home-cards-row .muted, .home-cards-row b {{ color:#000000 !important; }}
          /* Reinforce: ensure every descendant text in cards uses pure black */
          .home-cards-row .card, .home-cards-row .card * {{ color:#000000 !important; }}
                    .status-line {{ margin-top:4px; font-size:13px; font-weight:600; display:flex; align-items:center; gap:6px; }}
                    .status-badge {{ display:inline-block; padding:4px 10px 5px; border-radius:999px; font-size:12px; font-weight:700; letter-spacing:.3px; line-height:1; border:1px solid transparent; color:#ffffff; }}
                    .status-badge.pending {{ background:#6b7280; }}
                    .status-badge.progress {{ background:#2564d6; }}
                    .status-badge.warning {{ background:#d97706; }}
                    .status-badge.ready {{ background:#16a34a; }}
          
          /* Animated Icons */
          .animated-icon {{ display: inline-block; font-size: 24px; margin-left: 8px; }}
          
          @keyframes search-pulse {{
            0%, 100% {{ transform: scale(1) rotate(0deg); }}
            50% {{ transform: scale(1.15) rotate(-10deg); }}
          }}
          .icon-search {{ animation: search-pulse 2s ease-in-out infinite; }}
          
          @keyframes pencil-write {{
            0%, 100% {{ transform: translateX(0) rotate(0deg); }}
            25% {{ transform: translateX(-3px) rotate(-8deg); }}
            75% {{ transform: translateX(3px) rotate(8deg); }}
          }}
          .icon-pencil {{ animation: pencil-write 1.5s ease-in-out infinite; }}
          
          @keyframes warning-flash {{
            0%, 100% {{ opacity: 1; transform: scale(1); }}
            50% {{ opacity: 0.7; transform: scale(1.1); }}
          }}
          .icon-warning {{ animation: warning-flash 2s ease-in-out infinite; }}
          
          @keyframes test-tube {{
            0%, 100% {{ transform: rotate(0deg) translateY(0); }}
            25% {{ transform: rotate(-5deg) translateY(-2px); }}
            75% {{ transform: rotate(5deg) translateY(-2px); }}
          }}
          .icon-test {{ animation: test-tube 2.5s ease-in-out infinite; }}
          
          @keyframes folder-bounce {{
            0%, 100% {{ transform: translateY(0) scale(1); }}
            50% {{ transform: translateY(-4px) scale(1.05); }}
          }}
          .icon-folder {{ animation: folder-bounce 2s ease-in-out infinite; }}
        </style>
        <div class='home-cards-row'>
            <a class="card-link" href="?page=SYS1" target="_self">
                <div class='card sys1' style='background:linear-gradient(135deg,#bbdefb,#90caf9); box-shadow:0px 4px 12px rgba(0,0,0,0.15);'>
                    <h3>Agent 1: SYS.1 Elicitation<span class='animated-icon icon-search'>🔎</span></h3>
                    <div class='muted'>Requirement elicitation, pre-processing, and SYS.1 drafting</div>
                </div>
            </a>
            <a class="card-link" href="?page=SYS2" target="_self">
                <div class='card sys2' style='background:linear-gradient(135deg,#c8e6c9,#a5d6a7); box-shadow:0px 4px 12px rgba(0,0,0,0.15);'>
                    <h3>Agent 2: SYS.2 Analysis<span class='animated-icon icon-pencil'>✏️</span></h3>
                    <div class='muted'>SYS.2 requirement drafting and structuring</div>
                </div>
            </a>
            <a class="card-link" href="?page=Review" target="_self">
                <div class='card review' style='background:linear-gradient(135deg,#ffe0b2,#ffcc80); box-shadow:0px 4px 12px rgba(0,0,0,0.15);'>
                    <h3>Agent 3: SYS.2 Review<span class='animated-icon icon-warning'>⚠️</span></h3>
                    <div class='muted'>SYS.2 review, compliance with IREB, ISO/IEC/IEEE 29148:2018 / ISO 26262 / INCOSE SE guidelines</div>
                </div>
            </a>
            <a class="card-link" href="?page=SYS5" target="_self">
                <div class='card sys5' style='background:linear-gradient(135deg,#f8bbd0,#f48fb1); box-shadow:0px 4px 12px rgba(0,0,0,0.15);'>
                    <h3>Agent 4: SYS.5 Testcase Generation<span class='animated-icon icon-test'>🧪</span></h3>
                    <div class='muted'>SYS.2 to SYS.5 test case generation</div>
                </div>
            </a>
            <a class="card-link" href="?page=Manager" target="_self">
                <div class='card manager' style='background:linear-gradient(135deg,#d1c4e9,#b39ddb); box-shadow:0px 4px 12px rgba(0,0,0,0.15);'>
                    <h3>Agent 5: Manager<span class='animated-icon icon-folder'>🗂️</span></h3>
                    <div class='muted neon-flow'>
                        Run workflow
                        <span class='step'>Customer Req</span>
                        <span class='neon-arrow'>➜</span>
                        <span class='step'>SYS.1</span>
                        <span class='neon-arrow alt'>➜</span>
                        <span class='step'>SYS.2</span>
                        <span class='neon-arrow'>➜</span>
                        <span class='step'>Review</span>
                        <span class='neon-arrow alt'>➜</span>
                        <span class='step'>SYS.5</span>
                        <span class='muted'>(CSV/Excel/Word/PDF/REQIF)</span>
                    </div>
                </div>
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Final alert override: remove background and force black text/icons
    st.markdown(
        """
        <style id='final-alert-override'>
            /* Wrapper */
            [data-testid="stAlert"] { background:transparent !important; border:none !important; box-shadow:none !important; }
            /* Inner containers */
            [data-testid="stAlert"] > div, [data-testid="stAlert"] [role="alert"],
            [data-testid="stAlert"] [data-testid*="notification"], [data-testid="stAlert"] [data-baseweb="notification"] { background:transparent !important; border:none !important; box-shadow:none !important; }
            /* Remove BaseWeb Notification left stripe */
            [data-testid="stAlert"] [data-baseweb="notification"]::before { display:none !important; }
            /* Markdown/text */
            [data-testid="stAlert"] *, [data-testid="stAlert"] p, [data-testid="stAlert"] span, [data-testid="stAlert"] div { color:#000000 !important; }
            /* Icons */
            [data-testid="stAlert"] svg, [data-testid="stAlert"] [data-testid="stIcon"] { color:#000000 !important; fill:#000000 !important; }
            /* Remove any nested backgrounds/borders/left-stripes within alert content */
            [data-testid="stAlert"] * { background:transparent !important; background-color:transparent !important; border:none !important; box-shadow:none !important; filter:none !important; backdrop-filter:none !important; }
            /* Tighten padding so it blends with the page */
            [data-testid="stAlert"] [data-baseweb="notification"] { padding:0 !important; }
            /* Explicitly target variant wrappers */
            [data-testid="stWarning"], [data-testid="stInfo"], [data-testid="stSuccess"], [data-testid="stError"] { background:transparent !important; border:none !important; box-shadow:none !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Copyright footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(
        """
        <style>
            @keyframes underlineGlow {
                0%, 100% { width: 40%; opacity: 0.4; }
                50% { width: 60%; opacity: 0.7; }
            }
            .copyright-container {
                position: relative;
                margin-top: 20px;
                padding: 20px 0;
                border-top: 1px solid #e2e8f0;
                text-align: center;
            }
            .copyright-container::after {
                content: '';
                position: absolute;
                bottom: 0;
                left: 50%;
                transform: translateX(-50%);
                height: 2px;
                background: linear-gradient(90deg, transparent, #3b82f6, transparent);
                animation: underlineGlow 3s ease-in-out infinite;
            }
        </style>
        <div class='copyright-container'>
            <p style='font-size: 8pt; color: #4b5563; line-height: 1.6; margin: 0;'>
                © Copyright 2025 Onward Technologies Limited. All Rights Reserved.<br>
                This material, including all text, graphics, code, and related content, is the proprietary property of Onward Technologies. 
                No part of this publication may be reproduced, distributed, or transmitted in any form or by any means without prior written 
                permission from Onward Technologies, except where such use is explicitly permitted under applicable law.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# SYS.1 page
elif st.session_state.page == "SYS1":
    st.header("🔎 SYS.1 Elicitation")
    st.caption("Transform high-level customer requirements into well-structured SYS.1 system requirements.")
    uploaded = st.file_uploader("Upload Customer Requirements (.txt, .docx, .pdf, .xlsx, .reqif, .json)", type=["txt","docx","pdf","xlsx","reqif","json"], key="sys1_upload")
    req_text: str = ""
    if uploaded is not None:
        try:
            _bytes = uploaded.read()
            _text, _ = load_file_for_agent(_bytes, uploaded.name, "SYS1")
            req_text = _text or ""
            if not req_text.strip() and uploaded.name.lower().endswith('.reqif'):
                st.warning("⚠️ REQIF file uploaded but no content extracted. The file may be empty or use an unsupported REQIF format. Please check the file contains valid SPEC-OBJECT elements.")
        except Exception as e:
            req_text = ""
            st.error(f"Error reading file: {str(e)}")
        preview_text = req_text[:5000] if req_text.strip() else "(No content extracted)"
        st.text_area("Preview (read-only)", preview_text, height=200, disabled=True)
        st.caption("Using uploaded file content for SYS.1 agent.")
    else:
        req_text = st.text_area("Or paste Customer Requirements:", height=200, placeholder="Paste requirements here… (e.g., ‘The system shall maintain speed at a setpoint…’) \nYou can also upload .txt, .docx, .pdf, .xlsx, .reqif, .json above.")

    # RAG toggle
    st.write("🔍 **Enable RAG Context** (Standards-based)")
    use_rag_sys1 = st.checkbox("Use standards to enhance requirements generation", value=False, key="rag_sys1",
                               help="Augment prompts with relevant context from automotive standards (ISO 26262, ISO 29148, IREB CPRE, etc.)")

    st.session_state.setdefault("running_sys1", False)
    run_sys1_clicked = st.button("🤖 Generate SYS.1 Requirements", disabled=st.session_state.get("running_sys1", False))
    if run_sys1_clicked and not st.session_state.get("running_sys1", False):
        if not req_text.strip():
            st.warning("Please upload a .txt, .docx, .pdf, .xlsx, .json, or .reqif file, or paste requirements, before running.")
        else:
            st.session_state.running_sys1 = True
            st.session_state.sys1_use_rag = use_rag_sys1
            st.rerun()
    
    # Check if we should display existing data
    display_existing = "sys1_table_df" in st.session_state and not st.session_state.get("running_sys1", False)
    
    if st.session_state.get("running_sys1"):
        _loader = st.empty()
        _loader.markdown("<div class='loader-row'><div class='spinner-small'></div><div class='running-label'>Processing SYS.1...</div></div>", unsafe_allow_html=True)
        result = run_single("SYS1", req_text, use_rag=st.session_state.get("sys1_use_rag", False))
        st.session_state.running_sys1 = False
        _loader.empty()
        st.success("Done ✅")
        st.caption("Output available in Word, PDF, XLSX, REQIF formats.")
        # Parse and present in requested tabular format
        sys1_list = result.get("SYS1", []) if isinstance(result, dict) else []
        try:
            df_raw = pd.DataFrame(sys1_list)
        except Exception:
            df_raw = pd.DataFrame()

        # Normalize / auto-fill Domain to one of [SW, HW, System, Mechanical]
        def _norm_domain(val, requirement_text=""):
            allowed = {"sw":"SW", "software":"SW", "hw":"HW", "hardware":"HW", "system":"System", "mechanical":"Mechanical"}
            s = str(val).strip() if val is not None else ""
            if s:
                key = s.lower()
                if key in allowed:
                    return allowed[key]
                # partial words
                for k in ("software","hardware","mechanical","system","sw","hw"):
                    if k in key:
                        return allowed.get(k, "System")
            # infer from requirement text
            return infer_domain(requirement_text)

        # Column mapping to match attached format
        col_map = {
            "Customer Req. ID": "Customer Req_ID",
            "Customer Requirement": "Customer Requirement",
            "SYS.1 Req. ID": "SYS.1 Req._ID",
            "SYS.1 Requirement": "SYS.1 Requirement",
            "Domain": "Domain",
            "Priority": "Priority",
            "Rationale": "Rationale",
            "Requirement Status": "Requirement Status",
        }
        # Build display df with required columns/order
        desired_cols = list(col_map.values()) + ["Actions"]
        disp_df = pd.DataFrame()
        if not df_raw.empty:
            # Rename available columns
            tmp = df_raw.rename(columns=col_map)
            # Determine likely requirement text column for inference
            req_col = next((c for c in tmp.columns if c.lower().startswith("sys.1 requirement")), None)
            if req_col is None:
                req_col = next((c for c in tmp.columns if c.lower().endswith("requirement")), None)
            # Apply normalization
            if "Domain" in tmp.columns:
                dom_vals = tmp["Domain"].tolist()
                inferred = []
                for i, v in enumerate(dom_vals):
                    base_txt = ""
                    if req_col and req_col in tmp.columns:
                        try:
                            base_txt = str(tmp.iloc[i][req_col])
                        except Exception:
                            base_txt = ""
                    inferred.append(_norm_domain(v, base_txt))
                tmp["Domain"] = inferred
            else:
                base_texts = tmp[req_col].astype(str).tolist() if req_col and req_col in tmp.columns else [""] * len(tmp)
                tmp["Domain"] = [infer_domain(t) for t in base_texts]
            # Ensure all desired columns exist
            for c in desired_cols:
                if c not in tmp.columns:
                    tmp[c] = ""
            disp_df = tmp[desired_cols].copy()
        else:
            # If parsing failed, show note
            st.info("No structured SYS.1 data parsed. Showing raw output:")
            st.json(result)

        # Interactive editor with Actions, if we have data
        if not disp_df.empty:
            # Store original
            st.session_state["sys1_table_df"] = disp_df
            
            # Auto-save immediately after generation
            try:
                _p = os.path.join(BASE_DIR, "data", "outputs", "sys1_requirements.xlsx")
                _saved = save_excel_styled_to_path(disp_df.drop(columns=["Actions"], errors="ignore"), _p, "SYS1")
                if _saved:
                    st.session_state["sys1_last_saved"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state["sys1_save_notification"] = f"✅ Auto-saved to: {_p}"
                else:
                    st.session_state["sys1_save_notification"] = "⚠️ Auto-save returned False"
            except Exception as e:
                st.session_state["sys1_save_notification"] = f"⚠️ Auto-save failed: {e}"
            
            # Remove Actions column for display
            display_df = disp_df.drop(columns=["Actions"], errors="ignore")
            
            st.write("### SYS.1 Requirements")
            
            # Display save notification if available
            if "sys1_save_notification" in st.session_state:
                notif = st.session_state["sys1_save_notification"]
                if "✅" in notif:
                    st.toast(notif, icon="✅")
                else:
                    st.toast(notif, icon="⚠️")
                # Clear after displaying once
                del st.session_state["sys1_save_notification"]
            
            # Show autosave timestamp badge
            _ts = st.session_state.get("sys1_last_saved")
            if _ts:
                st.markdown(
                    f"<span style='display:inline-block;padding:4px 12px;border-radius:999px;background:#e6ffed;color:#065f46;font-size:13px;border:1px solid #34d399;margin-bottom:10px;'>✅ Auto-saved at {_ts}</span>",
                    unsafe_allow_html=True,
                )
            
            st.caption(f"Showing {len(display_df)} extracted requirements. Edit the table below and click 'Save Changes'.")
            
            # Configure editable columns
            column_config = {}
            for col in display_df.columns:
                if col == "Priority":
                    column_config[col] = st.column_config.SelectboxColumn(
                        col,
                        options=["High", "Medium", "Low"],
                        help="Select requirement priority",
                        width="medium"
                    )
                elif col in ["SYS.1 Requirement", "Domain", "Rationale"]:
                    column_config[col] = st.column_config.TextColumn(
                        col,
                        help=f"Editable: {col}",
                        width="large" if col == "SYS.1 Requirement" else "medium"
                    )
                else:
                    column_config[col] = st.column_config.TextColumn(
                        col,
                        disabled=True,
                        width="medium"
                    )
            
            # Editable data editor (collapsed by default)
            with st.expander("✏️ Edit Table", expanded=False):
                edited_df = st.data_editor(
                    display_df,
                    use_container_width=True,
                    num_rows="fixed",
                    column_config=column_config,
                    hide_index=True,
                    key="sys1_editor",
                    disabled=False
                )
            
            # Auto-update session state with edits
            st.session_state["sys1_table_df"] = edited_df.copy()
            
            # Wrapped view for readability (non-editable)
            render_static_white_table(edited_df, max_height="70vh", table_id="sys1_wrapped")

            # Save button + status badge (persist only on click)
            c_btn, c_badge = st.columns([1, 5])
            with c_btn:
                save_clicked = st.button("💾 Save Changes", key="save_sys1")
            if save_clicked:
                try:
                    # Use the latest edited data from session state
                    save_df = st.session_state.get("sys1_table_df", edited_df)
                    _p = os.path.join(BASE_DIR, "data", "outputs", "sys1_requirements.xlsx")
                    _saved = save_excel_styled_to_path(save_df, _p, "SYS1")
                    if _saved:
                        notify_saved(_saved, label="SYS.1")
                        st.session_state["sys1_last_saved"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.toast("✅ Changes saved!", icon="✅")
                    else:
                        st.toast("❌ Save operation returned False. Check file permissions.", icon="❌")
                except Exception as e:
                    st.toast(f"❌ Save failed: {str(e)}", icon="❌")
            with c_badge:
                _ts = st.session_state.get("sys1_last_saved")
                if _ts:
                    st.markdown(
                        f"<span style='display:inline-block;padding:2px 8px;border-radius:999px;background:#e6ffed;color:#065f46;font-size:12px;border:1px solid #34d399;'>Saved at {_ts}</span>",
                        unsafe_allow_html=True,
                    )

            render_export_buttons(edited_df, base_name="SYS1_requirements", label="Export Options")
            st.caption("Note: Domain is auto-inferred to one of SW, HW, System, Mechanical. You can edit directly in the table.")
            # Always render the Agent 1 traceability dashboard below the results
            st.write("---")
            render_traceability_section()
    
    # Display existing SYS.1 table if available and not currently running
    elif display_existing:
        existing_df = st.session_state["sys1_table_df"]
        # Remove Actions column for display
        display_df = existing_df.drop(columns=["Actions"], errors="ignore")
        
        st.write("### SYS.1 Requirements")
        st.caption(f"Showing {len(display_df)} extracted requirements. Edit the table below and click 'Save Changes'.")
        
        # Configure editable columns
        column_config = {}
        for col in display_df.columns:
            if col == "Priority":
                column_config[col] = st.column_config.SelectboxColumn(
                    col,
                    options=["High", "Medium", "Low"],
                    help="Select requirement priority",
                    width="medium"
                )
            elif col in ["SYS.1 Requirement", "Domain", "Rationale"]:
                column_config[col] = st.column_config.TextColumn(
                    col,
                    help=f"Editable: {col}",
                    width="large" if col == "SYS.1 Requirement" else "medium"
                )
            else:
                column_config[col] = st.column_config.TextColumn(
                    col,
                    disabled=True,
                    width="medium"
                )
        
        # Editable data editor (collapsed by default)
        with st.expander("✏️ Edit Table", expanded=False):
            edited_df = st.data_editor(
                display_df,
                use_container_width=True,
                num_rows="fixed",
                column_config=column_config,
                hide_index=True,
                key="sys1_editor",
                disabled=False
            )
        
        # Auto-update session state with edits
        st.session_state["sys1_table_df"] = edited_df.copy()
        
        # Wrapped view for readability (non-editable)
        render_static_white_table(edited_df, max_height="70vh", table_id="sys1_wrapped")

        # Save button + status badge (persist only on click)
        c_btn, c_badge = st.columns([1, 5])
        with c_btn:
            save_clicked = st.button("💾 Save Changes", key="save_sys1")
        if save_clicked:
            try:
                # Use the latest edited data from session state
                save_df = st.session_state.get("sys1_table_df", edited_df)
                _p = os.path.join(BASE_DIR, "data", "outputs", "sys1_requirements.xlsx")
                _saved = save_excel_styled_to_path(save_df, _p, "SYS1")
                if _saved:
                    notify_saved(_saved, label="SYS.1")
                    st.session_state["sys1_last_saved"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.toast("✅ Changes saved!", icon="✅")
                else:
                    st.toast("❌ Save operation returned False. Check file permissions.", icon="❌")
            except Exception as e:
                st.toast(f"❌ Save failed: {str(e)}", icon="❌")
        with c_badge:
            _ts = st.session_state.get("sys1_last_saved")
            if _ts:
                st.markdown(
                    f"<span style='display:inline-block;padding:2px 8px;border-radius:999px;background:#e6ffed;color:#065f46;font-size:12px;border:1px solid #34d399;'>Saved at {_ts}</span>",
                    unsafe_allow_html=True,
                )
        
        render_export_buttons(edited_df, base_name="SYS1_requirements", label="Export Options")
        st.caption("Note: Domain is auto-inferred to one of SW, HW, System, Mechanical. You can edit directly in the table.")
        # Always render the Agent 1 traceability dashboard below the results
        st.write("---")
        render_traceability_section()
    
    if st.button("⬅ Back to Home"):
        go_to("Home")


# SYS.2 page
elif st.session_state.page == "SYS2":
    st.header("✏️ SYS.2 Analysis")
    st.caption("Transform SYS.1 outputs into clear, structured SYS.2 system requirements.")
    up_sys2 = st.file_uploader("Upload SYS.1 Requirements (.txt, .docx, .pdf, .xlsx, .reqif, .json)", type=["json","txt","docx","pdf","xlsx","reqif"], key="sys2_upload")
    req_text = ""
    if up_sys2 is not None:
        try:
            _bytes = up_sys2.read()
            _text, _recs = load_file_for_agent(_bytes, up_sys2.name, "SYS2")
            raw = json.dumps(_recs) if _recs is not None else (_text or "")
        except Exception:
            raw = ""
        req_text = raw
        st.text_area("Preview (read-only)", raw[:6000], height=180, disabled=True)
        st.caption("Using uploaded file content for SYS.2 agent.")
    else:
        req_text = st.text_area("Or paste SYS.1 Requirements (JSON):", height=200, placeholder="Paste a JSON array of SYS.1 items or raw text that describes the SYS.1 requirements…")
    # Removed input-shape validation and 'Use last SYS.1 output' helper per request

    # RAG toggle
    st.write("🔎 **Enable RAG Context** (Standards-based)")
    use_rag_sys2 = st.checkbox("Use standards to enhance requirements generation", value=False, key="rag_sys2",
                               help="Augment prompts with relevant context from automotive standards (ISO 26262, AUTOSAR, etc.)")

    st.session_state.setdefault("running_sys2", False)
    run_sys2_clicked = st.button("✏️ Generate SYS.2 Requirements", disabled=st.session_state.get("running_sys2", False))
    if run_sys2_clicked and not st.session_state.get("running_sys2", False):
        if not req_text.strip():
            st.warning("Please upload a .txt, .docx, .pdf, .xlsx, .json, or .reqif file, or paste requirements, before running.")
        else:
            st.session_state.running_sys2 = True
            st.session_state.sys2_use_rag = use_rag_sys2
            st.rerun()
    result = None
    # If we already have a table from a previous run and we're not currently running,
    # keep showing it so edits don't make the table disappear on rerun.
    display_existing_sys2 = "sys2_table_df" in st.session_state and not st.session_state.get("running_sys2", False)
    if st.session_state.get("running_sys2"):
        _loader2 = st.empty()
        _loader2.markdown("<div class='loader-row'><div class='spinner-small'></div><div class='running-label'>Processing SYS.2...</div></div>", unsafe_allow_html=True)
        req_text_used = req_text
        if st.session_state.get("sys2_input_override"):
            req_text_used = st.session_state.pop("sys2_input_override")
        result = run_single("SYS2", req_text_used, use_rag=st.session_state.get("sys2_use_rag", False))
        st.session_state.running_sys2 = False
        _loader2.empty()
        st.success("Done ✅")
        st.caption("Output available in Word, PDF, XLSX, REQIF formats.")
        # Desired SYS.2 output format columns
        sys2_cols = [
            "SYS.1 Req. ID",
            "SYS.1 Requirement",
            "SYS.2 Req ID",
            "SYS.2 Requirement",
            "TYPE",
            "Verification Level",
            "Verification Criteria",
            "Domain",
            "Requirement Status",
        ]
        alias_map = {
            "SYS.1 Req. ID": ["SYS.1 Req. ID", "SYS1 Req ID", "SYS1 Requirement ID"],
            "SYS.1 Requirement": ["SYS.1 Requirement", "SYS1 Requirement"],
            "SYS.2 Req ID": ["SYS.2 Req ID", "SYS.2 Req. ID", "SYS2 Req ID"],
            "SYS.2 Requirement": ["SYS.2 Requirement", "SYS2 Requirement"],
            "TYPE": ["TYPE", "Type"],
            "Verification Criteria": ["Verification Criteria", "Criteria"],
            "Verification Level": ["Verification Level", "Level"],
            "Domain": ["Domain"],
            "Requirement Status": ["Requirement Status", "Status"],
        }
        items = []
        if isinstance(result, dict):
            candidate = result.get("SYS2") or []
            if isinstance(candidate, list):
                items = candidate
        # Helper: format verification criteria into readable statements
        def _format_verification_criteria(val, method_text: str = "") -> str:
            method_text = (method_text or "").strip()
            def _with_method(txt: str) -> str:
                if method_text:
                    # Collapse any repeated spaces/comma spacing in method string
                    m = method_text.replace(" ,", ",").strip()
                    return f"{txt} via {m}." if not txt.endswith(".") else f"{txt[:-1]} via {m}."
                return txt if txt.endswith(".") else txt + "."

            def _from_mapping(d: dict) -> str:
                # Normalize common keys
                keys = {k.lower(): v for k, v in d.items() if v not in (None, "")}
                metric = keys.get("metric") or keys.get("parameter") or keys.get("criterion") or keys.get("measure")
                threshold = keys.get("threshold") or keys.get("limit")
                rng = keys.get("range")
                target = keys.get("target") or keys.get("value")
                unit = keys.get("unit")
                cond = keys.get("condition") or keys.get("conditions")

                # Build the most natural sentence we can
                if metric and threshold:
                    base = f"Verify {str(metric).lower()} is {threshold}"
                    if cond:
                        base += f" under {cond}"
                    return _with_method(base)
                if metric and rng:
                    base = f"Verify {str(metric).lower()} is within {rng}"
                    if cond:
                        base += f" under {cond}"
                    return _with_method(base)
                if metric and (target or unit):
                    tgt = target
                    if not tgt and unit:
                        tgt = unit
                    base = f"Verify {str(metric).lower()} meets {tgt}"
                    if cond:
                        base += f" under {cond}"
                    return _with_method(base)

                # Fallback: flatten key-value pairs
                flat = "; ".join(f"{k.capitalize()}: {v}" for k, v in d.items() if v not in (None, ""))
                return _with_method(f"Verify criteria: {flat}")

            # Dispatch based on type
            if isinstance(val, dict):
                return _from_mapping(val)
            if isinstance(val, list):
                parts = []
                for item in val:
                    if isinstance(item, (dict, list)):
                        parts.append(_format_verification_criteria(item, method_text))
                    else:
                        s = str(item).strip()
                        parts.append(_with_method(s) if s else "")
                return "\n".join([p for p in parts if p])
            if isinstance(val, str):
                s = val.strip()
                # Try parsing JSON-like dict/list strings
                if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
                    try:
                        parsed = json.loads(s.replace("'", '"'))
                        return _format_verification_criteria(parsed, method_text)
                    except Exception:
                        try:
                            import ast
                            parsed = ast.literal_eval(s)
                            return _format_verification_criteria(parsed, method_text)
                        except Exception:
                            pass
                # Try extracting Metric/Threshold patterns from free text
                try:
                    m = re.search(r"[Mm]etric[\s'\":]*([^,;\n]+)", s)
                    t = re.search(r"[Tt]hreshold[\s'\":]*([^,;\n]+)", s)
                    if m and t:
                        metric = m.group(1).strip(" '\"")
                        thr = t.group(1).strip(" '\"")
                        return _with_method(f"Verify {metric.lower()} is {thr}")
                except Exception:
                    pass
                return _with_method(s) if s else ""
            # Unknown type -> string fallback
            try:
                return _with_method(str(val)) if val is not None else ""
            except Exception:
                return ""

        # Helpers for normalization/inference
        def _norm_type(val: str, requirement_text: str = "") -> str:
            v = (str(val) if val is not None else "").strip().lower()
            if v in ("functional",):
                return "Functional"
            if v in ("non-functional", "nonfunctional", "quality", "performance", "safety", "security", "reliability"):
                return "Non-Functional"
            if v in ("information", "informational", "note"):
                return "Information"
            # Heuristic: presence of interfaces/timing/diagnostics words → Functional; constraints → Non-Functional
            t = (requirement_text or "").lower()
            if any(k in t for k in ("shall interface", "shall use", "shall communicate", "calculate", "control", "monitor", "detect", "respond")):
                return "Functional"
            if any(k in t for k in ("latency", "response time", "throughput", "availability", "asil", "safety", "security", "reliability", "mtbf", "accuracy", "tolerance")):
                return "Non-Functional"
            return "Information"

        def _norm_domain_sys2(val: str, req_text: str = "") -> str:
            # Reuse Agent 1 inference
            allowed = {"sw":"SW", "software":"SW", "hw":"HW", "hardware":"HW", "system":"System", "mechanical":"Mechanical"}
            s = str(val).strip() if val is not None else ""
            if s:
                key = s.lower()
                if key in allowed:
                    return allowed[key]
                for k in ("software","hardware","mechanical","system","sw","hw"):
                    if k in key:
                        return allowed.get(k, "System")
            return infer_domain(req_text)

        def _norm_level(method_text: str, type_text: str, req_text: str = "") -> str:
            m = (method_text or "").lower()
            t = (type_text or "").lower()
            # Heuristic: integration-like methods/contexts now map to SYS.4 per swapped labels
            if any(k in m for k in ("hil", "road test", "integration")) or any(k in req_text.lower() for k in ("interface", "can", "ethernet", "sensor", "actuator")):
                return "System Integration Test (SYS.4)"
            # Otherwise default to SYS.5 qualification per new mapping
            return "System Qualification Test (SYS.5)"

        rows = []
        for it in items:
            if not isinstance(it, dict):
                continue
            row = {}
            for col, aliases in alias_map.items():
                val = ""
                for a in aliases:
                    if a in it and it[a] not in (None, ""):
                        val = it[a]
                        break
                # Normalize values
                if col == "Verification Criteria":
                    # Use any provided method text from the raw item to enrich criteria, even though it's not shown as a column
                    method_text = str(it.get("Verification Method") or it.get("Verification") or it.get("Method") or "").strip()
                    val = _format_verification_criteria(val, method_text)
                else:
                    if isinstance(val, list):
                        val = " | ".join(str(x) for x in val)
                row[col] = val
            # Post-normalization: TYPE, Domain, Verification Level
            base_req_text = str(row.get("SYS.2 Requirement") or it.get("SYS.2 Requirement") or row.get("SYS.1 Requirement") or "")
            row["TYPE"] = _norm_type(row.get("TYPE", ""), base_req_text)
            row["Domain"] = _norm_domain_sys2(row.get("Domain", ""), base_req_text)
            _method_for_level = str(it.get("Verification Method") or it.get("Verification") or it.get("Method") or "")
            row["Verification Level"] = _norm_level(_method_for_level, row.get("TYPE", ""), base_req_text)
            rows.append(row)
        import pandas as _pd
        sys2_table = _pd.DataFrame(rows, columns=sys2_cols) if rows else _pd.DataFrame(columns=sys2_cols)
        if sys2_table.empty:
            st.info("No structured SYS.2 data parsed; see raw JSON below.")
        else:
            st.write("### SYS.2 Requirements")
            
            # Display save notification if available
            if "sys2_save_notification" in st.session_state:
                notif = st.session_state["sys2_save_notification"]
                if "✅" in notif:
                    st.toast(notif, icon="✅")
                else:
                    st.toast(notif, icon="⚠️")
                del st.session_state["sys2_save_notification"]
            
            # Show input vs output coverage and capture parsed inputs for traceability
            parsed_in = None
            in_count = None
            try:
                _parsed = json.loads(req_text_used) if req_text_used.strip() else None
                if isinstance(_parsed, dict):
                    # Some inputs may wrap the list under a key
                    for k in ("SYS1", "SYS.1", "sys1", "sys_1", "inputs", "data"):
                        if k in _parsed and isinstance(_parsed[k], list):
                            parsed_in = _parsed[k]
                            break
                elif isinstance(_parsed, list):
                    parsed_in = _parsed
                if isinstance(parsed_in, list):
                    in_count = len(parsed_in)
            except Exception:
                parsed_in = None
                in_count = None
            if in_count is not None:
                st.caption(f"SYS.1 inputs: {in_count} | SYS.2 outputs: {len(sys2_table)}")
            else:
                st.caption(f"Showing {len(sys2_table)} SYS.2 requirements. Scroll if more appear.")
            # Editable columns for Agent 2
            editable_cols_sys2 = {}
            for c in ["SYS.2 Requirement", "TYPE", "Verification Level", "Verification Criteria", "Domain"]:
                if c not in sys2_table.columns:
                    continue
                if c == "Verification Level":
                    editable_cols_sys2[c] = st.column_config.SelectboxColumn(
                        c,
                        options=[
                            "System Integration Test (SYS.4)",
                            "System Qualification Test (SYS.5)",
                        ],
                        help="Choose verification level",
                        width="medium",
                    )
                elif c == "TYPE":
                    editable_cols_sys2[c] = st.column_config.SelectboxColumn(
                        c,
                        options=["Functional", "Non-Functional", "Information"],
                        help="Select requirement type",
                        width="medium",
                    )
                else:
                    editable_cols_sys2[c] = st.column_config.TextColumn(
                        c,
                        help=f"Editable: {c}",
                        width=("large" if c == "SYS.2 Requirement" else "medium"),
                    )
            column_config_sys2 = {}
            for c in sys2_table.columns:
                if c in editable_cols_sys2:
                    column_config_sys2[c] = editable_cols_sys2[c]
                else:
                    column_config_sys2[c] = st.column_config.TextColumn(c, disabled=True, width="medium")

            # Editor in an expander
            with st.expander("✏️ Edit Table (SYS.2)", expanded=False):
                edited_sys2 = st.data_editor(
                    sys2_table,
                    use_container_width=True,
                    num_rows="fixed",
                    column_config=column_config_sys2,
                    hide_index=True,
                    key="sys2_editor",
                    disabled=False
                )
            # Keep edits in session
            st.session_state["sys2_table_df"] = edited_sys2.copy()
            
            # Auto-save immediately after generation
            try:
                _p = os.path.join(BASE_DIR, "data", "outputs", "sys2_requirements.xlsx")
                _saved = save_excel_styled_to_path(edited_sys2, _p, "SYS2")
                if _saved:
                    st.session_state["sys2_last_saved"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state["sys2_save_notification"] = f"✅ Auto-saved to: {_p}"
                else:
                    st.session_state["sys2_save_notification"] = "⚠️ Auto-save returned False"
            except Exception as e:
                st.session_state["sys2_save_notification"] = f"⚠️ Auto-save failed: {e}"

            # Wrapped view for readability
            render_static_white_table(edited_sys2, max_height="95vh", table_id="sys2_table", no_limit=True)

            # Save button + status badge (placed ABOVE export options)
            c_btn2, c_badge2 = st.columns([1, 5])
            with c_btn2:
                save_sys2 = st.button("💾 Save Changes", key="save_sys2")
            if save_sys2:
                try:
                    # Use the latest edited data from session state
                    save_df = st.session_state.get("sys2_table_df", edited_sys2)
                    _p = os.path.join(BASE_DIR, "data", "outputs", "sys2_requirements.xlsx")
                    _saved = save_excel_styled_to_path(save_df, _p, "SYS2")
                    if _saved:
                        notify_saved(_saved, label="SYS.2")
                        st.session_state["sys2_last_saved"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.toast("✅ Changes saved!", icon="✅")
                    else:
                        st.toast("❌ Save operation returned False. Check file permissions.", icon="❌")
                except Exception as e:
                    st.toast(f"❌ Save failed: {str(e)}", icon="❌")
            with c_badge2:
                _ts2 = st.session_state.get("sys2_last_saved")
                if _ts2:
                    st.markdown(
                        f"<span style='display:inline-block;padding:2px 8px;border-radius:999px;background:#e6ffed;color:#065f46;font-size:12px;border:1px solid #34d399;'>Saved at {_ts2}</span>",
                        unsafe_allow_html=True,
                    )
            # Export options below save
            render_export_buttons(edited_sys2, base_name="SYS2_requirements", label="Export Options")
            st.caption("Note: TYPE is auto-inferred (Functional/Non-Functional/Information), Domain normalized to SW/HW/System/Mechanical, and Verification Level is prefilled (SYS.4 or SYS.5) based on method/context.")
            # Per-agent traceability (SYS.1 → SYS.2) right below the table
            st.write("---")
            render_sys2_traceability(sys2_table, sys1_input_list=parsed_in if isinstance(parsed_in, list) else None)

    # Display existing SYS.2 table if available and not currently running
    elif display_existing_sys2:
        existing_sys2 = st.session_state["sys2_table_df"].copy()
        st.write("### SYS.2 Requirements")
        st.caption(f"Showing {len(existing_sys2)} SYS.2 requirements. Edit the table below and click 'Save Changes'.")

        # Configure editable vs readonly columns like above
        editable_cols_sys2 = {}
        for c in ["SYS.2 Requirement", "TYPE", "Verification Level", "Verification Criteria", "Domain"]:
            if c not in existing_sys2.columns:
                continue
            if c == "Verification Level":
                editable_cols_sys2[c] = st.column_config.SelectboxColumn(
                    c,
                    options=[
                        "System Integration Test (SYS.4)",
                        "System Qualification Test (SYS.5)",
                    ],
                    help="Choose verification level",
                    width="medium",
                )
            elif c == "TYPE":
                editable_cols_sys2[c] = st.column_config.SelectboxColumn(
                    c,
                    options=["Functional", "Non-Functional", "Information"],
                    help="Select requirement type",
                    width="medium",
                )
            else:
                editable_cols_sys2[c] = st.column_config.TextColumn(
                    c,
                    help=f"Editable: {c}",
                    width=("large" if c == "SYS.2 Requirement" else "medium"),
                )
        column_config_sys2 = {}
        for c in existing_sys2.columns:
            if c in editable_cols_sys2:
                column_config_sys2[c] = editable_cols_sys2[c]
            else:
                column_config_sys2[c] = st.column_config.TextColumn(c, disabled=True, width="medium")

        with st.expander("✏️ Edit Table (SYS.2)", expanded=False):
            edited_sys2 = st.data_editor(
                existing_sys2,
                use_container_width=True,
                num_rows="fixed",
                column_config=column_config_sys2,
                hide_index=True,
                key="sys2_editor",
                disabled=False
            )
        st.session_state["sys2_table_df"] = edited_sys2.copy()
        render_static_white_table(edited_sys2, max_height="95vh", table_id="sys2_table", no_limit=True)

        # Save controls + badge (placed ABOVE export options)
        c_btn2, c_badge2 = st.columns([1, 5])
        with c_btn2:
            save_sys2 = st.button("💾 Save Changes", key="save_sys2")
        if save_sys2:
            try:
                # Use the latest edited data from session state
                save_df = st.session_state.get("sys2_table_df", edited_sys2)
                _p = os.path.join(BASE_DIR, "data", "outputs", "sys2_requirements.xlsx")
                _saved = save_excel_styled_to_path(save_df, _p, "SYS2")
                if _saved:
                    notify_saved(_saved, label="SYS.2")
                    st.session_state["sys2_last_saved"] = datetime.now().strftime("%Y-%m-%d %H:%M")
                    st.toast("✅ Changes saved!", icon="✅")
                else:
                    st.toast("❌ Save operation returned False. Check file permissions.", icon="❌")
            except Exception as e:
                st.toast(f"❌ Save failed: {str(e)}", icon="❌")
        with c_badge2:
            _ts2 = st.session_state.get("sys2_last_saved")
            if _ts2:
                st.markdown(
                    f"<span style='display:inline-block;padding:2px 8px;border-radius:999px;background:#e6ffed;color:#065f46;font-size:12px;border:1px solid #34d399;'>Saved at {_ts2}</span>",
                    unsafe_allow_html=True,
                )
        # Export options below save
        render_export_buttons(edited_sys2, base_name="SYS2_requirements", label="Export Options")
        st.write("---")
        # Traceability can work with existing table as well (no raw SYS.1 list available here)
        try:
            render_sys2_traceability(edited_sys2, sys1_input_list=None)
        except Exception:
            pass
    if st.button("⬅ Back to Home"):
        go_to("Home")


# Review page
elif st.session_state.page == "Review":
    st.header("⚠️ SYS.2 Review")
    st.caption("Review SYS.2 items for clarity, completeness, and standards compliance.")
    up_rev = st.file_uploader("Upload SYS.2 Requirements (.txt, .docx, .pdf, .xlsx, .reqif, .json)", type=["json","txt","docx","pdf","xlsx","reqif"], key="rev_upload")
    req_text = ""
    if up_rev is not None:
        try:
            _bytes = up_rev.read()
            _text, _recs = load_file_for_agent(_bytes, up_rev.name, "Review")
            raw = json.dumps(_recs) if _recs is not None else (_text or "")
        except Exception:
            raw = ""
        req_text = raw
        st.text_area("Preview (read-only)", raw[:6000], height=180, disabled=True)
        st.caption("Using uploaded file content for Review agent.")
    else:
        req_text = st.text_area("Or paste SYS.2 Requirements (JSON):", height=200, placeholder="Paste a JSON array of SYS.2 items or raw text that describes the SYS.2 requirements…")

    # RAG toggle
    st.write("🔎 **Enable RAG Context** (Standards-based)")
    use_rag_review = st.checkbox("Use standards to enhance requirements review", value=False, key="rag_review",
                               help="Augment prompts with relevant context from automotive standards (ISO 26262, ISO 29148, etc.)")
    
    st.session_state.setdefault("running_review", False)
    run_review_clicked = st.button("⚠️ Generate Review Feedback", disabled=st.session_state.get("running_review", False))
    if run_review_clicked and not st.session_state.get("running_review"):
        if not req_text.strip():
            st.warning("Please upload a .txt, .docx, .pdf, .xlsx, .json, or .reqif file, or paste requirements, before running.")
        else:
            st.session_state.running_review = True
            st.session_state.review_use_rag = use_rag_review
            st.rerun()
    if st.session_state.get("running_review"):
        _loader3 = st.empty()
        _loader3.markdown("<div class='loader-row'><div class='spinner-small'></div><div class='running-label'>Processing Review...</div></div>", unsafe_allow_html=True)
        result = run_single("Review", req_text, use_rag=st.session_state.get("review_use_rag", False))
        st.session_state.running_review = False
        _loader3.empty()
        st.success("Review Complete ✅")
        st.caption("Output available in Word, PDF, XLSX, REQIF formats.")

        # Desired review output format (all columns from Agent 3 schema)
        desired_cols = [
            "SYS.2 Req ID",
            "SYS.2 Requirement",
            "Review Feedback",
            "SMART Check",
            "SMART Fix Suggestion",
            "Proposed Rewrite",
            "Compliance Check",
            "Severity",
            "Suggested Improvement",
        ]
        aliases = {
            "SYS.2 Req ID": ["SYS.2 Req ID", "SYS.2 Req. ID", "SYS2 Req ID"],
            "SYS.2 Requirement": ["SYS.2 Requirement", "SYS.2 Req", "SYS2 Requirement"],
            "Review Feedback": ["Review Feedback", "Feedback", "Review"],
            "SMART Check": ["SMART Check", "Smart Check", "SMART"],
            "SMART Fix Suggestion": ["SMART Fix Suggestion", "Smart Fix Suggestion", "Fix Suggestion"],
            "Proposed Rewrite": ["Proposed Rewrite", "Rewrite", "Rewritten"],
            "Compliance Check": ["Compliance Check", "Compliance", "Check"],
            "Severity": ["Severity", "Priority"],
            "Suggested Improvement": ["Suggested Improvement", "Improvement", "Suggestion"],
        }
        raw_items = []
        if isinstance(result, dict):
            candidate = result.get("Review") or result.get("SYS2 Review") or []
            if isinstance(candidate, list):
                raw_items = candidate
        rows = []
        for it in raw_items:
            if not isinstance(it, dict):
                continue
            row = {}
            # Helper: turn list/dict into readable sentences
            def _fmt_sentences(value):
                try:
                    if isinstance(value, dict):
                        parts = [f"{str(k).capitalize()}: {v}" for k, v in value.items() if v not in (None, "")]
                        s = "; ".join(parts)
                        return s + "." if s and not s.endswith(".") else s
                    if isinstance(value, list):
                        lines = []
                        for x in value:
                            sx = str(x).strip() if not isinstance(x, (dict, list)) else _fmt_sentences(x)
                            if sx:
                                lines.append((sx + ".") if not sx.endswith(".") else sx)
                        return "\n".join(lines)
                    if isinstance(value, str):
                        s = value.strip()
                        return s + "." if s and not s.endswith(".") else s
                    return str(value)
                except Exception:
                    return str(value)
            for col, alist in aliases.items():
                val = ""
                for k in alist:
                    if k in it and it[k] not in (None, ""):
                        val = it[k]
                        break
                # Format lists/dicts in review fields as readable statements
                row[col] = _fmt_sentences(val)
            rows.append(row)
        import pandas as _pd
        review_table = _pd.DataFrame(rows, columns=desired_cols) if rows else _pd.DataFrame(columns=desired_cols)
        if review_table.empty:
            st.info("No structured review feedback parsed; see raw JSON below.")
        else:
            st.markdown(
                "<h3 style='margin:8px 0 8px; font-weight:800; color:#0f3355;'>⚠️ SYS.2 Review Feedback</h3>",
                unsafe_allow_html=True,
            )
            
            # Display save notification if available
            if "review_save_notification" in st.session_state:
                notif = st.session_state["review_save_notification"]
                if "✅" in notif:
                    st.toast(notif, icon="✅")
                else:
                    st.toast(notif, icon="⚠️")
                del st.session_state["review_save_notification"]
            
            st.caption(f"Showing {len(review_table)} review items. Scroll if more appear.")
            render_static_white_table(review_table, max_height="95vh", table_id="review_table", no_limit=True)
            render_export_buttons(review_table, base_name="SYS2_review", label="Export Options")
            st.session_state["review_table_df"] = review_table
            # Auto-save to disk (verified)
            try:
                _p = os.path.join(BASE_DIR, "data", "outputs", "sys2_requirements_reviewed.xlsx")
                _saved = save_excel_styled_to_path(review_table, _p, "Review")
                if _saved:
                    st.session_state["review_last_saved"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state["review_save_notification"] = f"✅ Auto-saved to: {_p}"
                else:
                    st.session_state["review_save_notification"] = "⚠️ Auto-save returned False"
            except Exception as e:
                st.session_state["review_save_notification"] = f"⚠️ Auto-save failed: {e}"
            if st.session_state.get("show_traceability"):
                st.write("---")
                render_traceability_section()
    if st.button("⬅ Back to Home"):
        go_to("Home")


# SYS.5 page
elif st.session_state.page == "SYS5":
    st.header("🧪 SYS.5 Testcase Generation")
    st.caption("Generate structured SYS.5 test cases from reviewed SYS.2 requirements.")
    up_sys5 = st.file_uploader("Upload SYS.2 Requirements (.txt, .docx, .pdf, .xlsx, .reqif, .json)", type=["json","txt","docx","pdf","xlsx","reqif"], key="sys5_upload")
    req_text = ""
    if up_sys5 is not None:
        try:
            _bytes = up_sys5.read()
            _text, _recs = load_file_for_agent(_bytes, up_sys5.name, "SYS5")
            raw = json.dumps(_recs) if _recs is not None else (_text or "")
        except Exception:
            raw = ""
        req_text = raw
        st.text_area("Preview (read-only)", raw[:6000], height=180, disabled=True)
        st.caption("Using uploaded file content for SYS.5 agent.")
    else:
        req_text = st.text_area("Or paste SYS.2 Requirements (JSON):", height=200, placeholder="Paste a JSON array of SYS.2 items (preferably reviewed) or raw text…")

    # RAG toggle
    st.write("🔎 **Enable RAG Context** (Standards-based)")
    use_rag_sys5 = st.checkbox("Use standards to enhance test case generation", value=False, key="rag_sys5",
                               help="Augment prompts with relevant context from automotive standards (ISO 26262, ASPICE, etc.)")

    # Removed input-shape validation and 'Use last SYS.2 output' helper per request
    st.session_state.setdefault("running_sys5", False)
    run_sys5_clicked = st.button("🧪 Generate Test Cases", disabled=st.session_state.get("running_sys5", False))
    if run_sys5_clicked and not st.session_state.get("running_sys5"):
        if not req_text.strip():
            st.warning("Please upload a .txt, .docx, .pdf, .xlsx, .json, or .reqif file, or paste requirements, before running.")
        else:
            st.session_state.running_sys5 = True
            st.session_state.sys5_use_rag = use_rag_sys5
            st.rerun()
    if st.session_state.get("running_sys5"):
        _loader4 = st.empty()
        _loader4.markdown("<div class='loader-row'><div class='spinner-small'></div><div class='running-label'>Processing SYS.5...</div></div>", unsafe_allow_html=True)
        req_text_used = req_text
        # Use override if present to ensure correct SYS.2 JSON is provided
        if st.session_state.get("sys5_input_override"):
            req_text = st.session_state.pop("sys5_input_override")
            req_text_used = req_text
        result = run_single("SYS5", req_text, use_rag=st.session_state.get("sys5_use_rag", False))
        st.session_state.running_sys5 = False
        _loader4.empty()
        st.success("Test Cases Generated ✅")
        st.caption("Output available in Word, PDF, XLSX, REQIF formats.")

        # Desired output columns & mapping (allowing for key variants)
        desired_cols = [
            "SYS.2 Req ID",
            "SYS.2 Requirement",
            "Test Case ID",
            "Description",
            "Preconditions",
            "Test Steps",
            "Expected Result",
            "Pass/Fail Criteria",
        ]
        key_aliases = {
            "SYS.2 Req ID": ["SYS.2 Req. ID", "SYS2 Req ID", "SYS_2 Req ID"],
            "SYS.2 Requirement": ["SYS.2 Requirement", "SYS2 Requirement"],
            "Test Case ID": ["Test Case ID", "TC ID", "Test_ID"],
            "Description": ["Description", "Test Description"],
            "Preconditions": ["Preconditions", "Precondition"],
            "Test Steps": ["Test Steps", "Steps", "Test_Steps"],
            "Expected Result": ["Expected Result", "Expected", "Expected_Result"],
            "Pass/Fail Criteria": ["Pass/Fail Criteria", "Pass Fail Criteria", "PassFail Criteria"],
        }

        # Extract list of test case dicts from result
        raw_list = []
        if isinstance(result, dict):
            candidate = result.get("SYS5") or result.get("SYS.5") or []
            if isinstance(candidate, list):
                raw_list = candidate
        # Normalize into rows
        rows = []
        for item in raw_list:
            if not isinstance(item, dict):
                continue
            row = {}
            # Helper mirroring SYS.2 criteria sentence formatting (without method)
            def _fmt_criteria(val):
                def _with_period(txt: str) -> str:
                    return txt if txt.endswith(".") else txt + "."
                def _from_mapping(d):
                    keys = {str(k).lower(): v for k, v in d.items() if v not in (None, "")}
                    metric = keys.get("metric") or keys.get("parameter") or keys.get("criterion") or keys.get("measure")
                    threshold = keys.get("threshold") or keys.get("limit")
                    rng = keys.get("range")
                    target = keys.get("target") or keys.get("value")
                    cond = keys.get("condition") or keys.get("conditions")
                    if metric and threshold:
                        base = f"Verify {str(metric).lower()} is {threshold}"
                        if cond: base += f" under {cond}"
                        return _with_period(base)
                    if metric and rng:
                        base = f"Verify {str(metric).lower()} is within {rng}"
                        if cond: base += f" under {cond}"
                        return _with_period(base)
                    if metric and target:
                        base = f"Verify {str(metric).lower()} meets {target}"
                        if cond: base += f" under {cond}"
                        return _with_period(base)
                    flat = "; ".join(f"{k.capitalize()}: {v}" for k, v in d.items() if v not in (None, ""))
                    return _with_period(f"Verify criteria: {flat}") if flat else ""
                if isinstance(val, dict):
                    return _from_mapping(val)
                if isinstance(val, list):
                    parts = []
                    for item in val:
                        if isinstance(item, (dict, list)):
                            parts.append(_fmt_criteria(item))
                        else:
                            s = str(item).strip()
                            if s: parts.append(_with_period(s))
                    return "\n".join([p for p in parts if p])
                if isinstance(val, str):
                    s = val.strip()
                    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
                        try:
                            parsed = json.loads(s.replace("'", '"'))
                            return _fmt_criteria(parsed)
                        except Exception:
                            try:
                                import ast
                                parsed = ast.literal_eval(s)
                                return _fmt_criteria(parsed)
                            except Exception:
                                pass
                    try:
                        import re as _re
                        m = _re.search(r"[Mm]etric[\s'\":]*([^,;\n]+)", s)
                        t = _re.search(r"[Tt]hreshold[\s'\":]*([^,;\n]+)", s)
                        if m and t:
                            metric = m.group(1).strip(" '\"")
                            thr = t.group(1).strip(" '\"")
                            return _with_period(f"Verify {metric.lower()} is {thr}")
                    except Exception:
                        pass
                    return _with_period(s) if s else ""
                return _with_period(str(val)) if val is not None else ""
            for target_col, aliases in key_aliases.items():
                value = ""
                for k in aliases:
                    if k in item and item[k] not in (None, ""):
                        value = item[k]
                        break
                # Join list steps or multiline content gracefully
                if isinstance(value, list):
                    # For Test Steps specifically produce numbered multiline list
                    if target_col == "Test Steps":
                        # Remove auto-numbering; items often already include 'Step 1:' etc
                        value = "\n".join(str(x).strip() for x in value if str(x).strip())
                    else:
                        value = " | ".join(str(x) for x in value)
                # Format pass/fail criteria into sentences
                if target_col == "Pass/Fail Criteria":
                    value = _fmt_criteria(value)
                row[target_col] = value
            rows.append(row)
        import pandas as _pd
        sys5_table = _pd.DataFrame(rows, columns=desired_cols) if rows else _pd.DataFrame(columns=desired_cols)

        if sys5_table.empty:
            st.info("No structured test cases parsed; showing raw output below.")
        else:
            st.write("### SYS.5 Test Cases")
            
            # Display save notification if available
            if "sys5_save_notification" in st.session_state:
                notif = st.session_state["sys5_save_notification"]
                if "✅" in notif:
                    st.toast(notif, icon="✅")
                else:
                    st.toast(notif, icon="⚠️")
                del st.session_state["sys5_save_notification"]
            
            st.caption(f"Showing {len(sys5_table)} test case(s). Scroll vertically to see more if present.")
            # Editable columns for Agent 4 (SYS.5)
            editable_cols_sys5 = {c: st.column_config.TextColumn(c, help=f"Editable: {c}", width=("large" if c in ["Description", "Expected Result", "Pass/Fail Criteria"] else "medium")) for c in [
                "Description", "Preconditions", "Test Steps", "Expected Result", "Pass/Fail Criteria"
            ] if c in sys5_table.columns}
            column_config_sys5 = {}
            for c in sys5_table.columns:
                if c in editable_cols_sys5:
                    column_config_sys5[c] = editable_cols_sys5[c]
                else:
                    column_config_sys5[c] = st.column_config.TextColumn(c, disabled=True, width="medium")
            with st.expander("✏️ Edit Table (SYS.5)", expanded=False):
                edited_sys5 = st.data_editor(
                    sys5_table,
                    use_container_width=True,
                    num_rows="fixed",
                    column_config=column_config_sys5,
                    hide_index=True,
                    key="sys5_editor",
                    disabled=False
                )
            # Keep edits in session
            st.session_state["sys5_table_df"] = edited_sys5.copy()
            
            # Auto-save immediately after generation
            try:
                _p = os.path.join(BASE_DIR, "data", "outputs", "sys.5_test_cases.xlsx")
                _saved = save_excel_styled_to_path(edited_sys5, _p, "SYS5")
                if _saved:
                    st.session_state["sys5_last_saved"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state["sys5_save_notification"] = f"✅ Auto-saved to: {_p}"
                else:
                    st.session_state["sys5_save_notification"] = "⚠️ Auto-save returned False"
            except Exception as e:
                st.session_state["sys5_save_notification"] = f"⚠️ Auto-save failed: {e}"
            
            render_static_white_table(edited_sys5, max_height="none", table_id="sys5_table", no_limit=True)
            render_export_buttons(edited_sys5, base_name="SYS5_testcases", label="Export Options")
            # Save button + status badge
            c_btn5, c_badge5 = st.columns([1, 5])
            with c_btn5:
                save_sys5 = st.button("💾 Save Changes", key="save_sys5")
            if save_sys5:
                try:
                    # Use the latest edited data from session state
                    save_df = st.session_state.get("sys5_table_df", edited_sys5)
                    _p = os.path.join(BASE_DIR, "data", "outputs", "sys.5_test_cases.xlsx")
                    _saved = save_excel_styled_to_path(save_df, _p, "SYS5")
                    if _saved:
                        notify_saved(_saved, label="SYS.5")
                        st.session_state["sys5_last_saved"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.toast("✅ Changes saved!", icon="✅")
                    else:
                        st.toast("❌ Save operation returned False. Check file permissions.", icon="❌")
                except Exception as e:
                    st.toast(f"❌ Save failed: {str(e)}", icon="❌")
            with c_badge5:
                _ts5 = st.session_state.get("sys5_last_saved")
                if _ts5:
                    st.markdown(
                        f"<span style='display:inline-block;padding:2px 8px;border-radius:999px;background:#e6ffed;color:#065f46;font-size:12px;border:1px solid #34d399;'>Saved at {_ts5}</span>",
                        unsafe_allow_html=True,
                    )

            # Parse input SYS.2 list for downstream traceability only (no debug UI)
            input_sys2 = None
            try:
                parsed_in = json.loads(req_text_used) if req_text_used.strip() else None
                if isinstance(parsed_in, dict):
                    for k in ("SYS2", "SYS.2", "sys2", "sys_2"):
                        if k in parsed_in and isinstance(parsed_in[k], list):
                            parsed_in = parsed_in[k]
                            break
                if isinstance(parsed_in, list):
                    input_sys2 = parsed_in
            except Exception:
                input_sys2 = None
        # Show per-agent traceability (SYS.2 → SYS.5) below the table
        try:
            _inputs_sys2 = input_sys2 if isinstance(input_sys2, list) else None
        except Exception:
            _inputs_sys2 = None
        st.write("---")
        render_sys5_traceability(sys5_table, sys2_input_list=_inputs_sys2)

    # Display existing SYS.5 table if available and not currently running (persistence after edits)
    elif "sys5_table_df" in st.session_state and not st.session_state.get("running_sys5", False):
        existing_sys5 = st.session_state["sys5_table_df"].copy()
        st.write("### SYS.5 Test Cases")
        st.caption(f"Showing {len(existing_sys5)} test case(s). Edit the table below and click 'Save Changes'.")
        # Configure editable columns same as generation path
        editable_cols_sys5 = {c: st.column_config.TextColumn(c, help=f"Editable: {c}", width=("large" if c in ["Description", "Expected Result", "Pass/Fail Criteria"] else "medium")) for c in [
            "Description", "Preconditions", "Test Steps", "Expected Result", "Pass/Fail Criteria"
        ] if c in existing_sys5.columns}
        column_config_sys5 = {}
        for c in existing_sys5.columns:
            if c in editable_cols_sys5:
                column_config_sys5[c] = editable_cols_sys5[c]
            else:
                column_config_sys5[c] = st.column_config.TextColumn(c, disabled=True, width="medium")
        with st.expander("✏️ Edit Table (SYS.5)", expanded=False):
            edited_sys5 = st.data_editor(
                existing_sys5,
                use_container_width=True,
                num_rows="fixed",
                column_config=column_config_sys5,
                hide_index=True,
                key="sys5_editor",
                disabled=False
            )
        st.session_state["sys5_table_df"] = edited_sys5.copy()
        render_static_white_table(edited_sys5, max_height="none", table_id="sys5_table", no_limit=True)
        # Save button above export options
        c_btn5, c_badge5 = st.columns([1,5])
        with c_btn5:
            save_sys5 = st.button("💾 Save Changes", key="save_sys5")
        if save_sys5:
            try:
                # Use the latest edited data from session state
                save_df = st.session_state.get("sys5_table_df", edited_sys5)
                _p = os.path.join(BASE_DIR, "data", "outputs", "sys.5_test_cases.xlsx")
                _saved = save_excel_styled_to_path(save_df, _p, "SYS5")
                if _saved:
                    notify_saved(_saved, label="SYS.5")
                    st.session_state["sys5_last_saved"] = datetime.now().strftime("%Y-%m-%d %H:%M")
                    st.toast("✅ Changes saved!", icon="✅")
                else:
                    st.toast("❌ Save operation returned False. Check file permissions.", icon="❌")
            except Exception as e:
                st.toast(f"❌ Save failed: {str(e)}", icon="❌")
        with c_badge5:
            _ts5 = st.session_state.get("sys5_last_saved")
            if _ts5:
                st.markdown(
                    f"<span style='display:inline-block;padding:2px 8px;border-radius:999px;background:#e6ffed;color:#065f46;font-size:12px;border:1px solid #34d399;'>Saved at {_ts5}</span>",
                    unsafe_allow_html=True,
                )
        render_export_buttons(edited_sys5, base_name="SYS5_testcases", label="Export Options")
        # Traceability without original input list
        st.write("---")
        try:
            render_sys5_traceability(edited_sys5, sys2_input_list=None)
        except Exception:
            pass

    if st.button("⬅ Back to Home"):
        go_to("Home")


# Manager page - full pipeline and consolidated RTM
elif st.session_state.page == "Manager":
    st.header("🗂️ Manager Orchestration")
    st.markdown("""
        <div class='neon-flow'>
            Run end-to-end workflow:
            <span class='step'>Customer Req</span>
            <span class='neon-arrow'>➜</span>
            <span class='step'>SYS.1</span>
            <span class='neon-arrow alt'>➜</span>
            <span class='step'>SYS.2</span>
            <span class='neon-arrow'>➜</span>
            <span class='step'>Review</span>
            <span class='neon-arrow alt'>➜</span>
            <span class='step'>SYS.5</span>
        </div>
        """, unsafe_allow_html=True)
    up_mgr = st.file_uploader("Upload Customer Requirements (.txt, .docx, .pdf, .xlsx, .reqif, .json)", type=["txt","docx","pdf","xlsx","reqif","json"], key="mgr_upload")
    cust_req: str = ""
    if up_mgr is not None:
        try:
            _bytes = up_mgr.read()
            _text, _ = load_file_for_agent(_bytes, up_mgr.name, "SYS1")
            cust_req = _text or ""
        except Exception:
            cust_req = ""
        st.text_area("Preview (read-only)", cust_req[:5000], height=200, disabled=True)
        st.caption("Using uploaded file content for full pipeline.")
    else:
        cust_req = st.text_area("Or paste Customer Requirements:", height=250, placeholder="Paste high-level customer requirements… (e.g., ‘The system shall maintain speed at a setpoint…’) ")
    
    # RAG toggle for full pipeline
    st.write("🔎 **Enable RAG Context for All Agents** (Standards-based)")
    use_rag_pipeline = st.checkbox("Use standards to enhance the entire pipeline", value=False, key="rag_pipeline",
                                   help="Augment prompts with relevant context from automotive standards across the entire pipeline")
    
    st.session_state.setdefault("running_pipeline", False)
    run_pipeline_clicked = st.button("🗂️ Run Full Pipeline", disabled=st.session_state.get("running_pipeline", False))
    if run_pipeline_clicked and not st.session_state.get("running_pipeline"):
        if not cust_req.strip():
            st.warning("Please upload a .txt, .docx, .pdf, .xlsx, .json, or .reqif file, or paste requirements, before running the pipeline.")
        else:
            st.session_state.running_pipeline = True
            st.session_state.pipeline_use_rag = use_rag_pipeline
            st.rerun()
    if st.session_state.get("running_pipeline"):
        _loader5 = st.empty()
        _loader5.markdown("<div class='loader-row'><div class='spinner-small'></div><div class='running-label'>Running Full Pipeline...</div></div>", unsafe_allow_html=True)
        outputs = run_pipeline(cust_req, use_rag=st.session_state.get("pipeline_use_rag", False))
        st.session_state.running_pipeline = False
        _loader5.empty()
        # Parse outputs
        cust_sys1_df = parse_agent_output(outputs.get('SYS1', []))
        sys1_sys2_df = parse_agent_output(outputs.get('SYS2', []))
        review_df    = parse_agent_output(outputs.get('Review', []))
        sys2_sys5_df = parse_agent_output(outputs.get('SYS5', []))

        st.success("Pipeline Complete ✅")
        st.caption("Output available in Word, PDF, XLSX, REQIF, and bundled ZIP from the sidebar.")

        st.write("---")
        st.markdown("## 📊 Consolidated Traceability Dashboard")
        st.write("")  # Add spacing
        
        # Metrics in a prominent card-style layout
        col1, col2, col3, col4 = st.columns(4)
        with col1: 
            st.metric("Customer Reqs", len(cust_sys1_df))
        with col2: 
            st.metric("SYS.1 Reqs", len(sys1_sys2_df))
        with col3: 
            st.metric("SYS.2 Reviewed", len(review_df))
        with col4: 
            st.metric("Test Cases", len(sys2_sys5_df))
        
        st.write("")  # Add spacing after metrics
        st.write("---")
        st.subheader("🔎 Customer Req -> SYS.1")
        responsive_dataframe(cust_sys1_df, hide_index=True)
        # Make SYS.1 mapping available to traceability visual
        try:
            if isinstance(cust_sys1_df, pd.DataFrame) and not cust_sys1_df.empty:
                st.session_state["sys1_table_df"] = cust_sys1_df
        except Exception:
            pass

        st.subheader("📝 SYS.1 -> SYS.2 (with Review)")
        # Be resilient to slight column name variations
        merged_sys2 = sys1_sys2_df.copy()
        if not sys1_sys2_df.empty and not review_df.empty:
            left_key = None
            for k in ["SYS.2 Req. ID", "SYS.2 Req ID", "SYS2 Req ID"]:
                if k in sys1_sys2_df.columns:
                    left_key = k
                    break
            right_key = None
            for k in ["SYS.2 Req. ID", "SYS.2 Req ID", "SYS2 Req ID"]:
                if k in review_df.columns:
                    right_key = k
                    break
            if left_key and right_key:
                merged_sys2 = sys1_sys2_df.merge(review_df, left_on=left_key, right_on=right_key, how='left')
            else:
                merged_sys2 = sys1_sys2_df
        responsive_dataframe(merged_sys2, hide_index=True)

        # Manager checks: approval gating and conflict detection
        with st.expander("Manager Checks: Approval & Conflicts", expanded=False):
            # Add CSS for center-aligned table
            st.markdown("""
                <style>
                    [data-testid="stDataFrame"] td, [data-testid="stDataFrame"] th {
                        text-align: center !important;
                    }
                </style>
            """, unsafe_allow_html=True)
            
            approved_ids = set()
            # Output approval summary and gating
            if not review_df.empty:
                st.markdown("#### Output Approval")
                comp_col = next((c for c in review_df.columns if c.lower().strip() == "compliance check"), None)
                sys2_id_in_review = next((c for c in review_df.columns if c.lower().startswith("sys.2 req")), None)
                if comp_col and sys2_id_in_review:
                    counts = review_df[comp_col].astype(str).str.strip().str.title().value_counts()
                    # Display as table instead of JSON
                    approval_summary_df = pd.DataFrame({
                        'Approval Status': counts.index,
                        'Count': counts.values
                    })
                    st.dataframe(
                        approval_summary_df, 
                        use_container_width=True, 
                        hide_index=True,
                        column_config={
                            'Approval Status': st.column_config.TextColumn('Approval Status', width="medium"),
                            'Count': st.column_config.NumberColumn('Count', width="medium")
                        }
                    )
                    
                    approved_ids = set(
                        review_df.loc[
                            review_df[comp_col].astype(str).str.strip().str.lower() == "yes",
                            sys2_id_in_review,
                        ].astype(str)
                    )
                    st.checkbox(
                        "Filter pipeline views and RTM to Approved-only (Compliance = Yes)",
                        value=False,
                        key="mgr_only_approved",
                    )
                else:
                    st.info("Review data present but required columns not found ('Compliance Check' and 'SYS.2 Req.*').")
            else:
                st.info("No Review data available yet for approval gating.")

            # Conflict detection for SYS.2 duplicate IDs with differing text
            st.markdown("#### Conflict Detection (SYS.2)")
            if not sys1_sys2_df.empty:
                sys2_id_col = next((c for c in sys1_sys2_df.columns if c.lower().startswith("sys.2 req")), None)
                sys2_txt_col = next((c for c in sys1_sys2_df.columns if c.lower().strip() == "sys.2 requirement"), None)
                if sys2_id_col and sys2_txt_col:
                    agg = sys1_sys2_df.groupby(sys2_id_col).agg({sys2_txt_col: lambda s: len(set(map(str, s)))})
                    conflicts = agg[agg[sys2_txt_col] > 1].reset_index()
                    if conflicts.empty:
                        st.caption("No SYS.2 conflicts detected (duplicate IDs with differing text).")
                    else:
                        st.warning(f"Detected {len(conflicts)} conflicting SYS.2 IDs. Review the variations below.")
                        conflict_ids = set(conflicts[sys2_id_col].astype(str))
                        conflict_rows = sys1_sys2_df[sys1_sys2_df[sys2_id_col].astype(str).isin(conflict_ids)]
                        responsive_dataframe(conflict_rows, hide_index=True)
                else:
                    st.info("SYS.2 columns not found to run conflict detection.")

        # Apply Approved-only filter if selected
        final_sys1_sys2_df = sys1_sys2_df
        final_review_df = review_df
        final_sys2_sys5_df = sys2_sys5_df
        try:
            if st.session_state.get("mgr_only_approved") and 'approved_ids' in locals() and approved_ids:
                sys2_id_col2 = next((c for c in sys1_sys2_df.columns if c.lower().startswith("sys.2 req")), None)
                if sys2_id_col2:
                    final_sys1_sys2_df = sys1_sys2_df[sys1_sys2_df[sys2_id_col2].astype(str).isin(approved_ids)].copy()
                sys2_id_col5 = next((c for c in sys2_sys5_df.columns if c.lower().startswith("sys.2 req")), None)
                if sys2_id_col5:
                    final_sys2_sys5_df = sys2_sys5_df[sys2_sys5_df[sys2_id_col5].astype(str).isin(approved_ids)].copy()
        except Exception:
            pass

        st.subheader("🧪 SYS.2 -> SYS.5 Test Cases")
        responsive_dataframe(final_sys2_sys5_df, hide_index=True)

        # SMART compliance (simple visualization)
        st.subheader("📊 SMART Compliance Overview")
        try:
            import matplotlib.pyplot as plt
            if not review_df.empty and 'SMART Check' in review_df.columns:
                smart_data = []
                for _, row in review_df.iterrows():
                    sc = row.get('SMART Check', {})
                    if isinstance(sc, dict):
                        for k,v in sc.items():
                            smart_data.append({'Criterion': k, 'Status': v})
                smart_df = pd.DataFrame(smart_data)
                if not smart_df.empty:
                    compliance_counts = smart_df['Status'].value_counts()
                    fig, ax = plt.subplots()
                    labels = [str(x) for x in list(compliance_counts.index)]
                    ax.pie(compliance_counts, labels=labels, autopct='%1.1f%%')
                    st.pyplot(fig)
                    st.subheader('Per-criterion compliance')
                    responsive_dataframe(smart_df.groupby(['Criterion','Status']).size().unstack(fill_value=0).reset_index(), hide_index=True)
            else:
                st.info('No SMART review data available yet.')
        except Exception as e:
            if plt is None:
                st.info('Install matplotlib to see charts.')
            else:
                st.warning(f'Could not render SMART compliance chart: {str(e)}')

        # Build RTM and exports
        rtm_df = build_traceability_matrix(cust_sys1_df, final_sys1_sys2_df, final_review_df, final_sys2_sys5_df)
        # Auto-save pipeline outputs to disk (verified) - save under agent5 folder
        try:
            agent5_dir = os.path.join(BASE_DIR, "data","outputs","agent5")
            p1 = save_excel_styled_to_path(cust_sys1_df, os.path.join(agent5_dir, "sys1_requirements.xlsx"), "SYS1")
            p2 = save_excel_styled_to_path(final_sys1_sys2_df, os.path.join(agent5_dir, "sys2_requirements.xlsx"), "SYS2")
            p3 = save_excel_styled_to_path(review_df, os.path.join(agent5_dir, "sys2_requirements_reviewed.xlsx"), "Review")
            p4 = save_excel_styled_to_path(final_sys2_sys5_df, os.path.join(agent5_dir, "sys.5_test_cases.xlsx"), "SYS5")
            if rtm_df is not None and not rtm_df.empty:
                p5 = save_excel_styled_to_path(rtm_df, os.path.join(agent5_dir, "rtm.xlsx"), "RTM")
            
            # Display success notifications
            saved_files = []
            if p1: saved_files.append(f"SYS.1: {p1}")
            if p2: saved_files.append(f"SYS.2: {p2}")
            if p3: saved_files.append(f"Review: {p3}")
            if p4: saved_files.append(f"SYS.5: {p4}")
            if 'p5' in locals() and p5: saved_files.append(f"RTM: {p5}")
            
            if saved_files:
                st.toast("✅ Pipeline outputs auto-saved successfully!", icon="✅")
                with st.expander("📂 View saved file locations"):
                    for file_info in saved_files:
                        st.write(f"• {file_info}")
        except Exception as e:
            st.toast(f"⚠️ Auto-save failed: {e}", icon="⚠️")
        st.write('---')
        st.subheader('📑 Requirements Traceability Matrix (RTM)')
        if not rtm_df.empty:
            responsive_dataframe(rtm_df, hide_index=True)

            # Export buttons individual files
            st.sidebar.header('📤 Export Manager Results')
            st.sidebar.download_button('Cust->SYS.1 CSV', export_csv(cust_sys1_df), 'cust_sys1.csv')
            st.sidebar.download_button('SYS.1->SYS.2 CSV', export_csv(merged_sys2 if not st.session_state.get('mgr_only_approved') else final_sys1_sys2_df), 'sys1_sys2.csv')
            st.sidebar.download_button('SYS.2->SYS.5 CSV', export_csv(final_sys2_sys5_df), 'sys2_sys5.csv')

            # Styled RTM exports
            st.sidebar.download_button('RTM (Excel Styled)', export_excel_styled(rtm_df), 'RTM_styled.xlsx')
            st.sidebar.download_button('RTM (Word Styled)', export_word_styled(rtm_df, project=project_name, version=version, author=author, logo_path=None), 'RTM.docx')
            st.sidebar.download_button('RTM (PDF Styled)', export_pdf_styled(rtm_df, project=project_name, version=version, author=author, logo_path=None), 'RTM.pdf')
            st.sidebar.download_button('RTM (REQIF)', export_reqif(rtm_df, spec_name=f"{project_name} Traceability Matrix"), 'RTM.reqif', mime='application/xml')

            # ZIP all
            tables = { 'Cust_SYS1': cust_sys1_df, 'SYS1_SYS2_withReview': (merged_sys2 if not st.session_state.get('mgr_only_approved') else final_sys1_sys2_df), 'SYS2_SYS5': final_sys2_sys5_df, 'RTM': rtm_df }
            # Optional: Approved-only RTM quick export when gate is active
            try:
                if st.session_state.get('mgr_only_approved'):
                    # detect SYS.2 Req column in RTM
                    sys2_in_rtm = next((c for c in rtm_df.columns if c.lower().startswith('sys.2 req')), None)
                    if sys2_in_rtm and 'approved_ids' in locals() and approved_ids:
                        rtm_approved = rtm_df[rtm_df[sys2_in_rtm].astype(str).isin(approved_ids)].copy()
                        st.sidebar.download_button('RTM Approved-only (Excel)', export_excel_styled(rtm_approved), 'RTM_approved.xlsx')
            except Exception:
                pass
            st.sidebar.download_button('📦 Export All (ZIP)', export_all_as_zip(tables), 'WHALE_traceability.zip', mime='application/zip')
        else:
            st.info('No RTM available yet. Run the pipeline with customer requirements.')

    if st.button('⬅ Back to Home'):
        go_to('Home')

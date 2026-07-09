# phishing_streamlit_ui_noLLM.py
# Deterministic Rendering: Bypasses LLM natural language summarization entirely to display the backend's raw tool outputs exactly as they are returned.
# Dynamic Data Translation: Automatically parses varying backend data structures directly into native UI components.
# Streamlined Interaction: Delegates 100% of tool selection and argument generation to llm_server.py via HTTP API.

import logging
import os
import json
import time
import html
import traceback
import requests
from datetime import datetime
from typing import Any, Dict, List, Optional
import streamlit as st

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("phishing_streamlit_ui_noLLM")

# Environment & Connection Setup
EXECUTION_MODE = os.getenv("MCP_EXECUTION_MODE", "direct").strip().lower()
MCP_SERVER_COMMAND = os.getenv("MCP_SERVER_COMMAND", "python")
MCP_SERVER_SCRIPT = os.getenv("MCP_SERVER_SCRIPT", "phishing_mcp_server.py")
DEFAULT_USER_ROLE = os.getenv("DEFAULT_USER_ROLE", "user")
REQUEST_TIMEOUT = int(os.getenv("UI_REQUEST_TIMEOUT", "300"))

# API endpoint for the LLM Router
LLM_SERVER_URL = os.getenv("LLM_SERVER_URL", "http://127.0.0.1:8001").rstrip("/")

APP_TITLE = "Phishing Simulation Analytics Copilot - No LLM"

st.set_page_config(page_title=APP_TITLE, page_icon="🎣", layout="wide", initial_sidebar_state="expanded")

TOOL_LABELS = {
    "run_analytics": "Historical analytics",
    "employee_lookup": "Employee lookup",
    "predict_risk": "ML prediction",
    "recommend_actions": "Recommendations",
    "simulation_users": "Simulation users",
    "cache_control": "Cache control",
    "cache_status": "Cache status",
    "system_info": "System info"
}

EXAMPLE_QUESTIONS = [
    "percentage of people clicked on January phishing campaign",
    "city with most clickers",
    "department wise click rate in 2026",
    "top risky employees",
    "show profile for BRID 75b4ae75-b",
    "predict no action probability for BRID 75b4ae75-b",
    "predict for city Pune department cyber subject password reset",
    "predicted high risk users",
    "how can BRID 75b4ae75-b improve",
    "who clicked in March 2026 simulation",
    "cache status",
    "refresh cache",
    "health check",
    "show table schema"
]

st.markdown("""
<style>
/* Ultra-Dark Obsidian Theme */
:root {
    --bg-main: #010204;
    --bg-sidebar: #000000;
    --bg-card: #04060a;
    --bg-card-2: #05080e;
    --border: #0a111a;
    --border-soft: #070b12;
    --text: #a5edf7;
    --muted: #5b708b;
    --accent: #2dd4bf;
    --blue: #3b82f6;
    --success: #22c55e;
    --warning: #f59e0b;
    --danger: #ef4444;
}

/* App Background & Layout */
.stApp { background: linear-gradient(135deg, #000000 0%, #010306 40%, #020409 100%); color: var(--text); }
.block-container { max-width: 1360px; padding-top: 1.4rem; padding-bottom: 2rem; }
[data-testid="stSidebar"] { background: var(--bg-sidebar); border-right: 1px solid var(--border); }
[data-testid="stSidebar"] .stMarkdown p { color: var(--text); }

/* Custom UI Components */
.hero-card { background: rgba(4,6,10,.95); border: 1px solid var(--border); border-radius: 20px; padding: 1.15rem 1.25rem; margin-bottom: 1rem; box-shadow: 0 18px 45px rgba(0,0,0,.6); }
.main-title { font-size: 2.05rem; font-weight: 780; letter-spacing: -.04em; color: #f8fafc; margin-bottom: .12rem; }
.sub-title { font-size: .95rem; color: var(--muted); }
.metric-card { background: rgba(5,8,14,.95); border: 1px solid var(--border-soft); border-radius: 16px; padding: .9rem 1rem; min-height: 88px; }
.metric-label { font-size: .78rem; color: var(--muted); margin-bottom: .28rem; }
.metric-value { font-size: 1.05rem; color: #f8fafc; font-weight: 720; }
.metric-caption { font-size: .75rem; color: var(--muted); margin-top: .18rem; }

/* Conversation Bubbles */
.user-line { background: rgba(59,130,246,.03); border: 1px solid rgba(59,130,246,.08); border-left: 4px solid var(--blue); border-radius: 16px; padding: .9rem 1rem; margin: .7rem 0; }
.assistant-line { background: rgba(45,212,191,.03); border: 1px solid rgba(45,212,191,.08); border-left: 4px solid var(--accent); border-radius: 16px; padding: .9rem 1rem; margin: .7rem 0; }
.msg-meta { font-size: .78rem; color: var(--muted); margin-bottom: .35rem; }

/* Badges & Trace Elements */
.badge { display: inline-block; padding: .18rem .52rem; border-radius: 999px; border: 1px solid var(--border-soft); background: #020407; color: var(--muted); font-size: .73rem; margin-right: .35rem; margin-top: .25rem; }
.badge-ok { border-color: rgba(34,197,94,.15); color: #86efac; background: rgba(34,197,94,.03); }
.badge-error { border-color: rgba(239,68,68,.2); color: #fca5a5; background: rgba(239,68,68,.03); }
.trace-title { font-size: .95rem; font-weight: 700; color: #f8fafc; margin-top: .7rem; margin-bottom: .25rem; }
.step-card { background: #010204; border: 1px solid var(--border-soft); border-radius: 12px; padding: .65rem .8rem; margin-bottom: .48rem; }
.step-main { font-size: .86rem; font-weight: 700; color: #99f6e4; }
.step-detail { font-size: .76rem; color: var(--muted); margin-top: .12rem; }

/* Standard Buttons & Inputs */
.stButton>button { background: #04060a; color: #e5edf7; border: 1px solid var(--border); border-radius: 12px; font-weight: 650; }
.stButton>button:hover { background: #060910; border-color: #2dd4bf; color: #fff; }
.stTextInput input, .stTextArea textarea { background: #020306 !important; color: #e5edf7 !important; border: 1px solid var(--border) !important; border-radius: 13px !important; }

/* Target Bottom Block Elements */
[data-testid="stBottomBlockContainer"] { background: #000000 !important; border-top: 1px solid var(--border); }
[data-testid="stChatInput"] { background: #020306 !important; border: 1px solid var(--border) !important; }
[data-testid="stChatInputTextArea"] { background: transparent !important; color: #e5edf7 !important; }
[data-testid="stToolbar"] { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

def now_time() -> str:
    return datetime.now().strftime("%H:%M:%S")

def clean_json_value(value: Any) -> Any:
    try:
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(k): clean_json_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean_json_value(x) for x in value]
        return str(value)
    except Exception:
        return str(value)

def compact_preview(value: Any, limit: int = 1800) -> str:
    try:
        text = json.dumps(clean_json_value(value), default=str, ensure_ascii=False, indent=2)
    except Exception:
        text = str(value)
    return text[:limit] + "\n...truncated..." if len(text) > limit else text

def add_step(steps: List[Dict[str, Any]], title: str, details: str = "", data: Any = None) -> None:
    item = {
        "time": now_time(),
        "title": title,
        "details": details,
        "data": clean_json_value(data) if data is not None else None
    }
    steps.append(item)
    print(f"[{item['time']}] {title} | {details}", flush=True)
    if data is not None:
        print(compact_preview(data), flush=True)

def backend_select_tool(question: str, user_role: str, top_k: int = 1) -> Dict[str, Any]:
    payload = {
        "question": question,
        "user_role": user_role,
        "top_k": top_k
    }
    try:
        response = requests.post(f"{LLM_SERVER_URL}/select_tool", json=payload, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return clean_json_value(response.json())
    except requests.exceptions.RequestException as e:
        LOGGER.error(f"HTTP request to LLM Server failed: {str(e)}")
        return {
            "selected_tool": None,
            "mode": None,
            "args": {},
            "validated": False,
            "validation_error": f"Failed to connect to LLM server at {LLM_SERVER_URL}. Error: {str(e)}"
        }

def select_tool_no_llm(question: str, user_role: str, steps: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    selection = backend_select_tool(question, user_role)
    tool_name = selection.get("selected_tool")
    result = {
        "selected_tool": tool_name,
        "selected_label": TOOL_LABELS.get(tool_name, str(tool_name)),
        "mode": selection.get("mode"),
        "args": selection.get("args") or {},
        "validated": bool(selection.get("validated", True))
    }
    if selection.get("validation_error"):
        result["validation_error"] = selection.get("validation_error")
        
    result["source"] = selection.get("source", "unknown")
    result["routing_type"] = "llm_server_delegated_no_llm_answer"
    
    add_step(steps, "Tool selected via API (llm_server.py)", f"{tool_name} | source={selection.get('source')}", result)
    return clean_json_value(result)

def safe_json_loads(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        return {}

def execute_tool_direct(tool_name: str, tool_args: Dict[str, Any], steps: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    steps = steps if steps is not None else []
    try:
        add_step(steps, "Direct execution started", f"Importing phishing_mcp_server")
        import phishing_mcp_server as server
        server_map = {
            "run_analytics": server.run_analytics,
            "employee_lookup": server.employee_lookup,
            "predict_risk": server.predict_risk,
            "recommend_actions": server.recommend_actions,
            "simulation_users": server.simulation_users,
            "cache_control": server.cache_control,
            "cache_status": server.cache_status,
            "system_info": server.system_info
        }
        if tool_name not in server_map:
            result = {"status": "error", "message": f"Unsupported tool: {tool_name}"}
            add_step(steps, "Unsupported tool", tool_name, result)
            return result
            
        started = time.time()
        add_step(steps, "Tool execution started", tool_name, tool_args)
        result = server_map[tool_name](**tool_args) or {}
        result = clean_json_value(result)
        add_step(steps, "Tool execution completed", f"{round((time.time() - started) * 1000, 2)} ms", {"status": result.get("status") if isinstance(result, dict) else "unknown"})
        return result
    except Exception as e:
        result = {"status": "error", "message": str(e), "traceback": traceback.format_exc()}
        add_step(steps, "Tool execution failed", str(e), result)
        return result

async def execute_tool_mcp_stdio_async(tool_name: str, tool_args: Dict[str, Any], steps: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    import asyncio
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        add_step(steps, "MCP stdio starting", f"{MCP_SERVER_COMMAND} {MCP_SERVER_SCRIPT}")
        server_params = StdioServerParameters(command=MCP_SERVER_COMMAND, args=[MCP_SERVER_SCRIPT])
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                add_step(steps, "MCP session initialized", tool_name)
                started = time.time()
                result = await session.call_tool(tool_name, arguments=tool_args or {})
                add_step(steps, "MCP tool returned", f"{round((time.time() - started) * 1000, 2)} ms")
                if hasattr(result, "content") and result.content:
                    text = getattr(result.content[0], "text", None)
                    if text:
                        parsed = safe_json_loads(text)
                        final = parsed if parsed else {"status": "success", "raw_text": text}
                        add_step(steps, "MCP response parsed", "", final)
                        return clean_json_value(final)
                final = result.model_dump() if hasattr(result, "model_dump") else result
                add_step(steps, "MCP response returned as object", "", final)
                return clean_json_value(final)
    except Exception as e:
        result = {"status": "error", "message": str(e), "traceback": traceback.format_exc()}
        add_step(steps, "MCP execution failed", str(e), result)
        return result

def execute_tool_mcp_stdio(tool_name: str, tool_args: Dict[str, Any], steps: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    import asyncio
    try:
        return asyncio.run(execute_tool_mcp_stdio_async(tool_name, tool_args, steps))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(execute_tool_mcp_stdio_async(tool_name, tool_args, steps))
        finally:
            loop.close()

def execute_selected_tool(tool_name: str, tool_args: Dict[str, Any], steps: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    if EXECUTION_MODE == "mcp_stdio":
        return execute_tool_mcp_stdio(tool_name, tool_args, steps)
    return execute_tool_direct(tool_name, tool_args, steps)

def summarize_deterministic(tool_name: str, tool_args: Dict[str, Any], tool_output: Dict[str, Any]) -> str:
    if isinstance(tool_output, dict) and "ui_summary" in tool_output:
        return tool_output["ui_summary"]

    status = tool_output.get("status", "unknown") if isinstance(tool_output, dict) else "unknown"
    if status == "error":
        return f"Request Failed. The selected tool returned an error: {tool_output.get('message', 'Unknown error')}"

    tool_label = TOOL_LABELS.get(tool_name, tool_name)
    return f"Executed `{tool_label}` successfully. View the structured output below:"

def run_full_pipeline(question: str, user_role: str) -> Dict[str, Any]:
    started = time.time()
    steps: List[Dict[str, Any]] = []
    add_step(steps, "Request received", f"role={user_role}, question='{question}'")
    try:
        selection = select_tool_no_llm(question, user_role, steps)
        if selection.get("validation_error") and not selection.get("validated"):
            add_step(steps, "Validation failed", selection.get("validation_error"))
            return {
                "status": "validation_error",
                "final_answer": selection.get("validation_error"),
                "selected_tool": selection.get("selected_tool"),
                "tool_args": selection.get("args", {}),
                "tool_output": {},
                "selection": selection,
                "steps": steps,
                "latency_ms": round((time.time() - started) * 1000, 2)
            }
            
        tool_name = selection.get("selected_tool")
        tool_args = selection.get("args") or {}
        tool_output = execute_selected_tool(tool_name, tool_args, steps)
        
        final_answer = summarize_deterministic(tool_name, tool_args, tool_output)
        status = "success" if isinstance(tool_output, dict) and tool_output.get("status") != "error" else "error"
        latency_ms = round((time.time() - started) * 1000, 2)
        
        # FIXED: Removed the invalid keyword arguments and passed the data as a dictionary
        add_step(steps, "Pipeline completed", f"{status} in {latency_ms} ms", {"status": status, "latency_ms": latency_ms})
        
        return {
            "status": status,
            "final_answer": final_answer,
            "selected_tool": tool_name,
            "tool_args": tool_args,
            "tool_output": tool_output,
            "selection": selection,
            "steps": steps,
            "latency_ms": latency_ms
        }
    except Exception as e:
        failure = {"status": "error", "message": str(e), "traceback": traceback.format_exc()}
        add_step(steps, "Pipeline failed", str(e), failure)
        return {
            "status": "error",
            "final_answer": "Something failed while processing the question: " + str(e),
            "selected_tool": None,
            "tool_args": {},
            "tool_output": failure,
            "selection": {},
            "steps": steps,
            "latency_ms": round((time.time() - started) * 1000, 2)
        }

def init_state() -> None:
    defaults = {"messages": [], "user_role": DEFAULT_USER_ROLE, "show_debug": True, "pending_question": ""}
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def queue_question(question: str) -> None:
    st.session_state.pending_question = str(question or "").strip()

def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### Control Panel")
        st.session_state.user_role = st.selectbox("Access mode", ["user", "admin"], index=0 if st.session_state.user_role != "admin" else 1)
        if st.session_state.user_role == "admin":
            st.warning("Admin mode may expose identifiers only if backend policy allows it.")
        else:
            st.info("User mode hides personal identifiers by default.")
        st.session_state.show_debug = st.toggle("Show trace below answers", value=st.session_state.show_debug)
        st.markdown("---")
        st.caption("(No LLM mode)")
        st.caption(f"Router API: {LLM_SERVER_URL}")
        st.caption("Summary: Deterministic only")
        st.caption(f"Execution mode: {EXECUTION_MODE}")
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Clear chat", width="stretch"):
                st.session_state.messages = []
                st.session_state.pending_question = ""
                st.rerun()
        with c2:
            if st.button("Cache status", width="stretch"):
                queue_question("Cache status")
                st.rerun()
        if st.button("Refresh cache", width="stretch"):
            queue_question("Refresh cache")
            st.rerun()
        st.markdown("---")
        st.markdown("### Quick prompts")
        for idx, item in enumerate(EXAMPLE_QUESTIONS):
            if st.button(item, key=f"ex_{idx}", width="stretch"):
                queue_question(item)
                st.rerun()

def render_header() -> None:
    st.markdown(f"""<div class="hero-card"><div class="main-title">{APP_TITLE}</div><div class="sub-title">Selection and argument shaping handled by external API ({LLM_SERVER_URL}) · This UI only renders and executes the chosen tool</div></div>""", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Routing</div><div class="metric-value">API Delegated</div><div class="metric-caption">HTTP POST /select_tool</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Execution</div><div class="metric-value">{EXECUTION_MODE}</div><div class="metric-caption">MCP/Direct tool layer</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Summary</div><div class="metric-value">Deterministic</div><div class="metric-caption">No LLM generation</div></div>""", unsafe_allow_html=True)
    with c4:
        role = st.session_state.user_role
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Access Mode</div><div class="metric-value">{role}</div><div class="metric-caption">Role-aware backend output</div></div>""", unsafe_allow_html=True)
    with st.expander("Current architecture", expanded=False):
        st.code(f"phishing_streamlit_ui_noLLM.py -> HTTP POST to {LLM_SERVER_URL} -> phishing_mcp_server.py executes tool -> UI summary", language="text")

def render_clear_tool_output(tool_output: Any) -> None:
    if isinstance(tool_output, list):
        if tool_output and isinstance(tool_output[0], dict):
            st.dataframe(tool_output, use_container_width=True)
        else:
            st.write(tool_output)
        return

    if not isinstance(tool_output, dict):
        st.code(str(tool_output), language="text")
        return

    ui_highlights = tool_output.get("ui_highlights", [])
    if ui_highlights:
        st.markdown("#### 📊 Highlights")
        cols = st.columns(min(len(ui_highlights), 4))
        for i, highlight in enumerate(ui_highlights):
            cols[i % 4].metric(label=highlight.get("label", ""), value=highlight.get("value", ""))
        st.write("")

    status = tool_output.get("status")
    message = tool_output.get("message")
    c1, c2 = st.columns(2)
    c1.metric("Status", str(status or "-").capitalize())

    count_val = tool_output.get("count", tool_output.get("user_count", tool_output.get("total_rows", tool_output.get("rows_analyzed", "-"))))
    c2.metric("Records / Count", str(count_val))

    if message:
        st.info(str(message))

    skip_keys = {
        "status", "message", "count", "user_count", "total_rows", "rows_analyzed",
        "tool_catalog_entry", "ui_instruction", "trace_instruction", "execution_trace",
        "ui_summary", "ui_highlights"
    }

    for key, value in tool_output.items():
        if key in skip_keys or value is None or value == "":
            continue

        display_title = key.replace("_", " ").title()
        st.markdown(f"#### {display_title}")

        if isinstance(value, list):
            if value and isinstance(value[0], dict):
                st.dataframe(value[:100], use_container_width=True)
                if len(value) > 100:
                    st.caption(f"Showing first 100 of {len(value)} items")
            else:
                for item in value[:20]:
                    st.markdown(f"- {item}")
                if len(value) > 20:
                    st.caption(f"...and {len(value) - 20} more items.")
        elif isinstance(value, dict):
            st.json(value)
        else:
            st.write(value)

def render_steps(steps: List[Dict[str, Any]]) -> None:
    if not steps:
        st.info("No trace steps captured.")
        return
    for step in steps:
        st.markdown(f"""<div class="step-card"><div class="step-main {html.escape(str(step.get("status", ""))) or ""}">{html.escape(str(step.get("time", "")))} | {html.escape(str(step.get("title", "")))}</div><div class="step-detail">{html.escape(str(step.get("details", "")))}</div></div>""", unsafe_allow_html=True)
        if step.get("data") is not None:
            st.code(compact_preview(step.get("data"), limit=1800), language="json")

def render_trace(backend: Dict[str, Any], idx) -> None:
    if not backend or not st.session_state.show_debug:
        return
    selected_tool = backend.get("selected_tool")
    latency_ms = backend.get("latency_ms")
    status = backend.get("status") or "completed"
    badge_class = "badge-ok" if status != "error" else "badge-error"
    tool_output = backend.get("tool_output") or {}
    tool_meta = (tool_output or {}).get("tool_catalog_entry") or {}
    ui_instruction = (tool_output or {}).get("ui_instruction") or (tool_output or {}).get("trace_instruction")
    
    st.markdown(f"""<div class="answer-toolbar"><span class="badge {badge_class}">Tool: {html.escape(str(selected_tool or "-"))}</span><span class="badge {badge_class}">Routing: HTTP API Request</span><span class="badge">Latency: {html.escape(str(latency_ms or "-"))} ms</span><span class="badge {badge_class}">Status: {html.escape(str(status))}</span></div>""", unsafe_allow_html=True)
    
    if tool_meta:
        st.caption(f"Tool contract: {tool_meta.get('description', '')}")
    if ui_instruction:
        st.info(ui_instruction)
        
    with st.expander("Trace", expanded=False):
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Tool Selection", "Tool Arguments", "Tool Contract", "Execution Steps", "Raw JSON", "Download"])
        with tab1:
            st.markdown('<div class="trace-title">Tool Selection - API Server</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.metric("Selected Tool", str(backend.get("selected_tool") or "-"))
            selection = backend.get("selection") or {}
            c2.metric("Mode", str(selection.get("mode") or "-"))
            st.markdown("#### Selection JSON")
            st.json(selection)
        with tab2:
            st.markdown('<div class="trace-title">Tool Arguments</div>', unsafe_allow_html=True)
            st.json(backend.get("tool_args") or {})
        with tab3:
            st.markdown('<div class="trace-title">Tool Contract</div>', unsafe_allow_html=True)
            st.json(tool_meta or {})
            if isinstance(tool_output, dict):
                st.json({
                    "ui_highlights": tool_output.get("ui_highlights") or [],
                    "ui_summary": tool_output.get("ui_summary"),
                    "execution_trace": tool_output.get("execution_trace") or [],
                })
        with tab4:
            st.markdown('<div class="trace-title">Execution Steps</div>', unsafe_allow_html=True)
            render_steps(backend.get("steps") or [])
        with tab5:
            st.markdown('<div class="trace-title">Raw JSON Trace</div>', unsafe_allow_html=True)
            st.json(backend)
        with tab6:
            trace_key = f"trace_download_{backend.get('request_id', id(backend))}"
            st.download_button("Download trace JSON", data=json.dumps(backend, indent=2, default=str), file_name=f"trace_{idx}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", mime="application/json", key=trace_key)

def render_chat() -> None:
    if not st.session_state.messages:
        st.info("Ask a question below or choose a quick prompt from the sidebar.")
        return
    st.markdown("### Conversation")
    for idx, msg in enumerate(st.session_state.messages):
        role = msg.get("role")
        if role == "user":
            st.markdown(f"""<div class="user-line"><div class="msg-meta">You ({msg.get('time', '')})</div><div class="msg-content">{msg.get('content', '')}</div></div>""", unsafe_allow_html=True)
        else:
            backend = msg.get("backend", {}) or {}
            tool = backend.get("selected_tool")
            latency = backend.get("latency_ms")
            meta = f"Copilot ({msg.get('time', '')})"
            if tool:
                meta += f" · {tool} ({latency} ms)"
            st.markdown(f"""<div class="assistant-line"><div class="msg-meta">{meta}</div><div class="msg-content">{msg.get('content', '')}</div></div>""", unsafe_allow_html=True)
            tool_output = backend.get("tool_output")
            if tool_output not in (None, {}):
                with st.container():
                    st.write("")
                    render_clear_tool_output(tool_output)
            render_trace(backend, idx)

def handle_question(question: str) -> None:
    question = str(question or "").strip()
    if not question:
        return
    st.session_state.messages.append({"role": "user", "content": question, "time": now_time()})
    try:
        with st.status("Running phishing analytics workflow...", expanded=False) as status:
            result = run_full_pipeline(question, st.session_state.user_role)
            if result.get("status") in ["success", "validation_error"]:
                status.update(label="Workflow completed", state="complete", expanded=False)
            else:
                status.update(label="Workflow completed with issue", state="error", expanded=False)
            st.session_state.messages.append({"role": "assistant", "content": result.get("final_answer") or "Could not generate a final answer.", "time": now_time(), "backend": result})
    except Exception as e:
        error_backend = {"status": "error", "traceback": traceback.format_exc(), "logs": [{"time": now_time(), "step": "UI EXCEPTION", "message": str(e), "data": traceback.format_exc()}]}
        print(f"UI EXCEPTION: {str(e)}", flush=True)
        print(traceback.format_exc(), flush=True)
        st.session_state.messages.append({"role": "assistant", "content": f"Something failed while processing the question: {str(e)}", "time": now_time(), "backend": error_backend})

def main() -> None:
    init_state()
    render_sidebar()
    render_header()
    render_chat()
    if st.session_state.pending_question:
        pending = st.session_state.pending_question
        st.session_state.pending_question = ""
        handle_question(pending)
        st.rerun()
    if question := st.chat_input("Ask about phishing analytics, predictions, recommendations, cache, or system status..."):
        handle_question(question)
        st.rerun()

if __name__ == "__main__":
    main()
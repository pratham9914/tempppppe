# llm_server.py
# 
# * **For `phishing_streamit_ui_noLLM`:** We use the top 1 tool and the LLM for argument generation, and nothing else.
# * **For `phishing_streamlit_ui`:** We choose top tools > threshold, then use the LLM for argument generation for all of them, and finally generate a summary.

# **Core Responsibilities (`llm_server.py`)**

# * **Semantic Router:** Acts as the brain to decide *which* tools to use and *how* to use them, without ever executing them.
# * **Hard-Rule Bypass:** Instantly routes system-level queries (e.g., "health check", "cache status") to tools, completely skipping the LLM to eliminate latency.
# * **Intent Overrides:** Forces specific tool selection for strong keywords (e.g., "predict", "improve") before checking semantic scores.
# * **Semantic Scoring:** Uses `SentenceTransformer` to mathematically rank tools based on the user's query.
# * **Argument Generation:** Uses a locally hosted LLM to extract user parameters into strict JSON payloads for the selected tool(s).
# * **Anti-Hallucination Guardrails:** Cleans LLM outputs (e.g., forcefully updating past years to "2026", preventing cities from being mapped as BRIDs).
# * **Schema Validation:** Verifies the generated JSON has all required fields; flags missing data before handing it off.
# * **Dual-Model Orchestration:** Manages the handoff between the lightweight semantic router and the heavier generation LLM.

import json
import logging
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
try:
    import torch
except ImportError:
    torch = None
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

# Required Environment Variables
APP_NAME = os.getenv("APP_NAME", "phishing_llm_router")
MODEL_PATH = os.getenv("MODEL_PATH", "")
SEMANTIC_MODEL_PATH = os.getenv("SEMANTIC_MODEL_PATH", "")
MAX_NEW_TOKENS_ROUTER = int(os.getenv("MAX_NEW_TOKENS_ROUTER", "200"))
MAX_NEW_TOKENS_SUMMARY = int(os.getenv("MAX_NEW_TOKENS_SUMMARY", "200"))
DEFAULT_THRESHOLD = float(os.getenv("DEFAULT_THRESHOLD", "0.60"))
MIN_PREDICTION_PAYLOAD_FIELDS = int(os.getenv("MIN_PREDICTION_PAYLOAD_FIELDS", "1"))

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

LLM_TOKENIZER = None
LLM_MODEL = None
SEMANTIC_MODEL = None
TOOL_EMBEDDINGS = None

app = FastAPI()

TOOL_CATALOG: Dict[str, Dict[str, Any]] = {
    "run_analytics": {
        "description": "Historical aggregate phishing analytics: counts, percentages, rates, trends, grouped summaries.",
        "modes": [],
        "keywords": ["percentage", "percent", "count", "rate", "trend", "city", "department", "campaign", "summary", "group", "aggregate"],
        "schema": {"analysis_type": "str", "group_by": "list[str]", "filters": "dict", "user_role": "str", "top_n": "int"},
        "mode_guidance": [
            "Use for historical aggregate metrics, counts, percentages, rates, trends, and grouped summaries.",
            "Use when the question asks about city, department, campaign, template, subject, month, year, designation, grade, or overall metrics."
        ]
    },
    "employee_lookup": {
        "description": "Employee historical lookup: profile, find employees, top historically risky employees. Not ML prediction.",
        "modes": ["profile", "top_risky", "find"],
        "keywords": ["profile", "employee", "employees", "brid", "find", "lookup", "history", "top risky", "historically risky", "historical high risk"],
        "schema": {"mode": "profile|top_risky|find", "brid": "str|null", "city": "str|null", "department": "str|null", "limit": "int", "user_role": "str"},
        "mode_guidance": [
            "Use profile when the user asks for a specific BRID profile, user history, or employee historical record.",
            "Use top_risky when the user asks for top risky employees or high-risk user history."
        ]
    },
    "predict_risk": {
        "description": "All ML prediction flows: by BRID, from manual fields, recent population, predicted high-risk population.",
        "modes": ["by_brid", "from_payload", "recent_population", "high_risk_population"],
        "keywords": ["predict", "prediction", "probability", "likely", "chance", "forecast", "ml", "model", "predicted high risk", "high risk population", "city", "department", "subject", "campaign", "payload"],
        "schema": {"mode": "by_brid|from_payload|recent_population|high_risk_population", "brid": "str|null", "payload": "dict|null", "limit": "int", "user_role": "str"},
        "mode_guidance": [
            "Use by_brid only when the user provides a real BRID value.",
            "Use from_payload when the user provides prediction fields in natural language instead of BRID.",
            "Use high_risk_population when the user asks for predicted high-risk users."
        ]
    },
    "recommend_actions": {
        "description": "Improvement guidance, training suggestions, risk-reduction actions.",
        "modes": ["employee_improvement", "group_recommendations", "overall_recommendations"],
        "keywords": ["improve", "improvement", "recommend", "recommendation", "training", "actions", "action", "reduce clicks", "reduce phishing clicks", "guidance"],
        "schema": {"mode": "employee_improvement|group_recommendations|overall_recommendations", "brid": "str|null", "group_by": "list[str]", "filters": "dict", "top_n": "int", "user_role": "str"},
        "mode_guidance": [
            "Use employee_improvement when the user asks how a specific BRID or employee can improve.",
            "Use group_recommendations for a department, city, campaign, group, designation, grade, template, or subject."
        ]
    },
    "simulation_users": {
        "description": "Users who clicked, reported, or took no action in a simulation or campaign.",
        "modes": [],
        "keywords": ["users who", "who clicked", "who reported", "trapped", "simulation users", "campaign users", "no action users", "list users"],
        "schema": {"campaign_month": "str|null", "campaign_year": "str|null", "campaign_name": "str|null", "event_type": "str", "user_role": "str", "limit": "int"},
        "mode_guidance": [
            "Use when the user asks who clicked, who reported, trapped users, no-action users, or campaign user lists."
        ]
    },
    "cache_control": {
        "description": "Cache-changing actions: refresh or clear.",
        "modes": ["refresh", "clear"],
        "keywords": ["refresh cache", "clear cache", "reload cache", "update cache", "rebuild cache", "reset cache", "delete cache", "new data added"],
        "schema": {"action": "refresh|clear"},
        "mode_guidance": ["Use refresh when the user asks to refresh, reload, update, or rebuild cache."]
    },
    "cache_status": {
        "description": "Read cache status or cache statistics.",
        "modes": [],
        "keywords": ["cache status", "cache statistics", "cache stats", "cache loaded", "refresh time", "cache rows", "cache count", "cache info"],
        "schema": {"include_statistics": "bool"},
        "mode_guidance": ["Use when the user asks to read cache status, cache statistics, refresh time, or cache rows."]
    },
    "system_info": {
        "description": "System diagnostics: health check, database schema, model features, environment/config.",
        "modes": ["health", "schema", "features", "environment"],
        "keywords": ["health", "health check", "schema", "table schema", "columns", "available columns", "feature", "features", "feature columns", "environment"],
        "schema": {"mode": "health|schema|features|environment"},
        "mode_guidance": ["Use health for health check or diagnostics.", "Use schema for table schema, database schema, or columns."]
    }
}

class SelectToolRequest(BaseModel):
    question: str
    user_role: str = "user"
    top_k: int = 1
    previous_selected_tools: Optional[List[str]] = None

class SelectToolBatchRequest(BaseModel):
    question: str
    user_role: str = "user"
    top_k: int = 3
    similarity_threshold: float = DEFAULT_THRESHOLD
    previous_selected_tools: Optional[List[str]] = None

class SummarizeRequest(BaseModel):
    question: str
    tool_name: str
    tool_args: Dict[str, Any] = Field(default_factory=dict)
    tool_output: Dict[str, Any]
    user_role: str = "user"

def clean_json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return None if (np is not None and np.isnan(value)) else value
    if isinstance(value, dict):
        return {str(k): clean_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json_value(x) for x in value]
    return str(value)

def terminal_json(title: str, data: Any, max_chars: int = 12000) -> None:
    try:
        text = json.dumps(clean_json_value(data), indent=2, ensure_ascii=False, default=str)
    except Exception:
        text = str(data)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...truncated..."
    print("\n" + "=" * 90, flush=True)
    print(title, flush=True)
    print("-" * 90, flush=True)
    print(text, flush=True)
    print("-" * 90 + "\n", flush=True)

def log_router_event(event: str, message: str, data: Any = None) -> None:
    payload = clean_json_value(data) if data is not None else {}
    LOGGER.info("%s | %s | %s", event, message, json.dumps(payload, ensure_ascii=False, default=str))

def parse_llm_json(raw: str) -> Dict[str, Any]:
    text = str(raw or "").strip()
    if not text:
        return {}
    if text.startswith("```json"):
        text = text[7:].strip()
    if text.startswith("```"):
        text = text[3:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}

def load_local_llm():
    global LLM_TOKENIZER, LLM_MODEL
    if LLM_TOKENIZER is not None and LLM_MODEL is not None:
        return LLM_TOKENIZER, LLM_MODEL
    if torch is None or AutoTokenizer is None or AutoModelForCausalLM is None:
        LOGGER.warning("LOCAL_LLM_DEPENDENCIES_UNAVAILABLE")
        return None, None
    try:
        LOGGER.info(f"LOCAL_LLM_LOAD_START | path={MODEL_PATH}")
        LLM_TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        LLM_MODEL = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        LLM_MODEL.eval()
        if LLM_TOKENIZER.pad_token_id is None:
            LLM_TOKENIZER.pad_token_id = LLM_TOKENIZER.eos_token_id
        print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
        LOGGER.info("LOCAL_LLM_LOAD_SUCCESS")
        return LLM_TOKENIZER, LLM_MODEL
    except Exception as exc:
        LOGGER.warning(f"LOCAL_LLM_LOAD_FAILED | {str(exc)}")
        return None, None

def call_local_llm(prompt: str, system: Optional[str] = None, temperature: float = 0.0, max_new_tokens: int = 200) -> str:
    if torch is None or AutoTokenizer is None or AutoModelForCausalLM is None:
        return ""
    try:
        tok, mdl = load_local_llm()
        if tok is None or mdl is None:
            return ""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        encoded = tok.apply_chat_template(messages, tokenize=True, return_tensors="pt", return_dict=True, add_generation_prompt=True)
        input_ids = encoded["input_ids"].to(mdl.device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(mdl.device)
        with torch.inference_mode():
            outputs = mdl.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tok.eos_token_id,
                eos_token_id=tok.eos_token_id,
            )
        text = tok.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        LOGGER.info(f"LOCAL_LLM_GENERATION_SUCCESS | chars={len(text)}")
        return text
    except Exception as e:
        LOGGER.exception(f"LOCAL_LLM_GENERATION_FAILED | {str(e)}")
        return ""

call_ollama = call_local_llm

def get_semantic_model():
    global SEMANTIC_MODEL
    if SEMANTIC_MODEL is not None:
        return SEMANTIC_MODEL if SEMANTIC_MODEL is not False else None
    try:
        from sentence_transformers import SentenceTransformer
        model_path = SEMANTIC_MODEL_PATH.strip() if SEMANTIC_MODEL_PATH else "all-MiniLM-L6-v2"
        LOGGER.info(f"SEMANTIC_MODEL_LOAD_START | path={model_path}")
        SEMANTIC_MODEL = SentenceTransformer(model_path)
        LOGGER.info("SEMANTIC_MODEL_LOAD_SUCCESS")
        return SEMANTIC_MODEL
    except Exception as e:
        LOGGER.warning(f"SEMANTIC_MODEL_UNAVAILABLE | {str(e)}")
        SEMANTIC_MODEL = False
        return None

def tool_text(name: str, meta: Dict[str, Any]) -> str:
    return " ".join([name, meta.get("description", ""), " ".join(meta.get("keywords", [])), " ".join(meta.get("mode_guidance", []))])

def keyword_top_1_tool(question: str) -> List[Dict[str, Any]]:
    low = str(question or "").lower()
    ranked = []
    for name, meta in TOOL_CATALOG.items():
        score = 0.0
        for kw in meta.get("keywords", []):
            if str(kw).lower() in low:
                score += 4.0 if " " in str(kw) else 1.0
        ranked.append((name, score))
    ranked = sorted(ranked, key=lambda x: x[1], reverse=True)
    name, score = ranked[0]
    return [{"tool_name": name, "score": round(float(score), 4), "description": TOOL_CATALOG[name].get("description")}]

def hard_rule_tool_selection(question: str) -> Optional[Dict[str, Any]]:
    low = str(question or "").lower()

    if any(token in low for token in ["cache status", "cache stats", "cache loaded", "refresh time", "cache rows", "is cache loaded"]):
        return {"tool_name": "cache_status", "score": 1.0, "source": "hard_rule"}
    if any(token in low for token in ["refresh cache", "clear cache", "reload cache", "reset cache", "delete cache", "rebuild cache"]):
        return {"tool_name": "cache_control", "score": 1.0, "source": "hard_rule"}
    if any(token in low for token in ["health", "health check", "schema", "table schema", "columns", "feature", "features", "environment", "config", "diagnostic"]):
        if "schema" in low or "table schema" in low or "columns" in low:
            mode = "schema"
        elif "features" in low or "feature" in low:
            mode = "features"
        elif "environment" in low or "config" in low:
            mode = "environment"
        else:
            mode = "health"
        return {"tool_name": "system_info", "mode": mode, "score": 1.0, "source": "hard_rule"}

    return None

def rank_tool_candidates(question: str, top_k: int = 3) -> List[Dict[str, Any]]:
    low = str(question or "").lower()
    ranked: List[Dict[str, Any]] = []
    for name, meta in TOOL_CATALOG.items():
        keyword_score = 0.0
        for kw in meta.get("keywords", []):
            token = str(kw).lower()
            if token in low:
                keyword_score += 4.0 if " " in token else 1.0
        ranked.append({
            "tool_name": name,
            "keyword_score": round(float(keyword_score), 4),
            "semantic_score": 0.0,
            "final_score": round(float(keyword_score), 4),
            "description": meta.get("description", "")
        })
    if np is None:
        ranked.sort(key=lambda item: item["final_score"], reverse=True)
        return ranked[: max(1, int(top_k))] if top_k is not None else ranked
        
    model = get_semantic_model()
    if model is not None:
        try:
            global TOOL_EMBEDDINGS
            names = [item["tool_name"] for item in ranked]
            if TOOL_EMBEDDINGS is None:
                tool_texts = [tool_text(name, TOOL_CATALOG[name]) for name in names]
                TOOL_EMBEDDINGS = model.encode(tool_texts, normalize_embeddings=True)
            q_emb = model.encode([question], normalize_embeddings=True)[0]
            scores = np.dot(TOOL_EMBEDDINGS, q_emb)
            for idx, item in enumerate(ranked):
                item["semantic_score"] = round(float(scores[idx]), 4)
                item["final_score"] = round(float(item["keyword_score"] + item["semantic_score"] * 0.35), 4)
        except Exception as exc:
            LOGGER.warning(f"TOOL_RANKING_FALLBACK | {str(exc)}")
            
    ranked.sort(key=lambda item: item["final_score"], reverse=True)
    return ranked[: max(1, int(top_k))] if top_k is not None else ranked

def orchestrate_tool_selection(question: str, user_role: str, previous_selected_tools: Optional[List[str]] = None, top_k: int = 1) -> Dict[str, Any]:
    hard_rule = hard_rule_tool_selection(question)
    if hard_rule is not None:
        return {
            "selected_tool": hard_rule["tool_name"],
            "mode": hard_rule.get("mode"),
            "args": {},
            "selection_mode": "hard_rule",
            "source": hard_rule["source"],
            "candidates": [hard_rule]
        }
    effective_top_k = max(1, int(top_k or 1))
    ranked = rank_tool_candidates(question, top_k=effective_top_k)
    selected = ranked[0]
    return {
        "selected_tool": selected["tool_name"], "mode": None, "args": {},
        "selection_mode": "semantic_top_1", "source": "semantic_ranking", "candidates": ranked
    }

def orchestrate_tool_batch_selection(question: str, user_role: str, previous_selected_tools: Optional[List[str]] = None, top_k: int = 3, similarity_threshold: float = 0.60) -> Dict[str, Any]:
    hard_rule = hard_rule_tool_selection(question)
    if hard_rule is not None:
        selected = dict(hard_rule)
        selected["mode"] = hard_rule.get("mode")
        return {"selected_tools": [selected], "selection_mode": "hard_rule", "source": hard_rule["source"], "threshold": similarity_threshold}
        
    effective_top_k = max(1, int(top_k or 3))
    ranked = rank_tool_candidates(question, top_k=max(effective_top_k, len(TOOL_CATALOG)))
    
    candidates = [item for item in ranked if item.get("final_score", 0.0) >= float(similarity_threshold)]
    if not candidates:
        candidates = ranked[:max(1, effective_top_k)]
        
    return {"selected_tools": candidates, "selection_mode": "semantic_batch", "source": "semantic_ranking", "threshold": similarity_threshold}

def build_batch_tool_argument_prompt(question: str, user_role: str, selected_tools: List[Dict[str, Any]]) -> str:
    payload = [{
        "tool_name": item["tool_name"],
        "description": TOOL_CATALOG.get(item["tool_name"], {}).get("description", ""),
        "allowed_modes": TOOL_CATALOG.get(item["tool_name"], {}).get("modes", []),
        "mode_guidance": TOOL_CATALOG.get(item["tool_name"], {}).get("mode_guidance", []),
        "schema": TOOL_CATALOG.get(item["tool_name"], {}).get("schema", {})
    } for item in selected_tools]
    return f"""User role: {user_role}
User question: {question}
Selected MCP tools: {json.dumps(payload, ensure_ascii=False)}
Task: Generate JSON arguments for each selected tool. Return valid JSON: {{ "tools": [ {{"tool_name": "<name>", "mode": <mode>, "args": {{...}} }} ] }}""".strip()

def parse_llm_tool_payloads(raw: str) -> List[Dict[str, Any]]:
    parsed = parse_llm_json(raw)
    if isinstance(parsed, dict) and "tools" in parsed:
        return [item for item in parsed["tools"] if isinstance(item, dict)]
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    return []

def llm_generate_arguments_for_tool_batch(question: str, user_role: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not candidates:
        return []
    prompt = build_batch_tool_argument_prompt(question, user_role, candidates)
    raw = call_ollama(prompt, system="Return only valid JSON. No markdown. No explanation.", temperature=0.0, max_new_tokens=MAX_NEW_TOKENS_ROUTER)
    parsed_payloads = parse_llm_tool_payloads(raw)

    results: List[Dict[str, Any]] = []
    for candidate in candidates:
        tool_name = candidate["tool_name"]
        matched = next((item for item in parsed_payloads if item.get("tool_name") == tool_name), None)
        if not matched:
            normalized = normalize_args_by_schema(tool_name, None, {}, user_role, question)
            results.append({"tool_name": tool_name, "mode": normalized.get("mode"), "args": normalized, "source": "semantic_batch_default"})
            continue

        args = normalize_args_by_schema(tool_name, matched.get("mode"), matched.get("args", {}), user_role, question)
        results.append({"tool_name": tool_name, "mode": args.get("mode"), "args": args, "source": "semantic_batch_llm_args"})
    return results


def build_single_tool_argument_prompt(question: str, user_role: str, selected_tool: Dict[str, Any]) -> str:
    payload = {
        "tool_name": selected_tool["tool_name"],
        "description": selected_tool.get("description", ""),
        "allowed_modes": selected_tool.get("modes", []),
        "mode_guidance": selected_tool.get("mode_guidance", []),
        "schema": selected_tool.get("schema", {})
    }
    return f"""User role: {user_role}
User question: {question}
Selected MCP tool: {json.dumps(payload, ensure_ascii=False)}
Task: Generate JSON arguments for the tool based on the user's question. Use null for missing values. 
Return strictly valid JSON: {{ "mode": null, "args": {{}} }}""".strip()


def llm_generate_arguments_for_tool(question: str, user_role: str, selected_tool: Dict[str, Any]) -> Dict[str, Any]:
    prompt = build_single_tool_argument_prompt(question, user_role, selected_tool)
    raw = call_ollama(prompt, system="Return only valid JSON. No markdown. No explanation.", temperature=0.0, max_new_tokens=MAX_NEW_TOKENS_ROUTER)
    parsed = parse_llm_json(raw)
    llm_args = parsed.get("args") if isinstance(parsed, dict) and isinstance(parsed.get("args"), dict) else {}
    normalized = normalize_args_by_schema(selected_tool["tool_name"], parsed.get("mode") if isinstance(parsed, dict) else None, llm_args, user_role, question)
    return clean_json_value({
        "selected_tool": selected_tool["tool_name"],
        "mode": normalized.get("mode"),
        "args": normalized,
        "source": "semantic_top_1_plus_llm_args"
    })


def extract_brid_from_question(question: str) -> Optional[str]:
    if not question:
        return None
    match = re.search(r"\bbrid\b[^a-z0-9]{0,5}([a-z0-9._-]{2,})", question, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r"\b([a-z0-9._-]{3,}-[a-z0-9._-]{1,})\b", question, flags=re.IGNORECASE)
    if match and "2026" not in match.group(1):
        return match.group(1)
    return None


def normalize_args_by_schema(tool: str, mode: Any, args: Dict[str, Any], user_role: str, question: str = "") -> Dict[str, Any]:
    args = dict(args or {})
    low_q = str(question).lower()

    for key in ["tool_name", "schema", "description", "allowed_modes", "mode_guidance", "modes", "threshold"]:
        args.pop(key, None)

    schema = TOOL_CATALOG.get(tool, {}).get("schema", {})
    if "user_role" in schema:
        args["user_role"] = user_role

    brid_value = args.get("brid") or extract_brid_from_question(question)
    if brid_value:
        args["brid"] = brid_value

    for k, v in args.items():
        if isinstance(v, str) and "2022" in v:
            args[k] = v.replace("2022", "2026")
    if "filters" in args and isinstance(args["filters"], dict):
        for k, v in args["filters"].items():
            if isinstance(v, str) and "2022" in v:
                args["filters"][k] = v.replace("2022", "2026")

    if tool == "run_analytics":
        args["analysis_type"] = args.get("analysis_type") or "overall_analysis"
        args["group_by"] = args.get("group_by") if isinstance(args.get("group_by"), list) else ([] if args.get("group_by") is None else [args.get("group_by")])
        args["filters"] = args.get("filters") if isinstance(args.get("filters"), dict) else {}
        args["top_n"] = int(args.get("top_n") or 20)
    elif tool == "employee_lookup":
        mode = str(mode or args.get("mode") or "").lower() or None
        if not mode:
            if args.get("brid"):
                mode = "profile"
            elif "top risky" in low_q or "historically risky" in low_q:
                mode = "top_risky"
            elif "find" in low_q or args.get("city") or args.get("department"):
                mode = "find"
        args["mode"] = mode
        args["limit"] = int(args.get("limit") or 10)
    elif tool == "predict_risk":
        mode = str(mode or args.get("mode") or "").lower() or None
        args["payload"] = args.get("payload") if isinstance(args.get("payload"), dict) else {}
        if not args.get("limit") and re.search(r"limit\s*(\d+)", low_q):
            args["limit"] = int(re.search(r"limit\s*(\d+)", low_q).group(1))
        else:
            args["limit"] = int(args.get("limit")) if args.get("limit") else None

        brid_val = str(args.get("brid") or "").strip()
        if brid_val and (len(brid_val) < 5 or (not any(char.isdigit() for char in brid_val) and "-" not in brid_val)):
            if brid_val.lower() not in ["null", "none", ""]:
                args["payload"]["city"] = brid_val
            args["brid"] = None
            brid_val = ""

        if not mode:
            if brid_val:
                mode = "by_brid"
            elif "recent population" in low_q:
                mode = "recent_population"
            elif "high risk" in low_q:
                mode = "high_risk_population"
            elif args.get("payload") or any(k in low_q for k in ["city", "department", "subject", "campaign"]):
                mode = "from_payload"
        args["mode"] = mode
    elif tool == "recommend_actions":
        mode = str(mode or args.get("mode") or "").lower() or None
        if not mode:
            if args.get("brid"):
                mode = "employee_improvement"
            elif "department" in low_q or "city" in low_q or "group" in low_q or "risky" in low_q:
                mode = "group_recommendations"
            elif "reduce" in low_q or "what should we do" in low_q:
                mode = "overall_recommendations"
        args["mode"] = mode
        args["group_by"] = args.get("group_by") if isinstance(args.get("group_by"), list) else ([] if args.get("group_by") is None else [args.get("group_by")])
        args["filters"] = args.get("filters") if isinstance(args.get("filters"), dict) else {}
        args["top_n"] = int(args.get("top_n") or 5)
    elif tool == "simulation_users":
        if "who clicked" in low_q:
            args["event_type"] = "clicked link"
        if "march" in low_q and not args.get("campaign_month"):
            args["campaign_month"] = "March"
        if "2026" in low_q and not args.get("campaign_year"):
            args["campaign_year"] = "2026"
        args["limit"] = int(args.get("limit") or 5000)
    elif tool == "cache_control":
        args["action"] = str(args.get("action") or "").lower() or "refresh"
    elif tool == "cache_status":
        args["include_statistics"] = bool(args.get("include_statistics", True))
    elif tool == "system_info":
        mode = str(mode or args.get("mode") or "").lower() or "health"
        if "schema" in low_q:
            mode = "schema"
        args["mode"] = mode

    return clean_json_value(args)


def validate_tool_selection(selection: Dict[str, Any], user_role: str, question: str = "") -> Dict[str, Any]:
    tool = selection.get("selected_tool")
    if tool not in TOOL_CATALOG:
        return clean_json_value({"selected_tool": None, "mode": None, "args": {}, "validated": False, "validation_error": f"Unknown tool selected: {tool}"})

    args = normalize_args_by_schema(tool, selection.get("mode"), selection.get("args", {}), user_role, question)
    mode = args.get("mode")

    missing_fields = []
    if tool == "employee_lookup":
        if mode not in ["profile", "top_risky", "find"]:
            missing_fields.append("mode")
        if mode == "profile" and not args.get("brid"):
            missing_fields.append("brid")
    elif tool == "predict_risk":
        if mode not in ["by_brid", "from_payload", "recent_population", "high_risk_population"]:
            missing_fields.append("mode")
        if mode == "by_brid" and not args.get("brid"):
            missing_fields.append("brid")
        if mode == "from_payload":
            payload = args.get("payload")
            useful_payload = {k: v for k, v in payload.items() if v not in [None, "", [], {}]} if isinstance(payload, dict) else {}
            if len(useful_payload) < MIN_PREDICTION_PAYLOAD_FIELDS:
                missing_fields.append("additional_prediction_fields")
    elif tool == "recommend_actions":
        if mode not in ["employee_improvement", "group_recommendations", "overall_recommendations"]:
            missing_fields.append("mode")
        if mode == "employee_improvement" and not args.get("brid"):
            missing_fields.append("brid")
    elif tool == "simulation_users":
        if not args.get("campaign_month") and not args.get("campaign_year") and not args.get("campaign_name"):
            missing_fields.append("campaign_month/campaign_year/campaign_name")
        if not args.get("event_type"):
            missing_fields.append("event_type")
    elif tool == "cache_control":
        if args.get("action") not in ["refresh", "clear"]:
            missing_fields.append("action")
    elif tool == "system_info":
        if args.get("mode") not in ["health", "schema", "features", "environment"]:
            missing_fields.append("mode")

    selection["mode"] = mode
    selection["args"] = clean_json_value(args)
    selection["validated"] = len(missing_fields) == 0
    if missing_fields:
        selection["validation_error"] = "Missing required field(s): " + ", ".join(missing_fields)
    return clean_json_value(selection)

def summarize_with_llm(question: str, tool_name: str, tool_args: Dict[str, Any], tool_output: Dict[str, Any], user_role: str = "user") -> Dict[str, Any]:
    ui_text = tool_output.get("ui_summary", "")
    
    prompt = f"""You are a helpful summarization engine. 
Task: Generate a VERY CONCISE, human-readable summary of the provided tool output. Focus purely on explaining the findings.
User Question: {question}
Tool Human-Readable Summary: {ui_text}
Tool Output JSON context (if needed): {json.dumps(clean_json_value(tool_output), ensure_ascii=False)[:3000]}

Produce a very concise 1-3 sentence summary.""".strip()

    raw = call_ollama(prompt, system="Be extremely concise. Answer the user directly based on the tool results.", temperature=0.0, max_new_tokens=MAX_NEW_TOKENS_SUMMARY)
    
    if raw:
        return {"answer": raw, "used_llm": True, "summary_mode": "local_transformers_llm"}
    return {"answer": ui_text or "Task completed successfully.", "used_llm": False, "summary_mode": "deterministic_fallback"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "app": APP_NAME,
        "tools": list(TOOL_CATALOG.keys()),
        "local_llm_model_path": MODEL_PATH,
        "local_llm_loaded": LLM_MODEL is not None,
        "semantic_model_loaded": SEMANTIC_MODEL is not None and SEMANTIC_MODEL is not False
    }

@app.get("/tools")
def get_tools():
    return {"status": "success", "tools": [{"tool_name": n, "description": c["description"]} for n, c in TOOL_CATALOG.items()]}

@app.post("/select_tool")
def select_tool(req: SelectToolRequest):
    start = time.time()
    try:
        question = req.question.strip()
        user_role = req.user_role or "user"
        selection = orchestrate_tool_selection(question, user_role, req.previous_selected_tools, top_k=max(1, int(req.top_k or 1)))
        
        if not selection.get("selected_tool"):
            return {"selected_tool": None, "mode": None, "args": {}, "validated": False, "validation_error": "No candidate tool found.", "latency_ms": round((time.time() - start) * 1000, 2)}
        
        if selection.get("selection_mode") == "hard_rule":
            selected = {
                "selected_tool": selection["selected_tool"],
                "mode": selection.get("mode"),
                "args": {},
                "selection_mode": "hard_rule",
                "source": "hard_rule_bypass"
            }
        else:
            selected_tool_info = {
                "tool_name": selection["selected_tool"],
                "description": TOOL_CATALOG[selection["selected_tool"]].get("description", ""),
                "modes": TOOL_CATALOG[selection["selected_tool"]].get("modes", []),
                "mode_guidance": TOOL_CATALOG[selection["selected_tool"]].get("mode_guidance", []),
                "schema": TOOL_CATALOG[selection["selected_tool"]].get("schema", {})
            }
            selected = llm_generate_arguments_for_tool(question, user_role, selected_tool_info)
            selected["selection_mode"] = selection.get("selection_mode")
            selected["source"] = selection.get("source") or selected.get("source")

        validated = validate_tool_selection(selected, user_role, question)
        validated["latency_ms"] = round((time.time() - start) * 1000, 2)
        return clean_json_value(validated)
    except Exception as exc:
        LOGGER.exception("SELECT_TOOL_ROUTE_FAILED | %s", str(exc))
        return {"selected_tool": None, "mode": None, "args": {}, "validated": False, "validation_error": str(exc), "latency_ms": round((time.time() - start) * 1000, 2)}

@app.post("/select_tool_batch")
def select_tool_batch(req: SelectToolBatchRequest):
    start = time.time()
    try:
        question = req.question.strip()
        user_role = req.user_role or "user"
        threshold = float(req.similarity_threshold or DEFAULT_THRESHOLD)
        
        selection = orchestrate_tool_batch_selection(question, user_role, req.previous_selected_tools, top_k=max(1, int(req.top_k or 3)), similarity_threshold=threshold)
        tools = selection.get("selected_tools", [])
        
        if not tools:
            return {"selected_tools": [], "validated": False, "validation_error": "No candidate tools found.", "latency_ms": round((time.time() - start) * 1000, 2)}
            
        if selection.get("selection_mode") == "hard_rule":
            selected_tool_args = [
                {"tool_name": t["tool_name"], "mode": t.get("mode"), "args": {}, "source": "hard_rule_bypass"}
                for t in tools
            ]
        else:
            selected_tool_args = llm_generate_arguments_for_tool_batch(question, user_role, tools)
            
        return clean_json_value({
            "selected_tools": selected_tool_args,
            "latency_ms": round((time.time() - start) * 1000, 2)
        })
    except Exception as exc:
        LOGGER.exception("SELECT_TOOL_BATCH_ROUTE_FAILED | %s", str(exc))
        return {"selected_tools": [], "validated": False, "validation_error": str(exc), "latency_ms": round((time.time() - start) * 1000, 2)}

@app.post("/summarize")
def summarize(req: SummarizeRequest):
    start = time.time()
    try:
        result = summarize_with_llm(question=req.question, tool_name=req.tool_name, tool_args=req.tool_args, tool_output=req.tool_output, user_role=req.user_role)
        return clean_json_value({
            "status": "success",
            "final_answer": result["answer"],
            "used_llm": result["used_llm"],
            "latency_ms": round((time.time() - start) * 1000, 2)
        })
    except Exception as exc:
        LOGGER.exception("SUMMARY_ROUTE_FAILED | %s", str(exc))
        return clean_json_value({"status": "error", "final_answer": str(exc), "latency_ms": round((time.time() - start) * 1000, 2)})


FLOW_1_CASES = [
    "show profile for BRID 75b4ae75-b",
    "top risky employees",
    "find employees in Pune",
    "predict no action probability for BRID 75b4ae75-b",
    "predict risk for recent population limit 100",
    "who clicked in March 2026 simulation"
]

FLOW_2_CASES = [
    "percentage of people clicked on January phishing mail",
    "city with most clickers",
    "predict for city Pune department Cyber subject password reset campaign January 2026",
    "how can BRID 75b4ae75-b improve",
    "recommend training for risky department",
    "what should we do to reduce phishing clicks",
    "show table schema"
]

SUMMARIZE_CASES = [
    {
        "question": "department wise click rate in 2026",
        "tool_name": "run_analytics",
        "tool_output": {
            "status": "success",
            "ui_summary": "Analysis completed on 45,000 rows. Highest-risk group is 'Cyber' with click rate 12.5% and report rate 60.1%.",
            "rows_analyzed": 45000,
            "highest_risk_group": {"department": "Cyber", "click_rate_percent": 12.5, "report_rate_percent": 60.1}
        }
    },
    {
        "question": "predict risk for a user from Pune",
        "tool_name": "predict_risk",
        "tool_output": {
            "status": "success",
            "prediction": "Clicked Link",
            "probabilities_labeled": {"No Action": 0.15, "Clicked Link": 0.70, "Reported": 0.15},
            "ui_summary": "Prediction completed: Clicked Link. Clicked: 70.0%, Reported: 15.0%, No Action: 15.0%."
        }
    },
    {
        "question": "how can this employee improve",
        "tool_name": "recommend_actions",
        "tool_output": {
            "status": "success",
            "ui_summary": "Recommendations generated. Suggested action: Focus on verifying login pages before entering credentials.",
            "recommended_actions": ["Focus on verifying login pages before entering credentials.", "Check sender carefully."],
            "focus_areas": ["security_credential"]
        }
    }
]


def run_router_tests() -> None:
    print("\n" + "=" * 80)
    print("🚀 STARTING FLOW 1 (Single Tool Selection - Top 1)")
    print("=" * 80)
    for idx, question in enumerate(FLOW_1_CASES, start=1):
        print(f"\n--- F1 Test {idx}: {question} ---")
        try:
            result = select_tool(SelectToolRequest(question=question, user_role="admin", top_k=1))
            output = {
                "selected_tool": result.get("selected_tool"),
                "mode": result.get("mode"),
                "args": result.get("args"),
                "validated": result.get("validated"),
                "validation_error": result.get("validation_error")
            }
            print(json.dumps(clean_json_value(output), indent=2))
        except Exception as e:
            print(f"Error: {e}")

    print("\n" + "=" * 80)
    print("🚀 STARTING FLOW 2 (Batch Tool Selection - Multi-Tool Threshold)")
    print("=" * 80)
    for idx, question in enumerate(FLOW_2_CASES, start=1):
        print(f"\n--- F2 Test {idx}: {question} ---")
        try:
            result = select_tool_batch(SelectToolBatchRequest(question=question, user_role="admin", top_k=3))
            print(json.dumps(clean_json_value(result.get("selected_tools", [])), indent=2))
        except Exception as e:
            print(f"Error: {e}")

    print("\n" + "=" * 80)
    print("🚀 STARTING SUMMARIZE TESTS (Post-Execution Generation)")
    print("=" * 80)
    for idx, case in enumerate(SUMMARIZE_CASES, start=1):
        print(f"\n--- Sum Test {idx}: {case['question']} ---")
        try:
            req = SummarizeRequest(question=case["question"], tool_name=case["tool_name"], tool_output=case["tool_output"])
            result = summarize(req)
            print(f"Final Answer: {result.get('final_answer')}")
        except Exception as e:
            print(f"Error: {e}")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    import uvicorn
    if "--test-router" in sys.argv:
        try:
            load_local_llm()
        except Exception:
            pass
        run_router_tests()
        sys.exit(0)

    host = os.getenv("LLM_SERVER_HOST", "127.0.0.1")
    port = int(os.getenv("LLM_SERVER_PORT", "8001"))
    try:
        load_local_llm()
    except Exception as e:
        LOGGER.exception(f"INITIAL_LOCAL_LLM_LOAD_FAILED | {str(e)}")

    LOGGER.info(f"LLM_SERVER_START | http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
Project Description

# Phishing MCP Project — Roadmap

## Project Summary

This project is a **phishing simulation analytics platform** built on MCP (Model Context Protocol). It defines clear responsibilities for each file in the system, with **two roles**: `admin` and `user`.

- **Admin** — can see everything (BRID, SSO ID, user name, last name, etc.)
- **User** — everything sensitive is masked

### High-level file map

| File | Responsibility |
|---|---|
| `Config.py` | Processes raw data and returns it in a format the ML model can predict on; defines the major functions and their working |
| `phishing_pandas_analytics.py` | Finds all analysis over the data and stores it in a JSON format that can answer any kind of query (e.g. "Who is the riskiest user for a department?", "Who is the riskiest user for email-related security?", "How can this user improve, keyed by BRID ID?") — output JSON should be structured to give the LLM as much useful context as possible |
| `phishing_mcp_server.py` | Supports only **8 tools** and nothing else. Each tool has a defined flow. Model output should be structured dynamically so it's both human-understandable and summarizable by the LLM |
| `llm_server.py` | Routing/orchestration architecture (see below) |
| `phishing_streamlit_ui.py` | Shows tool output, AI summary, and the full trace path (LLM, tool selection, semantic search tool outputs) in a dropdown |
| `phishing_streamlit_ui_noLLM.py` | A basic, deterministic tool-utility UI with no LLM in the loop |

---

## 4. `llm_server.py` — Architecture

```
User question
      ↓
Hard system rule check (health, cache, system info)
      ↓
Semantic top-1 tool match
      ↓
LLM receives:
   - question
   - user_role
   - top 3 tools only
   - examples for those tools
   - argument schemas
      ↓
LLM call: tool_name + arguments
      ↓
Validate
      ↓
Enforce UI user_role
      ↓
Return
```

---

## 1. `Config.py`

**Purpose:** Central configuration and utility file used by the pipeline for preprocessing, feature engineering, model classification, model management, and improvement-recommendation generation.

### Key Responsibilities

- **Configuration Management**
  - Loads environment variables and model paths
  - Defines columns to drop and feature settings
- **Data Cleaning & Utilities**
  - Text normalization, null handling, target mapping
- **Semantic AI Features**
  - Loads transformer embedding model
  - Generates embeddings
- **Feature Engineering** — creates:
  - Tenure features
  - Time-based features
  - Fraud risk features — HUGS, Grade, COO Area, City Zone features
  - Historical behaviour features
- **Categorical Encoding**
  - One-hot encoding
  - Feature alignment for training/inference consistency
- **Behaviour Analytics**
  - Calculates historical click, report, and no-action rates
- **Model Utilities**
  - Save and load trained models
- **Risk & Recommendation Engine**
  - Identifies top phishing risk drivers
  - Generates personalised improvement guidance
  - Generates population-level recommendations

### Main Functions

```text
apply_feature_engineering()     -> Complete feature engineering pipeline
ClassifyEmail()                 -> Email semantic classification
ExtractBusinessFeatures()       -> Business semantic classification
CalculateHistoricalBehavior()   -> User behaviour history features
build_improvement_guidance()    -> User-level recommendations
build_population_guidance()     -> Group-level recommendations
```

### In One Line

`Config.py` acts as the project's feature engineering, semantic classification, preprocessing, encoding, model utility, and recommendation-generation engine.

> This file does **not** have one single JSON output format — it returns different structures depending on the function.

### Main JSON Outputs

**1. `classify_semantic_text()`**

```json
{
  "security_credential_similarity": 0.89,
  "financial_similarity": 0.12,
  "urgency_pressure": 0.76,
  "authority_trust_similarity": 0.34,
  "link_attachment_similarity": 0.58
}
```

**2. `build_improvement_guidance()`** (Most Important)

```json
{
  "predicted_label": "Clicked Link",
  "click_probability": 0.72,
  "reported_probability": 0.11,
  "no_action_probability": 0.17,
  "department": "technology",
  "businessarea1": "cyber security",
  "template_name": "password reset",
  "template_subject": "reset your password immediately",
  "top_risk_drivers": [
    { "driver": "security_credential", "score": 0.94 },
    { "driver": "urgency_pressure", "score": 0.88 },
    { "driver": "link_attachment", "score": 0.72 }
  ],
  "focus_areas": [
    "security_credential",
    "urgency_pressure",
    "low_reporting"
  ],
  "recommended_actions": [
    "Focus on verifying login pages before entering credentials.",
    "Check sender, URL, and domain carefully before responding to account verification emails.",
    "Pause before acting on urgent, threatening, or deadline-driven emails."
  ]
}
```

**3. `build_population_guidance()`**

```json
{
  "summary": "Improvement guidance generated from predicted click-risk drivers.",
  "top_focus_areas": {
    "security_credential": 42,
    "urgency_pressure": 35,
    "low_reporting": 20
  },
  "recommended_actions": [
    "Focus on verifying login pages before entering credentials.",
    "Pause before acting on urgent emails."
  ]
}
```

**4. `apply_feature_engineering()`** — returns a **DataFrame**, not JSON. Output columns look like:

```json
{
  "TimeInCompany_years": 5.2,
  "sent_hour": 14,
  "sent_month": 6,
  "security_credential_similarity": 0.91,
  "financial_similarity": 0.12,
  "technology_security_similarity": 0.88,
  "corporate_grade_encoded": 4,
  "coo_area_encoded": 1,
  "city_zone_encoded": 0,
  "past_click_rate_ema": 0.45,
  "past_report_rate_ema": 0.20
}
```

### In Short

`Config.py` generates its most useful output via `build_improvement_guidance()`, because it produces a **Prediction → Risk Drivers → Focus Areas → Recommended Actions** structure — which is likely what the MCP tools and Streamlit UI will eventually display.

---

## 2. `phishing_pandas_analytics.py`

**Responsibilities:** This file is the core analytics engine for the phishing simulation copilot. It takes raw phishing simulation data, cleans it, enriches it, runs grouped user risk profiles, and returns privacy-safe JSON outputs for the MCP server/UI.

### Main Responsibilities

- **Data preparation**
  - Converts phishing event types into target labels: `0 = No Action`, `1 = Clicked`, `2 = Reported`
  - Adds time fields like year, month number, month name, and month-year
  - Resolves flexible user terms like `employee`, `user`, `brid`, `department`, `campaign`, etc. to actual dataframe columns
- **Filtering and grouping**
  - Supports filters by year, department, campaign, template, subject, BRID, business area, grade, etc.
- **Core phishing metrics** — calculates:
  - total events, clicked count, reported count, no-action count
  - click rate, report rate, no-action rate
- **Risk and insight generation**
  - Finds highest-risk, lowest-risk, and highest-reporting groups
  - Detects tied top groups
  - Generates basic insights and risk levels
  - Creates recommended actions based on click/report/no-action behaviour
- **Theme and training recommendation logic**
  - Detects phishing themes from similarity scores or text fallback
  - Maps themes to training recommendations
  - Supports themes like: Credential Theft, Financial Fraud, Urgent Action, Authority Impersonation, Attachment Risk
- **Employee-level analytics**
  - Builds cached employee profiles
  - Stores top clicked subjects, templates, campaigns, and themes
  - Tracks each employee's historical click/report/no-action behaviour
- **Privacy controls**
  - Hides BRID/personal identifiers for non-admin users in bulk
  - Allows BRID-specific insight when a specific BRID is requested
  - Shows full profile data only for admin users
- **Prediction support**
  - Identifies high-risk users from prediction probability columns such as `prob_clicked_link`
  - Returns high-risk count, training focus, campaign recommendations, and top risky subjects/templates/campaigns
- **Main analysis entry point**
  - `run_analysis()` is the central function used by the server — handles cleaning, filtering, grouping, analytics execution, privacy sanitisation, and final JSON response creation
- **Built-in testing**
  - The `__main__` block creates sample phishing data and runs test cases for overall, city, department, BRID, campaign, template, subject, monthly/yearly trends, privacy checks, employee improvement profiles, and prediction output

### In One Line

`phishing_pandas_analytics.py` converts raw phishing simulation data into privacy-safe analytics, trend summaries, training recommendations, and prediction-based high-risk population insights. The file mainly returns **Python dict/list objects that are JSON-safe**, using `clean_json_value()`.

### Main Output Formats

**1. `run_analysis()`**

```json
{
  "status": "success",
  "filters_applied": {},
  "user_role": "admin",
  "analysis_type": "city_performance",
  "rows_analyzed": 100,
  "dimensions": ["city"],
  "metadata": {},
  "metric_explanation": {
    "total_events": "Total phishing simulation events analysed after filters.",
    "clicked_count": "Number of events where the user clicked the phishing link.",
    "reported_count": "Number of events reported as phishing.",
    "no_action_count": "Number of events where no click or report action was recorded.",
    "click_rate_percent": "Percentage of events with no user action.",
    "report_rate_percent": "Percentage of events reported as phishing.",
    "no_action_rate_percent": "Percentage of events with no user action.",
    "risk_score": "Higher means more click behaviour and lower reporting behaviour."
  },
  "summary": [
    {
      "city": "pune",
      "total_events": 25,
      "clicked_count": 10,
      "reported_count": 5,
      "no_action_count": 10,
      "click_rate_percent": 40.0,
      "report_rate_percent": 20.0,
      "no_action_rate_percent": 40.0,
      "risk_score": 20.0,
      "sample_size_warning": ""
    }
  ],
  "highest_risk_group": {},
  "highest_reporting_group": {},
  "lowest_risk_group": {},
  "highest_risk_groups": [],
  "top_clicked_templates": [],
  "top_clicked_campaigns": [],
  "insights": [],
  "risk_analysis": [],
  "recommended_actions": []
}
```

**2. `get_employee_profile()`**

```json
{
  "brid": "75b4ae75-b",
  "department": "tax",
  "city": "nottingham",
  "total_events": 5,
  "clicked_count": 3,
  "reported_count": 1,
  "no_action_count": 1,
  "click_rate_percent": 60.0,
  "report_rate_percent": 20.0,
  "no_action_rate_percent": 20.0,
  "top_clicked_themes": [],
  "top_clicked_templates": [],
  "top_risk_campaigns": [],
  "risk_score": 65.0,
  "sample_size_warning": ""
}
```

**3. `get_employee_improvement_profile()`**

```json
{
  "brid": "75b4ae75-b",
  "department": "tax",
  "city": "nottingham",
  "total_events": 5,
  "clicked_count": 3,
  "reported_count": 1,
  "no_action_count": 1,
  "click_rate_percent": 60.0,
  "report_rate_percent": 20.0,
  "no_action_rate_percent": 20.0,
  "risk_score": 65.0,
  "top_risk_subjects": [],
  "top_risk_themes": ["Credential Theft"],
  "focus_areas": [],
  "recommended_actions": [""],
  "behaviour_summary": {
    "interpretation": "This is based on historical phishing security behaviour, not employee performance judgement.",
    "risk_score_formula": "click_rate_percent + clicked_count*2 - reported_count"
  }
}
```

**4. `get_top_risky_employees()`**

```json
[
  {
    "department": "tax",
    "city": "nottingham",
    "total_events": 5,
    "clicked_count": 3,
    "reported_count": 1,
    "no_action_count": 1,
    "click_rate_percent": 60.0,
    "report_rate_percent": 20.0,
    "risk_score": 65.0,
    "top_risk_themes": [],
    "top_clicked_subjects": [],
    "personal_information_hidden": true
  }
]
```

> For non-admin users, BRID/personal fields are hidden in bulk output.

**5. `predict_high_risk_population()`**

```json
{
  "status": "success",
  "high_risk_count": 2,
  "threshold": 0.6,
  "probability_column": "prob_clicked_link",
  "training_focus": [],
  "campaign_recommendations": [
    {
      "priority": "High",
      "action": "Run focused phishing simulations and awareness nudges for the identified high-risk population."
    }
  ],
  "top_risk_subjects": [],
  "top_risk_templates": [],
  "top_risk_campaigns": []
}
```

### In One Line

The file returns structured JSON with this general pattern:

```json
{
  "status": "success/error",
  "filters_applied": {},
  "analysis_type": "...",
  "rows_analyzed": 0,
  "summary": [],
  "insights": [],
  "risk_analysis": [],
  "training_recommendations": [],
  "recommended_actions": []
}
```

---

## 3. `phishing_mcp_server.py`

This file is the **backend MCP server** that exposes phishing analytics and prediction tools to the UI/LLM.

### Main Structure

**1. Setup + config**
- Loads `.env`, logging, constants, SQL config, model paths
- Creates MCP server: `FastMCP("PhishingAnalytics")`

**2. Global cache**
- Stores: raw phishing event data, employee profile cache, BRID lookup cache, prediction cache, cache status/health
- Functions:
  - `refresh_cache_internal()` — reloads data from DB
  - `clear_cache_internal()` — clears cache
  - `ensure_cache()` — auto-loads/refreshes cache
  - `execute_query()` — runs SQL queries

**3. Database + model loading**
- `get_sql_engine()` — connects to SQL Server
- `load_prediction_model()` — loads ML model
- `load_feature_columns()` — loads trained feature columns
- `load_global_dataset()` — loads latest DB rows

**4. Prediction backend**
- `validate_prediction_payload()` — checks missing fields
- `predict_dataframe()` — runs ML prediction
- `run_prediction_by_brid()` — predicts using the latest row for a BRID
- `run_prediction_by_payload()` — predicts from user-provided fields

**5. Analytics backend**
- `run_analytics()` — main MCP analytics tool
- Calls `run_analysis()` from `phishing_pandas_analytics.py`
- Handles city, department, campaign, template, subject, month/year trends, etc.

**6. Employee backend**
- `employee_lookup()` — fetches employee profile, top risky employees, or filters employees
- Uses employee profile cache
- Applies role/admin privacy logic

**7. Recommendation backend**
- `recommend_actions()` — gives improvement guidance
- Supports employee-specific recommendations, group recommendations, and overall/admin privacy controls

**8. Simulation backend**
- `simulation_users()` — finds users by campaign/month/year/event type (e.g. "clicked users in the January campaign")
- Hides BRIDs/PII for non-admin users

**9. Privacy layer**
- `is_admin()`, `sanitize_prediction_records()`, `sanitize_ui_response()`
- Creates short deterministic summaries for the Streamlit UI

**10. UI response layer**
- `build_ui_highlights()` — ensures PII/BRID is shown only for admin role

**11. System tools**
- `system_info()` — health, schema, features, environment checks
- `validate_environment()` — checks required env/config

**12. Test runner**
- `run_server_tests()` — used with:
  ```bash
  python phishing_mcp_server.py --test
  ```
- Tests: `employee_lookup`, `predict_risk`, `recommend_actions`, `simulation_users`, `cache_control`, `cache_status`, `system_info`, `validate_environment`

### One-Line Summary

This file is the backend MCP server that connects DB + cache + analytics + ML prediction + privacy controls, and exposes them as MCP tools for the phishing analytics copilot.

### Main JSON Outputs Per Tool

**1. `run_analytics` tool**

```json
{
  "status": "success",
  "analysis_type": "city_performance",
  "dimensions": ["city"],
  "rows_analyzed": 50000,
  "summary": {},
  "highest_risk_group": {},
  "highest_reporting_group": {},
  "lowest_risk_group": {},
  "top_clicked_subjects": [],
  "top_clicked_templates": [],
  "top_clicked_campaigns": [],
  "training_recommendations": [],
  "recommended_actions": [],
  "risk_analysis": {},
  "llm_context": {},
  "ui_summary": "Analysis completed...",
  "ui_highlights": [],
  "ui_response_type": "deterministic_mcp_summary"
}
```

**2. `employee_lookup` tool**

Profile mode:

```json
{
  "status": "success",
  "mode": "profile",
  "employee_profile": "Employee historical phishing profile was found.",
  "ui_highlights": [],
  "llm_context": {},
  "deterministic_mcp_summary": ""
}
```

Top-risky mode:

```json
{
  "status": "success",
  "mode": "top_risky",
  "count": 10,
  "employees": [],
  "filters": { "city": "pune", "department": "security" },
  "llm_context": {}
}
```

**3. `predict_risk` tool**

By BRID:

```json
{
  "predict_risk": "Clicked Link",
  "probabilities_labeled": [
    { "No Action": 0.2, "Clicked Link": 0.7, "Reported": 0.1 }
  ],
  "user_role": "user",
  "pii_exposed": false,
  "lookup_context": {
    "lookup_type": "BRID",
    "brid": "hidden_user_mode",
    "note": "Prediction uses latest available row..."
  },
  "llm_context": {}
}
```

From payload:

```json
{
  "status": "success",
  "prediction": "Clicked Link",
  "probabilities_labeled": [
    { "No Action": 0.2, "Clicked Link": 0.7, "Reported": 0.1 }
  ],
  "argument_quality": "medium",
  "input_validation": {},
  "user_guidance": { "summary": "Prediction can run...", "empty_fields": [] },
  "llm_context": {}
}
```

Recent population:

```json
{
  "status": "success",
  "rows": 100,
  "rows_predicted": 100,
  "prediction_distribution": { "No Action": 60, "Clicked Link": 25, "Reported": 15 },
  "llm_context": {}
}
```

High-risk population:

```json
{
  "status": "success",
  "input_quality": "medium",
  "completeness_percent": 55.0,
  "missing_fields_to_improve_prediction": [],
  "top_missing_fields_to_improve_prediction": [],
  "user_role": "user",
  "pii_exposed": false,
  "population_size": 500,
  "high_risk_count": 40,
  "high_risk_percentage": 8.0,
  "avg_click_probability": 0.34,
  "threshold": 0.6,
  "analytics_high_risk_summary": {},
  "prediction_cache_key": "...",
  "llm_context": {}
}
```

**4. `recommend_actions` tool**

Employee improvement:

```json
{
  "status": "success",
  "mode": "group_recommendations",
  "analysis_type": "department_performance",
  "dimensions": ["department"],
  "rows_analyzed": 50000,
  "top_clicked_subjects": [],
  "top_clicked_templates": [],
  "top_clicked_campaigns": [],
  "recommended_actions": [],
  "risk_analysis": {},
  "llm_context": {}
}
```

Overall recommendations:

```json
{
  "status": "success",
  "mode": "overall_recommendations",
  "rows_analyzed": 50000,
  "training_recommendations": [],
  "recommended_actions": [],
  "risk_analysis": {},
  "llm_context": {}
}
```

**5. `simulation_users` tool**

```json
{
  "status": "success",
  "campaignname": "January 2026 Campaign",
  "campaign_month": "January",
  "campaign_year": "2026",
  "campaign_search_type": "month_year",
  "event_type": "clicked link",
  "user_count": 100,
  "total_unique_brids": 0,
  "brid_ids": ["brid1", "brid2"],
  "pii_exposed": true,
  "users": [],
  "llm_context": {}
}
```

For admin, individual user records look like:

```json
{
  "campaignname": "January 2026 Campaign",
  "eventtype": "clicked link",
  "senttimestamp": "2026-01-12",
  "city": "Pune",
  "usertags-BRID": "brid1",
  "useremailaddress": "user@example.com",
  "userfirstname": "ABC",
  "userlastname": "XYZ"
}
```

**6. `cache_control` tool**

Refresh:

```json
{
  "status": "success",
  "action": "refresh",
  "loaded": true,
  "last_refresh_time": "2026-07-08 15:10:00",
  "source": "database",
  "validation": {},
  "total_rows": 50000,
  "employee_profiles": 12000,
  "llm_context": {}
}
```

Clear:

```json
{
  "status": "success",
  "action": "clear",
  "loaded": false,
  "last_refresh_time": null,
  "total_rows": 0,
  "employee_profiles": 0,
  "llm_context": {}
}
```

**7. `cache_status` tool**

```json
{
  "status": "success",
  "loaded": true,
  "last_refresh_time": "2026-07-08 15:10:00",
  "source": "database",
  "cache_enabled": true,
  "total_rows": 50000,
  "employee_profiles": 12000,
  "profile_lookup_count": 12000,
  "global_rows_limit": 50000,
  "statistics": {
    "rows": 50000,
    "employee_profiles": 12000,
    "departments": 50,
    "campaigns": 10,
    "templates": 30,
    "cities": 20
  },
  "llm_context": {}
}
```

**8. `system_info` tool**

Health:

```json
{
  "status": "healthy",
  "mode": "health",
  "environment": {},
  "model_loaded": false,
  "model_available": true,
  "model_path": "model.pkl",
  "feature_columns_available": true,
  "database_connected": true,
  "cache_loaded": true,
  "cache_rows": 50000,
  "employee_profiles": 12000,
  "llm_context": {}
}
```

Schema:

```json
{
  "status": "success",
  "mode": "schema",
  "columns": [],
  "column_count": 50,
  "llm_context": {}
}
```

Features:

```json
{
  "status": "success",
  "mode": "features",
  "feature_column_count": 120,
  "feature_columns_path": "feature_columns.pkl",
  "cache_config": {},
  "enable_cache_auto_refresh_minutes": true,
  "global_rows_limit": 50000,
  "llm_context": {}
}
```

Environment:

```json
{
  "status": "success",
  "mode": "environment",
  "environment": {},
  "default_population_limit": 1000,
  "high_risk_threshold": 0.6,
  "brid_column": "usertags-BRID",
  "llm_context": {}
}
```

### Common Error Format

```json
{
  "status": "error",
  "message": "Something went wrong",
  "llm_context": {}
}
```

Some responses also include:

```json
{
  "status": "success/error",
  "ui_summary": "...",
  "ui_highlights": [],
  "main_result_data": {},
  "llm_context": {}
}
```

### In Short

The MCP server mostly returns JSON with this pattern:

```json
{
  "status": "success/error",
  "ui_summary": "short deterministic summary",
  "ui_highlights": [],
  "main_result_data": {},
  "llm_context": {}
}
```

---

## 5. `phishing_streamlit_ui.py`

**Very concise summary:** This file is the **Streamlit frontend** for the Phishing Simulation Analytics Copilot. Its main responsibility is to take a user question, route it to the correct backend tool, execute the tool, get a grounded summary, and show the result with an optional debugging trace.

### Main Responsibilities

- Builds the Streamlit UI with sidebar, header, chat interface, quick prompts, and custom styling
- Maintains session state for chat history, user role, pending questions, and debug visibility
- Sends user questions to the LLM server's `/select_tool` endpoint to choose the best tool and generate arguments
- Calls the LLM server's `/summarize` endpoint to convert a tool execution result into a concise final answer
- Provides fallback answers via:
  - direct Python import from `phishing_mcp_server.py`, or
  - MCP stdio mode
- Supports tools:
  - `run_analytics`, `employee_lookup`, `predict_risk`, `recommend_actions`, `simulation_users`, `cache_control`, `cache_status`, `system_info`
- Captures detailed execution logs for routing, tool execution, latency, and final output
- Shows debug trace tabs for: tool selection, tool arguments, tool output, LLM summary, execution logs, full trace JSON
- Supports app/admin access controls, where visibility depends on backend policy

### In One Line

`phishing_streamlit_ui.py` is the chat-based UI layer that connects the user, the LLM routing server, MCP/direct analytics tools, summaries, cache controls, and debug tracing into one working phishing analytics copilot.

---

## 6. `phishing_streamlit_ui_noLLM.py`

**Very concise summary:** This file is the **No-LLM version Streamlit UI** for the Phishing Simulation Analytics Copilot. It lets users ask phishing analytics questions, selects the right backend tool using **local rule-based + top-1 cosine similarity routing**, builds deterministic arguments, executes the selected MCP/server tool, and shows a clean deterministic response with optional debug trace panels.

### Main Responsibilities

- Builds the Streamlit web UI with sidebar, chat interface, quick prompts, header cards, and debug toggle
- Provides a hard-coded list of supported backend tools:
  - `run_analytics`, `employee_lookup`, `predict_risk`, `recommend_actions`, `simulation_users`, `cache_control`, `cache_status`, `system_info`
- Routes user questions without using an LLM:
  - hard rules for cache/system queries
  - token cosine similarity over tool descriptions and keywords
  - keyword boosting
- Extracts simple entities from user text: BRID, month/year, limit/top N, event type, simple payload fields
- Builds deterministic payload fields before execution (e.g. BRID for profile/prediction/recommendation queries)
- Validates required inputs before execution
- Executes tools either:
  - directly by importing `phishing_mcp_server.py`, or
  - through MCP stdio, depending on `MCP_EXECUTION_MODE`
- Converts raw backend output into readable deterministic UI summaries
- Shows execution trace including: routing, tool selected, ranked candidates, raw output, execution steps, downloadable trace JSON
- Enforces role-aware UI mode with `user` and `admin`, while relying on backend policy for actual identifier exposure

### Function Groups

**Utility / formatting**
- `now_time`, `clean_json_value`, `compact_preview`, `add_step`, `percent`

**Routing helpers**
- `normalize_text`, `tokenize`, `cosine_similarity_tokens`, `extract_threshold`, `hard_rule_router`, `top1_similarity_tool`, `select_tool_no_llm`

**Entity extraction**
- `extract_brid`, `extract_year`, `extract_month`, `extract_limit`, `extract_event_type`, `extract_simple_payload`

**Argument building / validation**
- `infer_analysis_from_question`, `build_tool_args`, `validate_selection`

**Tool execution**
- `execute_tool_direct`, `execute_tool_mcp_stdio_async`, `execute_tool_mcp_stdio`, `execute_selected_tool`

**Answer generation**
- `summarize_deterministic`, `run_full_pipeline`

**Streamlit UI**
- `init_state`, `queue_question`, `render_sidebar`, `render_header`, `render_chat`, `render_trace`, `render_steps`, `render_clear_tool_output`, `handle_question`, `main`

### One-Line Summary

`phishing_streamlit_ui_noLLM.py` is a fast, deterministic Streamlit frontend that routes phishing-related user questions to backend analytics, prediction, recommendation, cache, and system tools — without using any LLM for tool selection or response generation.

---

## Reference: Employee Improvement Recommendation Output

Suggested structure so both the MCP server and LLM can consume it easily:

```json
{
  "brid": "75b4ae75-b",
  "employee_profile": {
    "department": "COO",
    "city": "Pune",
    "risk_level": "High",
    "risk_score": 78.5
  },
  "behaviour_summary": {
    "total_events": 25,
    "clicked_count": 12,
    "reported_count": 2,
    "no_action_count": 11,
    "click_rate_percent": 48.0,
    "report_rate_percent": 8.0,
    "no_action_rate_percent": 44.0
  },
  "risk_drivers": [
    { "factor": "Security themed emails", "clicks": 6, "risk_weight": "High" },
    { "factor": "Finance themed emails", "clicks": 4, "risk_weight": "Medium" }
  ],
  "theme_analysis": [
    { "theme": "Security Update", "events": 8, "clicked": 6, "click_rate_percent": 75.0 },
    { "theme": "Finance", "events": 5, "clicked": 4, "click_rate_percent": 80.0 }
  ],
  "training_analysis": [
    { "training": "Phishing Awareness", "completed": true, "days_since_completion": 120 }
  ],
  "recommendations": [
    {
      "priority": 1,
      "category": "Training",
      "action": "Retake Security Awareness training",
      "reason": "High click rate on security themed phishing simulations"
    },
    {
      "priority": 2,
      "category": "Behaviour",
      "action": "Verify sender domains before opening emails",
      "reason": "Multiple clicks on impersonation campaigns"
    }
  ],
  "prediction_insights": {
    "predicted_click_probability": 0.74,
    "predicted_report_probability": 0.11,
    "predicted_no_action_probability": 0.15
  },
  "executive_summary": "User is a high-risk employee primarily vulnerable to security and finance themed phishing emails. Immediate refresher training is recommended."
}
```

> For this project, **`theme_analysis`**, **`risk_drivers`**, and **`recommendations`** should be treated as mandatory fields — these are what actually differentiate the solution from a simple risk-prediction model. The LLM can then generate a concise summary directly from this JSON.


git add COMMANDS.txt .requirements.txt phu.py .env con.py dman.py lser.py phana.py phnoL.py phser.py
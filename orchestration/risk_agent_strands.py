"""
Risk Agent as a Strands LLM agent: gathers weather/news from API + web search (union),
fuses conflicting data, then calls calculate_risk_score_tool.
Exposed to the orchestrator via risk_agent_tool(features_json, risk_queries_json).
"""

import json
import os
import re
from pathlib import Path
from typing import Optional

_CODE_DIR = Path(__file__).resolve().parent.parent
if os.getcwd() != str(_CODE_DIR):
    os.chdir(_CODE_DIR)

RISK_AGENT_PROMPT = """You are a supply chain risk assessment agent. You reason about delivery risk by gathering data from multiple sources and fusing them.

Your tools:
1. **weather_api_tool(city)** – Get weather from OpenWeatherMap API for a city. Returns weather_condition (clear, rainy, cold, hot, foggy, stormy).
2. **news_api_tool(query)** – Search news (NewsAPI) for disruption context. Input a query like "supply chain disruption India" or "port strike Mumbai". Returns list of articles.
3. **web_search_tool(queries)** – Run web search. Input: single query string OR JSON array of query strings (e.g. ["Mumbai weather today", "Mumbai port strike 2025"]). Returns snippets per query.
4. **calculate_risk_score_tool(package_type, weather_condition, risk_factors_json, route_dict_json)** – Compute risk using Module 07. weather_condition is your fused canonical weather; risk_factors_json is a JSON array of strings (risk factors you identified); route_dict_json is a JSON object with at least delivery_partner, package_type, vehicle_type, delivery_mode, region, weather_condition, distance_km, package_weight_kg, delivery_rating.

Workflow:
1. **Gather from BOTH APIs and web search (union)** – Do not use web search only when API fails. Always call weather_api_tool for origin and destination, news_api_tool for disruption (e.g. "supply chain disruption India"), and web_search_tool with the provided risk_queries list. This gives you weather from API + web, and news from API + web.
2. **Fuse and check consistency** – Compare weather from API vs web search: are they consistent? Compare news from API vs web: any conflicts or extra events (e.g. port strike) only in web? Decide the canonical weather_condition and the final list of risk_factors (union of API news + web findings). Do not treat API failure as critical by itself; use whatever data you have.
3. **Call calculate_risk_score_tool** – Pass package_type, your fused weather_condition, risk_factors as JSON array, and a minimal route_dict (include origin/destination from features, weather_condition, package_type, distance_km if known, package_weight_kg if known; use defaults: delivery_partner "delhivery", vehicle_type "van", delivery_mode "express", region "west", delivery_rating 4).
4. **Return** – Reply with the risk assessment as JSON: risk_level, risk_score, risk_factors, delay_probability, recommended_buffer_days, warnings, alert_required. Your final message must be valid JSON so the orchestrator can parse it."""


_risk_agent = None


def get_risk_agent(region_name: Optional[str] = None, model_id: Optional[str] = None):
    """Create or return the Risk Strands agent (LLM) with tools."""
    global _risk_agent
    if _risk_agent is not None:
        return _risk_agent
    try:
        from strands import Agent
        from strands.models import BedrockModel
    except ImportError as e:
        raise ImportError("Strands and Bedrock required for Risk Agent. pip install strands-agents boto3") from e
    from orchestration.agent_tools import get_risk_agent_tools

    region_name = region_name or os.environ.get("AWS_REGION", "us-east-1")
    model_id = model_id or os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-3-7-sonnet-20250219-v1:0")
    bedrock_model = BedrockModel(
        model_id=model_id,
        region_name=region_name,
        temperature=0.0,
    )
    _risk_agent = Agent(
        model=bedrock_model,
        system_prompt=RISK_AGENT_PROMPT,
        tools=get_risk_agent_tools(),
    )
    return _risk_agent


def run_risk_agent(features_json: str, risk_queries_json: str, region_name: Optional[str] = None, model_id: Optional[str] = None) -> str:
    """
    Invoke the Risk Strands agent with features and adaptive risk queries.
    Returns JSON string: risk_level, risk_factors, delay_probability, recommended_buffer_days, etc.
    """
    agent = get_risk_agent(region_name=region_name, model_id=model_id)
    try:
        features = json.loads(features_json)
    except json.JSONDecodeError:
        features = {}
    try:
        queries = json.loads(risk_queries_json) if risk_queries_json.strip() else []
    except json.JSONDecodeError:
        queries = [risk_queries_json] if risk_queries_json.strip() else []
    if not isinstance(queries, list):
        queries = [queries]
    origin = str(features.get("origin") or "mumbai").lower()
    destination = str(features.get("destination") or "delhi").lower()
    package_type = (features.get("package_type") or "clothing").lower()
    message = (
        f"Assess risk for this shipment.\n\n"
        f"Features: origin={origin}, destination={destination}, package_type={package_type}. "
        f"Full features: {json.dumps(features)}\n\n"
        f"Use these search queries to gather data (call web_search_tool with this list): {json.dumps(queries)}\n\n"
        f"Call weather_api_tool for '{origin}' and '{destination}', news_api_tool for disruption, "
        f"web_search_tool with the queries above. Then fuse API + web results and call calculate_risk_score_tool. "
        f"Return the risk assessment as JSON."
    )
    response = agent(message)
    text = str(response)
    # Return parseable JSON for orchestrator/sourcing_agent_tool
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text)
    if m:
        try:
            obj = json.loads(m.group(0))
            if "risk_level" in obj or "risk_factors" in obj:
                return json.dumps(obj)
        except json.JSONDecodeError:
            pass
    return text

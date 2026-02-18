"""
Low-level tools for Risk and Sourcing Strands agents.
Used by Risk Agent (weather_api, news_api, web_search, calculate_risk_score) and
Sourcing Agent (distance_api, routes_lookup, web_search, get_carrier_options).
"""

import json
import os
from pathlib import Path
from typing import Optional

_CODE_DIR = Path(__file__).resolve().parent.parent
if os.getcwd() != str(_CODE_DIR):
    os.chdir(_CODE_DIR)

try:
    from strands import tool
except ImportError:
    tool = lambda f: f


def _get_orchestrator():
    from orchestration._orchestrator_instance import get_orchestrator
    return get_orchestrator()


def _json_safe(obj):
    import numpy as np
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# --- Risk Agent tools ---

@tool
def weather_api_tool(city: str) -> str:
    """Get current weather condition for a city from OpenWeatherMap API. Input: city name. Output: JSON with weather_condition (clear, rainy, cold, hot, foggy, stormy) or error."""
    try:
        from config.api_config import API_KEYS
        key = (API_KEYS or {}).get("openweathermap", "")
        if not key:
            return json.dumps({"weather_condition": None, "source": "api", "error": "no_api_key"})
        from data.apis.weather_api import get_weather
        condition = get_weather(city.strip(), key)
        return json.dumps({"weather_condition": condition, "source": "api", "city": city})
    except Exception as e:
        return json.dumps({"weather_condition": None, "source": "api", "error": str(e)})


@tool
def news_api_tool(query: str) -> str:
    """Search news for disruption/risk context (NewsAPI). Input: search query (e.g. 'supply chain disruption India'). Output: JSON list of articles (title, description, url) or error."""
    try:
        from config.api_config import API_KEYS
        key = (API_KEYS or {}).get("newsapi", "")
        if not key:
            return json.dumps({"articles": [], "source": "api", "error": "no_api_key"})
        from data.apis.news_api import search_news
        articles = search_news(query.strip(), key)
        out = [{"title": a.get("title"), "description": a.get("description"), "url": a.get("url")} for a in (articles or [])]
        return json.dumps({"articles": out, "source": "api"})
    except Exception as e:
        return json.dumps({"articles": [], "source": "api", "error": str(e)})


@tool
def web_search_tool(queries: str) -> str:
    """Run web search for one or more queries. Input: single query string OR JSON array of query strings. Output: JSON with results per query (snippet text)."""
    try:
        from tools.web_search import web_search as do_web_search
        if queries.strip().startswith("["):
            qlist = json.loads(queries)
        else:
            qlist = [queries.strip()]
        results = []
        for q in qlist:
            if not q:
                continue
            raw = do_web_search(q)
            results.append({"query": q, "snippet": raw or ""})
        return json.dumps({"results": results})
    except Exception as e:
        return json.dumps({"results": [], "error": str(e)})


@tool
def calculate_risk_score_tool(
    package_type: str,
    weather_condition: str,
    risk_factors_json: str,
    route_dict_json: str,
) -> str:
    """Compute risk score using Module 07 (Early-Warning). Input: package_type, weather_condition (fused), risk_factors (JSON array of strings), route_dict (JSON). Output: risk_level, risk_score, risk_factors, delay_probability, recommended_buffer_days."""
    orch = _get_orchestrator()
    ews = orch.ews
    try:
        route_dict = json.loads(route_dict_json) if route_dict_json else {}
        risk_factors = json.loads(risk_factors_json) if risk_factors_json else []
        if not isinstance(risk_factors, list):
            risk_factors = [risk_factors] if risk_factors else []
        # Ensure route_dict has required columns for ews
        route_dict.setdefault("weather_condition", weather_condition or "clear")
        route_dict.setdefault("package_type", package_type or "clothing")
        route_dict.setdefault("delivery_partner", "delhivery")
        route_dict.setdefault("vehicle_type", "van")
        route_dict.setdefault("delivery_mode", "express")
        route_dict.setdefault("region", "west")
        route_dict.setdefault("distance_km", 150)
        route_dict.setdefault("package_weight_kg", 25)
        route_dict.setdefault("delivery_rating", 4)
        delay_prob = ews.predict_delay_probability(route_dict)
        risk = ews.calculate_risk_score(package_type, delay_prob, route_dict)
        # Merge LLM-provided risk_factors with Module 07 factors
        merged_factors = list(risk.get("risk_factors", [])) + [f for f in risk_factors if f and f not in risk.get("risk_factors", [])]
        buffer_map = {"CRITICAL": 2, "HIGH": 1, "MEDIUM": 1, "LOW": 0}
        recommended_buffer_days = buffer_map.get(risk["risk_level"], 0)
        out = {
            "risk_level": risk["risk_level"],
            "risk_score": risk["risk_score"],
            "risk_factors": merged_factors,
            "warnings": merged_factors.copy(),
            "delay_probability": delay_prob,
            "recommended_buffer_days": recommended_buffer_days,
            "alert_required": risk.get("alert_required", False),
        }
        if risk.get("alert_required"):
            out["warnings"].append("Alert required: high disruption risk")
        return json.dumps(_json_safe(out))
    except Exception as e:
        return json.dumps({"risk_level": "MEDIUM", "risk_score": 2.0, "risk_factors": [str(e)], "delay_probability": 0.2, "recommended_buffer_days": 1, "error": str(e)})


# --- Sourcing Agent tools ---

@tool
def distance_api_tool(origin: str, destination: str) -> str:
    """Get road distance in km between two places (OpenRouteService API). Input: origin, destination city names. Output: JSON with distance_km and source."""
    try:
        from config.api_config import API_KEYS
        key = (API_KEYS or {}).get("openrouteservice", "")
        if not key:
            return json.dumps({"distance_km": None, "source": "api", "error": "no_api_key"})
        from data.apis.distance_api import get_distance_km
        d = get_distance_km(origin.strip(), destination.strip(), key)
        return json.dumps({"distance_km": d, "source": "openrouteservice"})
    except Exception as e:
        return json.dumps({"distance_km": None, "source": "api", "error": str(e)})


@tool
def routes_lookup_tool(origin: str, destination: str) -> str:
    """Get route info (distance_km, region, is_metro_to_metro) from local routes or cascade. Input: origin, destination. Output: JSON."""
    try:
        from tools.sourcing_tools import lookup_route
        info = lookup_route(origin.strip(), destination.strip())
        if info is None:
            return json.dumps({"distance_km": None, "region": "west", "is_metro_to_metro": False, "source": "none"})
        return json.dumps(_json_safe(info))
    except Exception as e:
        return json.dumps({"distance_km": None, "error": str(e)})


@tool
def get_carrier_options_tool(features_json: str, risk_assessment_json: str) -> str:
    """Get carrier options (cost, on_time_pct, carbon, route) for features and risk. Uses Python sourcing logic. Output: JSON list of carrier options."""
    orch = _get_orchestrator()
    try:
        from tools.extraction_tools import extract_features_from_dict
        features = json.loads(features_json)
        risk = json.loads(risk_assessment_json) if risk_assessment_json else {}
        features = extract_features_from_dict(features)
        options = orch.sourcing_service.get_carrier_options(features, risk)
        out = []
        for o in options:
            out.append({
                "carrier": o.get("carrier"),
                "carrier_code": o.get("carrier_code"),
                "predicted_cost": o.get("predicted_cost"),
                "predicted_on_time_pct": o.get("predicted_on_time_pct"),
                "total_carbon_gco2": o.get("total_carbon_gco2"),
                "meets_sla": o.get("meets_sla"),
                "route": o.get("route"),
            })
        return json.dumps(_json_safe(out))
    except Exception as e:
        return json.dumps({"error": str(e), "options": []})


def get_risk_agent_tools():
    """Tools passed to the Risk Strands Agent."""
    return [
        weather_api_tool,
        news_api_tool,
        web_search_tool,
        calculate_risk_score_tool,
    ]


def get_sourcing_agent_tools():
    """Tools passed to the Sourcing Strands Agent."""
    return [
        distance_api_tool,
        routes_lookup_tool,
        web_search_tool,
        get_carrier_options_tool,
    ]

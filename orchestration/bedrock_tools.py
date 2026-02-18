"""
Strands @tool wrappers for supply chain: risk, sourcing, carbon, full pipeline.
Used by the LLM orchestrator to decide which steps to call and in what order.
"""

import json
import os
from pathlib import Path
from typing import Optional

# Ensure code root is cwd for imports
_CODE_DIR = Path(__file__).resolve().parent.parent
if os.getcwd() != str(_CODE_DIR):
    os.chdir(_CODE_DIR)

from orchestration._orchestrator_instance import get_orchestrator


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


try:
    from strands import tool
except ImportError:
    tool = lambda f: f  # no-op if Strands not installed


@tool
def extract_features_tool(query: str) -> str:
    """Extract shipment features from a natural language query (with semantic package_type e.g. insulin->pharmacy). Returns JSON with features and defaults_used."""
    from tools.extraction_tools import extract_from_query_and_merge_defaults
    features, defaults_used = extract_from_query_and_merge_defaults(query)
    return json.dumps(_json_safe({"features": features, "defaults_used": defaults_used}))


@tool
def risk_agent_tool(features_json: str, risk_queries_json: str) -> str:
    """Assess delivery risk using the Risk Strands Agent (LLM). Gathers weather/news from API + web search (union), fuses conflicting data, returns risk_level, risk_factors, delay_probability, recommended_buffer_days. You must pass risk_queries_json: a JSON array of adaptive search queries (e.g. ["Mumbai weather today", "Delhi weather today", "Mumbai port strike 2025", "India logistics disruption"]). Generate these queries from the user request and features (origin, destination, package_type)."""
    from orchestration.risk_agent_strands import run_risk_agent
    return run_risk_agent(features_json, risk_queries_json)


@tool
def sourcing_agent_tool(features_json: str, risk_assessment_json: str, sourcing_queries_json: str) -> str:
    """Get carrier options and recommendation using the Sourcing Strands Agent (LLM). Returns list of carrier options (for run_optimization_tool). You must pass sourcing_queries_json: a JSON array of adaptive search queries (e.g. ["Delhivery rates Mumbai Delhi 2025", "BlueDart pharmacy shipping rates"]). Generate these from features and package_type."""
    from orchestration.sourcing_agent_strands import run_sourcing_agent
    return run_sourcing_agent(features_json, risk_assessment_json, sourcing_queries_json)


@tool
def run_optimization_tool(features_json: str, carrier_options_json: str) -> str:
    """Run route optimization using features and carrier options. Returns best route, cost, on_time, carbon, early_warning, governance."""
    features = json.loads(features_json)
    carrier_options = json.loads(carrier_options_json)
    orch = get_orchestrator()
    from tools.extraction_tools import extract_features_from_dict
    features = extract_features_from_dict(features)
    package_type = (features.get("package_type") or "clothing").lower()
    route_options = [opt["route"] for opt in carrier_options if opt.get("route")]
    if not route_options:
        return json.dumps({"error": "No route options available"})
    result = orch.sourcing_service.run_optimization(
        package_type=package_type,
        route_options=route_options,
        cost_predictor=orch.cost_predictor,
        on_time_predictor=orch.on_time_predictor,
        origin=features.get("origin"),
        destination=features.get("destination"),
        priority="carbon",
    )
    # Add early-warning and governance
    best_route = result.get("best_route", route_options[0])
    delay_prob = orch.ews.predict_delay_probability(best_route)
    risk = orch.ews.calculate_risk_score(package_type, delay_prob, best_route)
    result["early_warning"] = risk
    result["governance"] = orch.governance.generate_policy_recommendations(package_type, result)
    return json.dumps(_json_safe(result))


@tool
def carbon_analysis_tool(carrier_options_json: str, package_type: str, optimization_result_json: str) -> str:
    """Analyze carbon and governance for the given carrier options and optimization result. Returns greenest_viable, ai_carbon, transport_carbon, governance."""
    carrier_options = json.loads(carrier_options_json)
    opt_result = json.loads(optimization_result_json)
    orch = get_orchestrator()
    carbon_result = orch.carbon_service.analyze(
        carrier_options=carrier_options,
        package_type=package_type,
        optimization_result=opt_result,
        country="india",
    )
    return json.dumps(_json_safe(carbon_result))


# Export for orchestrator_agent
def get_all_tools():
    return [
        extract_features_tool,
        risk_agent_tool,
        sourcing_agent_tool,
        run_optimization_tool,
        carbon_analysis_tool,
    ]

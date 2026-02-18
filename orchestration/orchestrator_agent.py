"""
Strands orchestrator agent: Bedrock LLM decides which tools to call and in what order.
Coordinates risk, sourcing, optimization, and carbon analysis.
"""

import os
from pathlib import Path
from typing import Optional

_CODE_DIR = Path(__file__).resolve().parent.parent
if os.getcwd() != str(_CODE_DIR):
    os.chdir(_CODE_DIR)

ORCHESTRATOR_PROMPT = """You are a supply chain optimization orchestrator that coordinates specialized agents and tools. You read and understand the user query and decide which tools or agents to call.

Your tools:
1. **extract_features_tool(query)** – Extract shipment features from natural language (e.g. "insulin Mumbai to Delhi 150km"). Returns features + defaults_used.
2. **risk_agent_tool(features_json, risk_queries_json)** – Assess delivery risk via the Risk Agent (LLM). It gathers weather/news from API + web search (union), fuses conflicting data, and returns risk_level, risk_factors, delay_probability, recommended_buffer_days. You MUST pass risk_queries_json: a JSON array of adaptive search queries. Generate these from the user request and features (origin, destination, package_type). Examples: ["Mumbai weather today", "Delhi weather today", "Mumbai port strike 2025", "India logistics disruption", "Mumbai port status today"].
3. **sourcing_agent_tool(features_json, risk_assessment_json, sourcing_queries_json)** – Get carrier options via the Sourcing Agent (LLM). Returns list of carrier options. You MUST pass sourcing_queries_json: a JSON array of adaptive search queries (e.g. ["Delhivery rates Mumbai Delhi 2025", "BlueDart pharmacy shipping rates", "cold chain logistics Mumbai Delhi cost"]).
4. **run_optimization_tool(features_json, carrier_options_json)** – Run route optimization. Use after sourcing_agent_tool. carrier_options_json is the return value of sourcing_agent_tool. Returns best route, cost, on-time, carbon, governance.
5. **carbon_analysis_tool(carrier_options_json, package_type, optimization_result_json)** – Carbon and governance analysis (Python tool, no LLM). Use carrier options and optimization result.

Guidelines:
- Always use the step-by-step multi-agent path: extract_features_tool(query) → risk_agent_tool(features_json, risk_queries_json) → sourcing_agent_tool(features_json, risk_assessment_json, sourcing_queries_json) → run_optimization_tool(features_json, carrier_options_json) → carbon_analysis_tool(carrier_options_json, package_type, optimization_result_json). When calling risk_agent_tool, generate 4–6 adaptive risk queries (weather, disruptions, port/lane events) from origin, destination, and package_type. When calling sourcing_agent_tool, generate 2–4 pricing/sourcing queries.
- Always summarize the outcome: recommended carrier, cost, on-time probability, carbon, risk level, and any governance notes.
- Package types are classified semantically (e.g. insulin → pharmacy). Use extracted features as-is unless the user asks to change them."""

_orchestrator_agent = None


def get_orchestrator_agent(region_name: Optional[str] = None, model_id: Optional[str] = None):
    """Create or return the Strands orchestrator agent with Bedrock and tools."""
    global _orchestrator_agent
    if _orchestrator_agent is not None:
        return _orchestrator_agent
    try:
        from strands import Agent
        from strands.models import BedrockModel
    except ImportError as e:
        raise ImportError("Strands and Bedrock are required for orchestration. Install: pip install strands-agents boto3") from e
    from orchestration.bedrock_tools import get_all_tools

    region_name = region_name or os.environ.get("AWS_REGION", "us-east-1")
    model_id = model_id or os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-3-7-sonnet-20250219-v1:0")
    bedrock_model = BedrockModel(
        model_id=model_id,
        region_name=region_name,
        temperature=0.0,
    )
    _orchestrator_agent = Agent(
        model=bedrock_model,
        system_prompt=ORCHESTRATOR_PROMPT,
        tools=get_all_tools(),
    )
    return _orchestrator_agent


def run_orchestrator(message: str, region_name: Optional[str] = None, model_id: Optional[str] = None) -> str:
    """Run the orchestrator with a user message and return the response text."""
    agent = get_orchestrator_agent(region_name=region_name, model_id=model_id)
    return agent(message)


if __name__ == "__main__":
    import sys
    msg = sys.argv[1] if len(sys.argv) > 1 else "Optimize delivery for insulin from Mumbai to Delhi, 150 km."
    print(run_orchestrator(msg))

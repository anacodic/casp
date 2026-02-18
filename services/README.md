# Services (Python Backend)

This folder contains **Python backend services**, not LLM agents.

| Component | Role | Used by |
|-----------|------|---------|
| **RiskService** | Risk assessment (Module 07, weather cascade) | calculate_risk_score_tool uses orch.ews |
| **SourcingService** | Carrier options + optimization (Modules 01, 02, carriers, routes) | get_carrier_options_tool, run_optimization_tool |
| **CarbonService** | Carbon and governance (Modules 03–06) | carbon_analysis_tool |
| **StakesOptimizer** | Single optimizer for all stakes levels (critical 99%, high_value 95%, standard 85%); carbon-priority re-sort for standard | SourcingService.run_optimization, SupplyChainOrchestrator.optimize_delivery |

**LLM agents** (Strands) are in **orchestration/**: `orchestrator_agent.py`, `risk_agent_strands.py`, `sourcing_agent_strands.py`.

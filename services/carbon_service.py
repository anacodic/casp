"""
Carbon Service: Python backend for carbon and governance (Modules 03–06).
Used by carbon_analysis_tool. No LLM; tool only.
"""

from typing import Dict, List, Any
from tools.carbon_tools import (
    calculate_ai_carbon,
    get_grid_intensity,
    analyze_tradeoffs,
    get_governance_advice,
)


class CarbonService:
    """Python service for carbon and governance using Modules 03, 04, 05, 06."""

    def __init__(self, carbon_intel, frontiers, governance, country: str = "india"):
        """
        Args:
            carbon_intel: CarbonCostOfIntelligence instance (Module 03)
            frontiers: TradeOffFrontiers instance (Module 05)
            governance: GovernanceLevers instance (Module 06)
            country: Default country for grid carbon (Module 04 via config)
        """
        self.carbon_intel = carbon_intel
        self.frontiers = frontiers
        self.governance = governance
        self.country = country

    def analyze(
        self,
        carrier_options: List[Dict],
        package_type: str,
        optimization_result: Dict[str, Any] = None,
        country: str = None,
    ) -> Dict[str, Any]:
        """
        Analyze carbon trade-offs and get governance advice.
        carrier_options: list from Sourcing Service (with carbon, cost, on_time).
        optimization_result: optional full result from sourcing (for governance input).
        """
        country = country or self.country

        ai_carbon_result = calculate_ai_carbon(
            self.carbon_intel,
            model_type="gemini-flash",
            country=country,
            num_inferences=1,
        )
        ai_carbon_gco2 = ai_carbon_result["ai_carbon_gco2"]
        grid_intensity = get_grid_intensity(country)
        tradeoff_result = analyze_tradeoffs(
            self.frontiers,
            carrier_options,
            package_type,
        )

        viable = [o for o in carrier_options if o.get("meets_sla", True)]
        options_to_rank = viable if viable else carrier_options
        if options_to_rank:
            greenest = min(
                options_to_rank,
                key=lambda x: x.get("total_carbon_gco2")
                or x.get("predicted_carbon_gco2")
                or (x.get("carbon_kg", 0) * 1000),
            )
            transport_carbon = (
                greenest.get("total_carbon_gco2")
                or greenest.get("predicted_carbon_gco2")
                or (greenest.get("carbon_kg", 0) * 1000)
            )
            greenest_carrier = (
                greenest.get("carrier")
                or greenest.get("route", {}).get("delivery_partner", "")
            )
        else:
            transport_carbon = 0
            greenest_carrier = ""

        total_carbon = transport_carbon + ai_carbon_gco2
        gov_input = optimization_result or {
            "best_route": carrier_options[0].get("route", {}) if carrier_options else {},
            "breakdown": {
                "total_carbon_gco2": total_carbon,
                "predicted_on_time": carrier_options[0].get("predicted_on_time_pct", 0)
                if carrier_options
                else 0,
                "cost": carrier_options[0].get("predicted_cost", 0) if carrier_options else 0,
            },
            "package_type": package_type,
        }
        governance_recs = get_governance_advice(
            self.governance, package_type, gov_input
        )

        return {
            "greenest_viable": greenest_carrier,
            "transport_carbon_gco2": transport_carbon,
            "ai_carbon_gco2": ai_carbon_gco2,
            "total_carbon_gco2": total_carbon,
            "grid_intensity_gco2_kwh": grid_intensity,
            "tradeoff_analysis": tradeoff_result,
            "governance": governance_recs,
        }

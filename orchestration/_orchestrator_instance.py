"""
Singleton SupplyChainOrchestrator instance for use by bedrock_tools and agent_tools.
Avoids circular imports when agent_tools need the orchestrator (ews, sourcing_service, etc.).
"""

import os
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parent.parent
if os.getcwd() != str(_CODE_DIR):
    os.chdir(_CODE_DIR)

_orchestrator = None


def get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        from supply_chain_orchestrator import SupplyChainOrchestrator
        _orchestrator = SupplyChainOrchestrator(
            data_path=str(_CODE_DIR / "data" / "datasets" / "Delivery_Logistics.csv")
        )
    return _orchestrator

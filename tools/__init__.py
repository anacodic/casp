"""
Tools layer: wrap existing modules for use by services and orchestration.
"""

from tools.risk_tools import assess_risk
from tools.sourcing_tools import (
    lookup_route,
    get_package_rules,
    get_carrier_quotes_for_features,
    predict_performance,
    get_carrier_cluster_name,
    filter_options_by_sla,
    build_route_options_from_quotes,
)
from tools.carbon_tools import (
    calculate_ai_carbon,
    get_grid_intensity,
    analyze_tradeoffs,
    get_governance_advice,
)
from tools.extraction_tools import extract_features_from_dict

__all__ = [
    'assess_risk',
    'lookup_route',
    'get_package_rules',
    'get_carrier_quotes_for_features',
    'predict_performance',
    'get_carrier_cluster_name',
    'filter_options_by_sla',
    'build_route_options_from_quotes',
    'calculate_ai_carbon',
    'get_grid_intensity',
    'analyze_tradeoffs',
    'get_governance_advice',
    'extract_features_from_dict',
]


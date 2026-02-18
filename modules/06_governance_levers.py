"""
OUTPUT 2: GOVERNANCE LEVERS
Policy recommendations
Sourcing portfolio rules
Buffer sizing
Compute policies
"""

import pandas as pd
from typing import Dict, List
from config.agent_mapping import AGENT_MAPPING, get_agent_config

class GovernanceLevers:
    """Generate governance policy recommendations."""
    
    def __init__(self):
        self.policies = []
    
    def generate_policy_recommendations(
        self,
        package_type: str,
        analysis_results: Dict
    ) -> Dict:
        """
        Generate policy recommendations for a package type.
        
        Args:
            package_type: Type of package
            analysis_results: Results from optimization analysis
        
        Returns:
            Dictionary with policy recommendations
        """
        config = get_agent_config(package_type)
        stakes_level = config['agent']  # 'critical' | 'high_value' | 'standard'
        
        policies = {
            'package_type': package_type,
            'stakes_level': stakes_level,
            'on_time_threshold': config['on_time_threshold'] * 100,
            'recommendations': []
        }
        
        # Vehicle recommendations (map critical->high, standard->low, high_value->medium)
        if stakes_level == 'critical':
            policies['recommendations'].append({
                'category': 'Vehicle Selection',
                'recommendation': 'Use EV Van (minimize carbon within safety constraints)',
                'rationale': 'High-stakes packages require reliable transport with carbon optimization'
            })
        elif stakes_level == 'standard':
            policies['recommendations'].append({
                'category': 'Vehicle Selection',
                'recommendation': 'Use Bike/Scooter (maximize carbon savings)',
                'rationale': 'Low-stakes packages can use low-carbon vehicles without penalty'
            })
        else:
            policies['recommendations'].append({
                'category': 'Vehicle Selection',
                'recommendation': 'Use EV Van or Van (balanced approach)',
                'rationale': 'Medium-stakes packages balance cost and carbon'
            })
        
        # Buffer recommendations
        if stakes_level == 'critical':
            policies['recommendations'].append({
                'category': 'Inventory Buffer',
                'recommendation': '+20% safety stock buffer',
                'rationale': 'High-stakes packages require safety stock to meet ≥99% on-time'
            })
        elif stakes_level == 'standard':
            policies['recommendations'].append({
                'category': 'Inventory Buffer',
                'recommendation': '-10% lean inventory (reduce waste)',
                'rationale': 'Low-stakes packages can operate with lean inventory'
            })
        else:
            policies['recommendations'].append({
                'category': 'Inventory Buffer',
                'recommendation': '+10% standard buffer',
                'rationale': 'Medium-stakes packages need moderate buffer'
            })
        
        # Carrier recommendations
        if stakes_level == 'critical':
            policies['recommendations'].append({
                'category': 'Carrier Selection',
                'recommendation': 'Premium partners only (≥98% reliability)',
                'rationale': 'High-stakes packages require maximum reliability'
            })
        elif stakes_level == 'standard':
            policies['recommendations'].append({
                'category': 'Carrier Selection',
                'recommendation': 'Budget carriers OK (≥85% reliability)',
                'rationale': 'Low-stakes packages can use cost-effective carriers'
            })
        else:
            policies['recommendations'].append({
                'category': 'Carrier Selection',
                'recommendation': 'Standard carriers (≥95% reliability)',
                'rationale': 'Medium-stakes packages need reliable but cost-effective options'
            })
        
        # Compute recommendations
        if stakes_level == 'critical':
            policies['recommendations'].append({
                'category': 'AI Compute Policy',
                'recommendation': 'Use Gemini Flash (lowest AI carbon)',
                'rationale': 'Minimize AI compute carbon while maintaining safety'
            })
        elif stakes_level == 'standard':
            policies['recommendations'].append({
                'category': 'AI Compute Policy',
                'recommendation': 'Batch queries (reduce AI calls)',
                'rationale': 'Low-stakes packages can batch optimizations to reduce AI carbon'
            })
        else:
            policies['recommendations'].append({
                'category': 'AI Compute Policy',
                'recommendation': 'Use Gemini Flash with moderate query frequency',
                'rationale': 'Balance AI optimization benefits with carbon cost'
            })
        
        # Location recommendations
        if stakes_level == 'high_stakes':
            policies['recommendations'].append({
                'category': 'Sourcing Location',
                'recommendation': 'Ship from low-grid-carbon regions (France, Canada)',
                'rationale': 'High-stakes packages benefit from low-carbon AI compute'
            })
        else:
            policies['recommendations'].append({
                'category': 'Sourcing Location',
                'recommendation': 'Flexible location (optimize for cost)',
                'rationale': 'Lower-stakes packages prioritize cost over carbon location'
            })
        
        return policies
    
    def generate_portfolio_rules(self) -> Dict:
        """Generate sourcing portfolio rules."""
        return {
            'high_stakes_portfolio': {
                'pharmacy': {
                    'min_reliability': 0.98,
                    'max_cost_premium': 1.5,
                    'required_vehicles': ['ev van', 'van'],
                    'buffer_percentage': 0.20
                },
                'electronics': {
                    'min_reliability': 0.95,
                    'max_cost_premium': 1.3,
                    'required_vehicles': ['ev van', 'van'],
                    'buffer_percentage': 0.15
                },
                'groceries': {
                    'min_reliability': 0.95,
                    'max_cost_premium': 1.4,
                    'required_vehicles': ['ev van', 'truck'],
                    'buffer_percentage': 0.20,
                    'cold_chain_required': True
                }
            },
            'medium_stakes_portfolio': {
                'min_reliability': 0.90,
                'max_cost_premium': 1.2,
                'allowed_vehicles': ['ev van', 'van', 'truck'],
                'buffer_percentage': 0.10
            },
            'low_stakes_portfolio': {
                'min_reliability': 0.85,
                'max_cost_premium': 1.0,
                'allowed_vehicles': ['bike', 'ev bike', 'scooter', 'ev van'],
                'buffer_percentage': -0.10  # Lean inventory
            }
        }
    
    def format_policy_report(self, policies: Dict) -> str:
        """Format policy recommendations as readable report."""
        output = f"""
📋 GOVERNANCE POLICY RECOMMENDATIONS
Package Type: {policies['package_type']}
Stakes Level: {policies['stakes_level']}
On-Time Threshold: ≥{policies['on_time_threshold']}%

POLICY RECOMMENDATIONS:
"""
        for i, rec in enumerate(policies['recommendations'], 1):
            output += f"""
{i}. {rec['category']}
   Recommendation: {rec['recommendation']}
   Rationale: {rec['rationale']}
"""
        return output

if __name__ == "__main__":
    # Test the module
    governance = GovernanceLevers()
    
    # Generate policies for different package types
    print("📋 Policy Recommendations for Pharmacy:")
    pharma_policies = governance.generate_policy_recommendations('pharmacy', {})
    print(governance.format_policy_report(pharma_policies))
    
    print("\n📋 Policy Recommendations for Clothing:")
    clothing_policies = governance.generate_policy_recommendations('clothing', {})
    print(governance.format_policy_report(clothing_policies))
    
    # Portfolio rules
    print("\n📊 Sourcing Portfolio Rules:")
    portfolio = governance.generate_portfolio_rules()
    print(f"High-Stakes Portfolio: {portfolio['high_stakes_portfolio']}")

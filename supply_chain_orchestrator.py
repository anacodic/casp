"""
Supply Chain Orchestrator - Multi-Agent System
Routes packages to appropriate services/agents based on package_type.
Coordinates all modules: predictive analytics, optimization, governance, early-warning.

Integrated flow: Extract → Risk Service (Module 07) → Sourcing Service (01, 02, carriers, routes)
→ Carbon Service (03, 04, 05, 06) → Final decision.
Python backend services live in services/; LLM agents live in orchestration/.
"""

import importlib
from typing import List, Dict, Optional, Any
from config.agent_mapping import get_agent_config, get_stakes_level
from services.stakes_optimizer import StakesOptimizer
from services.risk_service import RiskService
from services.sourcing_service import SourcingService
from services.carbon_service import CarbonService
# Module names starting with digits are invalid in "from X import Y"; load via importlib.
_mod01 = importlib.import_module("modules.01_predictive_analytics")
create_predictors = _mod01.create_predictors
_mod02 = importlib.import_module("modules.02_vendor_segmentation")
VendorSegmentation = _mod02.VendorSegmentation
_mod03 = importlib.import_module("modules.03_carbon_cost_intelligence")
CarbonCostOfIntelligence = _mod03.CarbonCostOfIntelligence
_mod07 = importlib.import_module("modules.07_early_warning_system")
EarlyWarningSystem = _mod07.EarlyWarningSystem
_mod06 = importlib.import_module("modules.06_governance_levers")
GovernanceLevers = _mod06.GovernanceLevers
_mod05 = importlib.import_module("modules.05_trade_off_frontiers")
TradeOffFrontiers = _mod05.TradeOffFrontiers
from tools.extraction_tools import extract_features_from_dict
from tools.sourcing_tools import build_route_options_from_quotes

class SupplyChainOrchestrator:
    """Main orchestrator for supply chain optimization."""
    
    def __init__(self, data_path: str = 'data/datasets/Delivery_Logistics.csv'):
        """Initialize orchestrator with all services and modules."""
        self.stakes_optimizer = StakesOptimizer()
        
        # Initialize ML predictors
        print("🔧 Initializing ML predictors...")
        self.cost_predictor, self.on_time_predictor, self.analytics = create_predictors(data_path)
        
        # Initialize early-warning system
        print("🔧 Initializing early-warning system...")
        self.ews = EarlyWarningSystem(data_path)
        self.ews.load_data()
        self.ews.train_delay_predictor()
        
        # Initialize governance
        self.governance = GovernanceLevers()
        
        # Initialize trade-off frontiers
        self.frontiers = TradeOffFrontiers()
        
        # Module 02: Vendor segmentation (for Sourcing Agent)
        print("🔧 Initializing vendor segmentation...")
        self.vendor_segmentation = VendorSegmentation(data_path)
        self.vendor_segmentation.load_data()
        self.vendor_segmentation.prepare_vendor_features()
        self.vendor_segmentation.cluster_vendors(n_clusters=4)
        
        # Module 03: Carbon cost of intelligence (for Carbon Agent)
        self.carbon_intel = CarbonCostOfIntelligence(default_country='india')
        
        # Integrated flow services (Python backend): Risk, Sourcing, Carbon
        self.risk_service = RiskService(self.ews)
        self.sourcing_service = SourcingService(
            self.cost_predictor,
            self.on_time_predictor,
            self.analytics,
            self.vendor_segmentation,
        )
        self.carbon_service = CarbonService(
            self.carbon_intel,
            self.frontiers,
            self.governance,
            country='india',
        )
        
        print("✅ Orchestrator initialized!")
    
    def route_to_agent(self, package_type: str):
        """Return the single StakesOptimizer (stakes level is derived inside optimizer from package_type)."""
        return self.stakes_optimizer
    
    def optimize_delivery(
        self,
        package_type: str,
        route_options: List[Dict],
        origin: str = None,
        destination: str = None,
        priority: str = 'carbon'
    ) -> Dict:
        """
        Optimize delivery route for a package.
        
        Args:
            package_type: Type of package
            route_options: List of route dictionaries
            origin: Origin location (optional)
            destination: Destination location (optional)
            priority: Optimization priority ('carbon' or 'cost')
        
        Returns:
            Complete optimization results with all analyses
        """
        optimizer = self.route_to_agent(package_type)
        result = optimizer.optimize_route(
            package_type,
            route_options,
            self.cost_predictor,
            self.on_time_predictor,
            origin,
            destination,
            priority,
        )
        
        # Add early-warning analysis
        best_route = result['best_route']
        delay_prob = self.ews.predict_delay_probability(best_route)
        risk = self.ews.calculate_risk_score(package_type, delay_prob, best_route)
        result['early_warning'] = risk
        
        # Add governance recommendations
        governance_recs = self.governance.generate_policy_recommendations(
            package_type, result
        )
        result['governance'] = governance_recs
        
        # Add to trade-off frontier
        breakdown = result['breakdown']
        self.frontiers.add_result(
            package_type,
            breakdown['total_carbon_gco2'],
            breakdown.get('predicted_on_time', 0),
            breakdown.get('cost', 0),
            result.get('casp_score', 0)
        )
        
        return result
    
    def generate_comprehensive_report(
        self,
        package_type: str,
        route_options: List[Dict],
        origin: str = None,
        destination: str = None
    ) -> str:
        """
        Generate comprehensive optimization report.
        
        Args:
            package_type: Type of package
            route_options: List of route dictionaries
            origin: Origin location (optional)
            destination: Destination location (optional)
        
        Returns:
            Formatted report string
        """
        # Run optimization
        result = self.optimize_delivery(package_type, route_options, origin, destination)
        
        optimizer = self.route_to_agent(package_type)
        report = f"""
{'='*70}
COMPREHENSIVE SUPPLY CHAIN OPTIMIZATION REPORT
{'='*70}

"""
        report += optimizer.format_output(result)
        
        # Early-warning section
        ews = result['early_warning']
        report += f"""

⚠️ EARLY-WARNING ANALYSIS:
• Delay Probability: {ews['delay_probability']*100:.1f}%
• Risk Score: {ews['risk_score']:.2f} ({ews['risk_level']})
• Alert Required: {'YES ⚠️' if ews['alert_required'] else 'NO'}
"""
        if ews['risk_factors']:
            report += f"• Risk Factors: {', '.join(ews['risk_factors'])}\n"
        
        # Governance section
        report += f"""

📋 GOVERNANCE RECOMMENDATIONS:
"""
        for rec in result['governance']['recommendations']:
            report += f"• {rec['category']}: {rec['recommendation']}\n"
        
        report += f"""

{'='*70}
"""
        
        return report
    
    def compare_package_types(
        self,
        route_options: List[Dict],
        package_types: List[str] = None
    ) -> Dict:
        """
        Compare optimization results across different package types.
        
        Args:
            route_options: List of route dictionaries
            package_types: List of package types to compare (default: all)
        
        Returns:
            Comparison results
        """
        if package_types is None:
            package_types = ['pharmacy', 'clothing', 'electronics', 'fragile items']
        
        results = {}
        
        for ptype in package_types:
            result = self.optimize_delivery(ptype, route_options)
            results[ptype] = {
                'casp_score': result.get('casp_score', 0),
                'total_carbon': result['breakdown']['total_carbon_gco2'],
                'service_level': result['breakdown'].get('predicted_on_time', 0),
                'cost': result['breakdown'].get('cost', 0),
                'risk_score': result['early_warning']['risk_score']
            }
        
        return results
    
    def run_integrated_pipeline(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the integrated multi-agent flow: Extract → Risk → Sourcing → Carbon → Final decision.
        
        features: dict with at least package_type, origin, destination; optional distance_km,
                  package_weight_kg, weather_condition, delivery_mode, region.
        
        Returns: recommendation (carrier), cost, on_time_probability, transport_carbon, ai_carbon,
                 risk_level, risk_factors, governance, tradeoff_analysis.
        """
        features = extract_features_from_dict(features)
        package_type = (features.get('package_type') or 'clothing').lower()
        origin = str(features.get('origin') or 'mumbai').lower()
        destination = str(features.get('destination') or 'delhi').lower()
        weather = (features.get('weather_condition') or 'clear').lower()
        distance_km = features.get('distance_km')
        package_weight_kg = features.get('package_weight_kg')
        
        # 1. Risk Service (Module 07)
        risk_assessment = self.risk_service.assess(
            origin=origin,
            destination=destination,
            weather_condition=weather,
            package_type=package_type,
            distance_km=distance_km,
            package_weight_kg=package_weight_kg,
        )
        
        # 2. Sourcing Service: get carrier options (Module 01, 02, carriers, routes)
        carrier_options = self.sourcing_service.get_carrier_options(features, risk_assessment)
        if not carrier_options:
            return {
                'recommendation': None,
                'error': 'No carrier options found for this shipment',
                'risk_assessment': risk_assessment,
                'carrier_options': [],
                'carbon_analysis': {},
            }
        
        # Build route_options for full optimization (stakes agent + gradient descent)
        if distance_km is None:
            from tools.sourcing_tools import lookup_route
            route_info = lookup_route(origin, destination)
            distance_km = route_info['distance_km'] if route_info else 150
        route_options = [o['route'] for o in carrier_options]
        
        # 3. Run full optimization (stakes optimizers + Module 01)
        opt_result = self.sourcing_service.run_optimization(
            package_type=package_type,
            route_options=route_options,
            cost_predictor=self.cost_predictor,
            on_time_predictor=self.on_time_predictor,
            origin=origin,
            destination=destination,
            priority='carbon',
        )
        
        # Add early-warning (Module 07) and governance (Module 06) to opt_result
        best_route = opt_result.get('best_route', route_options[0] if route_options else {})
        delay_prob = self.ews.predict_delay_probability(best_route)
        risk = self.ews.calculate_risk_score(package_type, delay_prob, best_route)
        opt_result['early_warning'] = risk
        opt_result['governance'] = self.governance.generate_policy_recommendations(
            package_type, opt_result
        )
        
        # Add to trade-off frontier (Module 05)
        breakdown = opt_result.get('breakdown', {})
        self.frontiers.add_result(
            package_type,
            breakdown.get('total_carbon_gco2', 0),
            breakdown.get('predicted_on_time', 0),
            breakdown.get('cost', 0),
            opt_result.get('casp_score', 0),
        )
        
        # 4. Carbon Service (Module 03, 04, 05, 06)
        carbon_result = self.carbon_service.analyze(
            carrier_options,
            package_type,
            optimization_result=opt_result,
            country='india',
        )
        
        # 5. Industry benchmark: web search first, then fallback to avg carrier cost_inr
        cost = breakdown.get('cost', 0)
        industry_benchmark = None
        efficiency = None
        efficiency_percentage = None
        try:
            from tools.web_search import web_search, parse_cost_from_text
            dist = distance_km if distance_km is not None else 150
            weight = package_weight_kg if package_weight_kg is not None else 25
            query = f"delivery cost {package_type} {dist}km {weight}kg India"
            snippet = web_search(query)
            if snippet:
                industry_benchmark = parse_cost_from_text(snippet)
        except Exception:
            pass
        # Fallback: use average of carrier cost_inr / predicted_cost as benchmark when web search returns nothing
        if industry_benchmark is None and carrier_options:
            costs = []
            for o in carrier_options:
                c = o.get("cost_inr") or o.get("predicted_cost")
                if c is not None and c > 0:
                    costs.append(float(c))
            if costs:
                industry_benchmark = sum(costs) / len(costs)
        if industry_benchmark is not None and cost and industry_benchmark > 0:
            pct = ((industry_benchmark - cost) / industry_benchmark) * 100
            efficiency_percentage = f"{pct:.1f}% {'below' if pct > 0 else 'above'} benchmark"
            efficiency = "Below benchmark" if cost < industry_benchmark else "Above benchmark"

        # Attach paper-aligned early-warning indicators to opt_result['early_warning']
        opt_result['early_warning']['early_warning_indicators'] = risk_assessment.get('early_warning_indicators', {})

        # 6. Final combined output
        best_route = opt_result.get('best_route', {})
        out = {
            'recommendation': best_route.get('delivery_partner') or (carrier_options[0].get('carrier') if carrier_options else None),
            'cost': cost,
            'on_time_probability': breakdown.get('predicted_on_time', 0),
            'transport_carbon': breakdown.get('total_carbon_gco2', 0),
            'ai_carbon': carbon_result.get('ai_carbon_gco2', 0),
            'risk_level': risk_assessment.get('risk_level', 'LOW'),
            'risk_factors': risk_assessment.get('risk_factors', []),
            'warnings': risk_assessment.get('warnings', []),
            'recommended_buffer_days': risk_assessment.get('recommended_buffer_days', 0),
            'early_warning_indicators': risk_assessment.get('early_warning_indicators', {}),
            'governance': carbon_result.get('governance', opt_result.get('governance', {})),
            'tradeoff': carbon_result.get('tradeoff_analysis', {}),
            'greenest_viable': carbon_result.get('greenest_viable', ''),
            'carrier_options': carrier_options,
            'optimization_result': opt_result,
            'carbon_analysis': carbon_result,
        }
        if industry_benchmark is not None:
            out['industry_benchmark'] = round(industry_benchmark, 2)
        if efficiency is not None:
            out['efficiency'] = efficiency
        if efficiency_percentage is not None:
            out['efficiency_percentage'] = efficiency_percentage
        return out

if __name__ == "__main__":
    # Test the orchestrator
    print("🌍 Initializing Supply Chain Orchestrator...\n")
    orchestrator = SupplyChainOrchestrator()
    
    # Example route options
    test_routes = [
        {
            'route_id': 'Route_A',
            'distance_km': 150,
            'vehicle_type': 'ev van',
            'delivery_partner': 'delhivery',
            'delivery_mode': 'express',
            'region': 'west',
            'weather_condition': 'clear',
            'package_weight_kg': 25,
            'delivery_rating': 4
        },
        {
            'route_id': 'Route_B',
            'distance_km': 200,
            'vehicle_type': 'van',
            'delivery_partner': 'dhl',
            'delivery_mode': 'same day',
            'region': 'west',
            'weather_condition': 'clear',
            'package_weight_kg': 25,
            'delivery_rating': 5
        },
        {
            'route_id': 'Route_C',
            'distance_km': 120,
            'vehicle_type': 'bike',
            'delivery_partner': 'shadowfax',
            'delivery_mode': 'two day',
            'region': 'west',
            'weather_condition': 'clear',
            'package_weight_kg': 25,
            'delivery_rating': 3
        }
    ]
    
    # Test with pharmacy (high-stakes)
    print("\n" + "="*70)
    print("TEST 1: PHARMACY (High-Stakes)")
    print("="*70)
    report1 = orchestrator.generate_comprehensive_report('pharmacy', test_routes)
    print(report1)
    
    # Test with clothing (low-stakes)
    print("\n" + "="*70)
    print("TEST 2: CLOTHING (Low-Stakes)")
    print("="*70)
    report2 = orchestrator.generate_comprehensive_report('clothing', test_routes)
    print(report2)
    
    # Compare package types
    print("\n" + "="*70)
    print("TEST 3: PACKAGE TYPE COMPARISON")
    print("="*70)
    comparison = orchestrator.compare_package_types(test_routes)
    print("\nComparison Results:")
    for ptype, metrics in comparison.items():
        print(f"\n{ptype}:")
        print(f"  CASP: {metrics['casp_score']:.6f}")
        print(f"  Carbon: {metrics['total_carbon']:.0f} gCO2")
        print(f"  Service: {metrics['service_level']:.1f}%")
        print(f"  Risk: {metrics['risk_score']:.2f}")

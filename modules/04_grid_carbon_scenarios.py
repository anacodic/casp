"""
MODULE 2: NATIONAL ENERGY-TRANSITION CONDITIONS
Grid Carbon Intensity by Country
Compare India, USA, France, China grids
Show impact on total carbon
Scenario analysis by country
"""

import pandas as pd
import numpy as np
from config.grid_carbon import (
    GRID_CARBON_INTENSITY,
    calculate_ai_compute_carbon,
    calculate_model_carbon,
    get_country_from_region
)

class GridCarbonScenarios:
    """Analyze grid carbon scenarios by country."""
    
    def __init__(self):
        self.countries = list(GRID_CARBON_INTENSITY.keys())
    
    def compare_countries(self, energy_joules: float = 500) -> pd.DataFrame:
        """
        Compare AI compute carbon across countries.
        
        Args:
            energy_joules: Energy consumption in Joules (default: 500 J for ~1000 tokens)
        
        Returns:
            DataFrame with country comparisons
        """
        results = []
        
        for country in self.countries:
            grid_intensity = GRID_CARBON_INTENSITY[country]
            ai_carbon = calculate_ai_compute_carbon(energy_joules, country)
            
            results.append({
                'country': country,
                'grid_intensity_gco2_kwh': grid_intensity,
                'ai_compute_carbon_gco2': ai_carbon,
                'energy_joules': energy_joules
            })
        
        df = pd.DataFrame(results)
        df = df.sort_values('ai_compute_carbon_gco2')
        
        return df
    
    def analyze_route_by_country(
        self,
        transport_carbon: float,
        ai_tokens: int = 500,
        model: str = 'gemini-flash'
    ) -> pd.DataFrame:
        """
        Analyze total carbon (transport + AI) by country.
        
        Args:
            transport_carbon: Transport carbon in gCO2
            ai_tokens: Number of AI tokens
            model: AI model name
        
        Returns:
            DataFrame with total carbon by country
        """
        results = []
        
        for country in self.countries:
            ai_carbon = calculate_model_carbon(ai_tokens, model, country)
            total_carbon = transport_carbon + ai_carbon
            
            results.append({
                'country': country,
                'transport_carbon_gco2': transport_carbon,
                'ai_compute_carbon_gco2': ai_carbon,
                'total_carbon_gco2': total_carbon,
                'ai_carbon_pct': (ai_carbon / total_carbon * 100) if total_carbon > 0 else 0
            })
        
        df = pd.DataFrame(results)
        df = df.sort_values('total_carbon_gco2')
        
        return df
    
    def get_optimal_country(
        self,
        transport_carbon: float,
        ai_tokens: int = 500,
        model: str = 'gemini-flash'
    ) -> dict:
        """
        Find country with lowest total carbon.
        
        Args:
            transport_carbon: Transport carbon in gCO2
            ai_tokens: Number of AI tokens
            model: AI model name
        
        Returns:
            Dictionary with optimal country info
        """
        df = self.analyze_route_by_country(transport_carbon, ai_tokens, model)
        optimal = df.iloc[0]
        
        return {
            'optimal_country': optimal['country'],
            'total_carbon_gco2': optimal['total_carbon_gco2'],
            'ai_carbon_gco2': optimal['ai_compute_carbon_gco2'],
            'transport_carbon_gco2': optimal['transport_carbon_gco2'],
            'savings_vs_india': df[df['country'] == 'India'].iloc[0]['total_carbon_gco2'] - optimal['total_carbon_gco2']
        }
    
    def scenario_analysis(
        self,
        distance_km: float,
        vehicle_type: str = 'van',
        ai_tokens: int = 500,
        model: str = 'gemini-flash'
    ) -> pd.DataFrame:
        """
        Full scenario analysis: transport + AI carbon by country.
        
        Args:
            distance_km: Distance in kilometers
            vehicle_type: Vehicle type
            ai_tokens: Number of AI tokens
            model: AI model name
        
        Returns:
            DataFrame with scenario results
        """
        from config.vehicle_emissions import calculate_transport_carbon
        
        transport_carbon = calculate_transport_carbon(distance_km, vehicle_type)
        
        return self.analyze_route_by_country(transport_carbon, ai_tokens, model)

if __name__ == "__main__":
    # Test the module
    scenarios = GridCarbonScenarios()
    
    # Compare countries for AI compute
    print("🌍 AI Compute Carbon by Country (500 Joules):")
    print(scenarios.compare_countries(energy_joules=500))
    
    # Analyze route by country
    print("\n📊 Total Carbon (Transport + AI) by Country:")
    print("Route: 150 km, Van, 500 AI tokens")
    result = scenarios.scenario_analysis(150, 'van', 500, 'gemini-flash')
    print(result)
    
    # Find optimal country
    print("\n🎯 Optimal Country for Lowest Carbon:")
    optimal = scenarios.get_optimal_country(120000, 500, 'gemini-flash')
    print(f"  Country: {optimal['optimal_country']}")
    print(f"  Total Carbon: {optimal['total_carbon_gco2']:.0f} gCO2")
    print(f"  Savings vs India: {optimal['savings_vs_india']:.0f} gCO2")

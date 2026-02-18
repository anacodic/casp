"""
OUTPUT 1: CARBON-SERVICE-COST TRADE-OFF FRONTIERS
Pareto frontier visualization
CASP metric calculation
Carbon-service-cost trade-offs
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

class TradeOffFrontiers:
    """Analyze and visualize trade-off frontiers."""
    
    def __init__(self):
        self.results = []
    
    def add_result(
        self,
        package_type: str,
        carbon_gco2: float,
        service_level: float,
        cost: float,
        casp_score: float = None
    ):
        """Add a result point to the frontier."""
        if casp_score is None:
            casp_score = service_level / carbon_gco2 if carbon_gco2 > 0 else 0
        
        self.results.append({
            'package_type': package_type,
            'carbon_gco2': carbon_gco2,
            'service_level': service_level,
            'cost': cost,
            'casp_score': casp_score
        })
    
    def calculate_pareto_frontier(self, dimension: str = 'carbon') -> pd.DataFrame:
        """
        Calculate Pareto frontier.
        
        Args:
            dimension: 'carbon' or 'cost' (what to minimize). 'carbon' maps to column carbon_gco2.
        
        Returns:
            DataFrame with Pareto-optimal points
        """
        if not self.results:
            return pd.DataFrame()
        
        # Map logical dimension names to DataFrame column names
        dim_col = 'carbon_gco2' if dimension == 'carbon' else dimension
        
        df = pd.DataFrame(self.results)
        
        # Sort by service level (descending) and dimension (ascending)
        df = df.sort_values(['service_level', dim_col], ascending=[False, True])
        
        # Find Pareto-optimal points
        pareto_points = []
        best_dimension = float('inf')
        
        for _, row in df.iterrows():
            if row[dim_col] < best_dimension:
                pareto_points.append(row)
                best_dimension = row[dim_col]
        
        return pd.DataFrame(pareto_points)
    
    def plot_pareto_frontier_2d(
        self,
        x_axis: str = 'carbon_gco2',
        y_axis: str = 'service_level',
        save_path: str = None
    ):
        """
        Plot 2D Pareto frontier.
        
        Args:
            x_axis: X-axis metric
            y_axis: Y-axis metric
            save_path: Path to save figure
        """
        if not self.results:
            print("No results to plot!")
            return
        
        df = pd.DataFrame(self.results)
        pareto = self.calculate_pareto_frontier()
        
        plt.figure(figsize=(10, 6))
        
        # Plot all points
        plt.scatter(df[x_axis], df[y_axis], alpha=0.5, label='All Routes', color='gray')
        
        # Plot Pareto frontier
        if not pareto.empty:
            pareto_sorted = pareto.sort_values(x_axis)
            plt.plot(pareto_sorted[x_axis], pareto_sorted[y_axis], 
                    'r-', linewidth=2, label='Pareto Frontier', marker='o')
        
        # Color by package type
        for ptype in df['package_type'].unique():
            subset = df[df['package_type'] == ptype]
            plt.scatter(subset[x_axis], subset[y_axis], 
                       label=f'{ptype}', alpha=0.7, s=100)
        
        plt.xlabel(x_axis.replace('_', ' ').title())
        plt.ylabel(y_axis.replace('_', ' ').title())
        plt.title('Pareto Frontier: Trade-off Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved figure to {save_path}")
        
        plt.show()
    
    def plot_3d_frontier(
        self,
        save_path: str = None
    ):
        """Plot 3D Pareto frontier (Carbon, Service, Cost)."""
        if not self.results:
            print("No results to plot!")
            return
        
        from mpl_toolkits.mplot3d import Axes3D
        
        df = pd.DataFrame(self.results)
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot points
        scatter = ax.scatter(
            df['carbon_gco2'],
            df['service_level'],
            df['cost'],
            c=df['casp_score'],
            cmap='viridis',
            s=100,
            alpha=0.7
        )
        
        ax.set_xlabel('Carbon (gCO2)')
        ax.set_ylabel('Service Level (%)')
        ax.set_zlabel('Cost (₹)')
        ax.set_title('3D Trade-off Frontier: Carbon vs Service vs Cost')
        
        plt.colorbar(scatter, ax=ax, label='CASP Score')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved figure to {save_path}")
        
        plt.show()
    
    def calculate_casp_ranking(self) -> pd.DataFrame:
        """Calculate and rank by CASP score."""
        if not self.results:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.results)
        df = df.sort_values('casp_score', ascending=False)
        
        return df
    
    def get_recommendations(self) -> Dict:
        """Get trade-off recommendations."""
        if not self.results:
            return {}
        
        df = pd.DataFrame(self.results)
        
        # Best CASP
        best_casp = df.loc[df['casp_score'].idxmax()]
        
        # Lowest carbon
        lowest_carbon = df.loc[df['carbon_gco2'].idxmin()]
        
        # Highest service
        highest_service = df.loc[df['service_level'].idxmax()]
        
        # Lowest cost
        lowest_cost = df.loc[df['cost'].idxmin()]
        
        return {
            'best_casp': best_casp.to_dict(),
            'lowest_carbon': lowest_carbon.to_dict(),
            'highest_service': highest_service.to_dict(),
            'lowest_cost': lowest_cost.to_dict()
        }

if __name__ == "__main__":
    # Test the module
    frontiers = TradeOffFrontiers()
    
    # Add sample results
    frontiers.add_result('pharmacy', 150000, 96, 2000, 0.00064)
    frontiers.add_result('pharmacy', 120000, 94, 1800, 0.00078)
    frontiers.add_result('clothing', 50000, 88, 800, 0.00176)
    frontiers.add_result('clothing', 40000, 85, 600, 0.00213)
    frontiers.add_result('electronics', 100000, 95, 1500, 0.00095)
    
    # Calculate Pareto frontier
    print("📊 Pareto Frontier:")
    pareto = frontiers.calculate_pareto_frontier()
    print(pareto)
    
    # CASP ranking
    print("\n📈 CASP Ranking:")
    ranking = frontiers.calculate_casp_ranking()
    print(ranking[['package_type', 'casp_score', 'carbon_gco2', 'service_level']])
    
    # Recommendations
    print("\n💡 Recommendations:")
    recs = frontiers.get_recommendations()
    print(f"  Best CASP: {recs['best_casp']['package_type']} (CASP={recs['best_casp']['casp_score']:.6f})")
    print(f"  Lowest Carbon: {recs['lowest_carbon']['package_type']} ({recs['lowest_carbon']['carbon_gco2']:.0f} gCO2)")

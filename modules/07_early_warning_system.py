"""
OUTPUT 3: EARLY-WARNING INDICATORS
Disruption prediction model
Risk scoring system
Amplification indicators

Paper-aligned indicators (Section 4.8, Figure 7):
1. Supplier Concentration Index: >40% single-supplier dependence → disruption amplification
2. Geographic Clustering: >60% regional concentration → weather-related disruption correlation
3. Cold-Chain Fragility: Critical (e.g. pharmacy) with limited temperature buffer → spoilage risk
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix
from config.agent_mapping import get_agent_config
import warnings
warnings.filterwarnings('ignore')

class EarlyWarningSystem:
    """Early-warning system for supply chain disruptions."""
    
    def __init__(self, data_path: str = 'data/datasets/Delivery_Logistics.csv'):
        """Initialize with delivery logistics data."""
        self.data_path = data_path
        self.df = None
        self.delay_model = None
        self.preprocessor = None
        
    def load_data(self):
        """Load and prepare data."""
        self.df = pd.read_csv(self.data_path)
        
        # Create binary delay target
        self.df['is_delayed'] = (self.df['delayed'] == 'yes').astype(int)
        
        print(f"✓ Loaded {len(self.df)} records")
        print(f"  Delay rate: {self.df['is_delayed'].mean()*100:.1f}%")
    
    def train_delay_predictor(self):
        """Train model to predict delays."""
        # Prepare features
        drop_cols = ['delivery_id', 'delivery_time_hours', 'expected_time_hours', 'delayed']
        X = self.df.drop(columns=drop_cols + ['is_delayed', 'delivery_status'])
        y = self.df['is_delayed'].values
        
        # Define feature types
        categorical_features = [
            'delivery_partner', 'package_type', 'vehicle_type',
            'delivery_mode', 'region', 'weather_condition'
        ]
        numerical_features = ['distance_km', 'package_weight_kg', 'delivery_rating']
        
        # Preprocessing
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            ]
        )
        
        X_processed = self.preprocessor.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        self.delay_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.delay_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.delay_model.predict(X_test)
        print("\n📊 Delay Prediction Model Performance:")
        print(classification_report(y_test, y_pred, target_names=['On-Time', 'Delayed']))
        
        return self.delay_model
    
    def predict_delay_probability(self, route_dict: dict) -> float:
        """
        Predict delay probability for a route.
        
        Args:
            route_dict: Route features dictionary
        
        Returns:
            Delay probability (0-1)
        """
        if self.delay_model is None:
            self.load_data()
            self.train_delay_predictor()
        
        # Convert to DataFrame
        route_df = pd.DataFrame([route_dict])
        
        # Ensure all required columns exist
        required_cols = ['delivery_partner', 'package_type', 'vehicle_type', 'delivery_mode',
                        'region', 'weather_condition', 'distance_km', 'package_weight_kg', 'delivery_rating']
        
        for col in required_cols:
            if col not in route_df.columns:
                if col in ['distance_km', 'package_weight_kg', 'delivery_rating']:
                    route_df[col] = self.df[col].mean()
                else:
                    route_df[col] = self.df[col].mode()[0]
        
        # Preprocess
        X_processed = self.preprocessor.transform(route_df)
        
        # Predict probability
        delay_prob = self.delay_model.predict_proba(X_processed)[0][1]
        
        return delay_prob
    
    def calculate_risk_score(
        self,
        package_type: str,
        delay_probability: float,
        route_dict: dict = None
    ) -> Dict:
        """
        Calculate early-warning risk score.
        
        Risk Score = P(delay) × Impact Multiplier
        
        Args:
            package_type: Type of package
            delay_probability: Predicted delay probability
            route_dict: Route features (optional, for additional factors)
        
        Returns:
            Dictionary with risk score and breakdown
        """
        config = get_agent_config(package_type)
        impact_multiplier = config['impact_multiplier']
        
        # Base risk score
        risk_score = delay_probability * impact_multiplier
        
        # Additional risk factors
        risk_factors = []
        
        if route_dict:
            # Weather risk
            if route_dict.get('weather_condition') == 'stormy':
                risk_score *= 1.5
                risk_factors.append('Stormy weather (+50% risk)')
            
            # Distance risk
            distance = route_dict.get('distance_km', 0)
            weight = route_dict.get('package_weight_kg', 0)
            if distance > 250 and weight > 40:
                risk_score *= 1.3
                risk_factors.append('Long distance + heavy weight (+30% risk)')
            
            # Partner risk (if we have historical data)
            partner = route_dict.get('delivery_partner', '')
            if partner:
                partner_delay_rate = self.df[self.df['delivery_partner'] == partner]['is_delayed'].mean()
                if partner_delay_rate > 0.15:
                    risk_score *= 1.2
                    risk_factors.append(f'High-risk partner: {partner} (+20% risk)')
        
        # Risk level
        if risk_score > 5:
            risk_level = 'CRITICAL'
        elif risk_score > 3:
            risk_level = 'HIGH'
        elif risk_score > 1.5:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        return {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'delay_probability': delay_probability,
            'impact_multiplier': impact_multiplier,
            'package_type': package_type,
            'risk_factors': risk_factors,
            'alert_required': risk_score > 5
        }

    def compute_early_warning_indicators(
        self,
        package_type: str,
        route_dict: Optional[Dict] = None,
        portfolio_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Compute the three paper-aligned early-warning indicators for disruption amplification.

        Paper (Section 4.8): (1) Supplier Concentration Index, (2) Geographic Clustering,
        (3) Cold-Chain Fragility, each with quantified amplification risk.

        Args:
            package_type: Package type (e.g. pharmacy, clothing).
            route_dict: Optional route features (for context; not required for portfolio metrics).
            portfolio_df: Optional subset of deliveries to use as portfolio. If None, uses
                         self.df filtered by package_type.

        Returns:
            Dict with:
              - supplier_concentration_index: { value_pct, threshold_pct, exceeds, amplification_risk }
              - geographic_clustering: { value_pct, threshold_pct, exceeds, amplification_risk, weather_delay_correlation }
              - cold_chain_fragility: { is_fragile, reason, amplification_risk }
              - quantified_amplification_risk: overall score 0-1 or label
        """
        if self.df is None:
            self.load_data()
        df = self.df
        pkg = (package_type or '').lower().strip()
        portfolio = portfolio_df if portfolio_df is not None else df[df['package_type'].astype(str).str.lower() == pkg]
        if portfolio.empty:
            portfolio = df
        n = len(portfolio)

        # ---- 1. Supplier Concentration Index: >40% single-supplier dependence ----
        supplier_threshold_pct = 40.0
        partner_counts = portfolio['delivery_partner'].value_counts()
        top_share = float(partner_counts.iloc[0] / n) if n else 0.0
        top_partner = partner_counts.index[0] if len(partner_counts) else ''
        supplier_exceeds = bool(top_share * 100 > supplier_threshold_pct)
        # Quantified amplification risk: 0-1 scale when exceeds (e.g. 45% -> 0.25, 80% -> 1.0)
        supplier_amp = 0.0
        if supplier_exceeds:
            excess = (top_share * 100 - supplier_threshold_pct) / (100 - supplier_threshold_pct)
            supplier_amp = min(1.0, max(0.0, excess))

        supplier_concentration_index = {
            'name': 'Supplier Concentration Index',
            'value_pct': round(float(top_share * 100), 2),
            'threshold_pct': supplier_threshold_pct,
            'exceeds': supplier_exceeds,
            'dominant_partner': str(top_partner),
            'interpretation': f'>{supplier_threshold_pct}% single-supplier dependence' if supplier_exceeds else f'Within threshold (max {top_share*100:.1f}%)',
            'amplification_risk': round(supplier_amp, 3),
        }

        # ---- 2. Geographic Clustering: >60% regional concentration ----
        region_threshold_pct = 60.0
        region_counts = portfolio['region'].value_counts()
        top_region_share = float(region_counts.iloc[0] / n) if n else 0.0
        top_region = region_counts.index[0] if len(region_counts) else ''
        geo_exceeds = bool(top_region_share * 100 > region_threshold_pct)
        geo_amp = 0.0
        if geo_exceeds:
            excess = (top_region_share * 100 - region_threshold_pct) / (100 - region_threshold_pct)
            geo_amp = min(1.0, max(0.0, excess))
        # Weather-related disruption correlation: delay rate in worst region vs overall
        weather_delay_correlation = None
        if 'region' in portfolio.columns and 'is_delayed' in portfolio.columns and n > 0:
            overall_delay = portfolio['is_delayed'].mean()
            by_region = portfolio.groupby('region')['is_delayed'].agg(['mean', 'count'])
            by_region = by_region[by_region['count'] >= 10]
            if not by_region.empty and overall_delay > 0:
                max_region_delay = by_region['mean'].max()
                weather_delay_correlation = round(float(max_region_delay - overall_delay), 4)

        geographic_clustering = {
            'name': 'Geographic Clustering',
            'value_pct': round(float(top_region_share * 100), 2),
            'threshold_pct': region_threshold_pct,
            'exceeds': geo_exceeds,
            'dominant_region': str(top_region),
            'interpretation': f'>{region_threshold_pct}% regional concentration' if geo_exceeds else f'Within threshold (max {top_region_share*100:.1f}%)',
            'amplification_risk': round(geo_amp, 3),
            'weather_delay_correlation': weather_delay_correlation,
        }

        # ---- 3. Cold-Chain Fragility: critical/high_value + cold chain ----
        config = get_agent_config(package_type)
        cold_mult = config.get('cold_chain_multiplier', 1.0)
        tier = (config.get('agent') or config.get('tier_name') or '').lower()
        is_cold_chain = cold_mult > 1.0
        is_fragile = is_cold_chain and tier in ('critical', 'high_value')
        cold_amp = 0.5 if is_fragile else (0.2 if is_cold_chain else 0.0)
        reason = 'Critical/cold-chain (e.g. pharmacy): limited temperature buffer' if is_fragile else ('Cold chain present' if is_cold_chain else 'Ambient only')

        cold_chain_fragility = {
            'name': 'Cold-Chain Fragility',
            'is_fragile': is_fragile,
            'cold_chain_multiplier': cold_mult,
            'tier': config.get('tier_name', tier),
            'reason': reason,
            'interpretation': 'Higher spoilage risk during disruptions' if is_fragile else 'Lower spoilage risk',
            'amplification_risk': round(cold_amp, 3),
        }

        # ---- Quantified overall amplification risk ----
        overall_amp = (supplier_amp + geo_amp + cold_amp) / 3.0
        if overall_amp >= 0.6:
            amplification_label = 'HIGH'
        elif overall_amp >= 0.3:
            amplification_label = 'MEDIUM'
        else:
            amplification_label = 'LOW'

        return {
            'supplier_concentration_index': supplier_concentration_index,
            'geographic_clustering': geographic_clustering,
            'cold_chain_fragility': cold_chain_fragility,
            'quantified_amplification_risk': {
                'score': round(overall_amp, 3),
                'label': amplification_label,
            },
        }

    def identify_high_risk_routes(
        self,
        routes: List[Dict],
        package_type: str
    ) -> pd.DataFrame:
        """
        Identify high-risk routes from a list.
        
        Args:
            routes: List of route dictionaries
            package_type: Type of package
        
        Returns:
            DataFrame with risk scores
        """
        results = []
        
        for route in routes:
            delay_prob = self.predict_delay_probability(route)
            risk = self.calculate_risk_score(package_type, delay_prob, route)
            
            results.append({
                'route_id': route.get('route_id', 'Unknown'),
                'delay_probability': delay_prob,
                'risk_score': risk['risk_score'],
                'risk_level': risk['risk_level'],
                'alert_required': risk['alert_required'],
                'risk_factors': ', '.join(risk['risk_factors'])
            })
        
        df = pd.DataFrame(results)
        df = df.sort_values('risk_score', ascending=False)
        
        return df

if __name__ == "__main__":
    # Test the module
    ews = EarlyWarningSystem()
    ews.load_data()
    ews.train_delay_predictor()
    
    # Test prediction
    test_route = {
        'delivery_partner': 'shadowfax',
        'package_type': 'pharmacy',
        'vehicle_type': 'van',
        'delivery_mode': 'express',
        'region': 'west',
        'weather_condition': 'stormy',
        'distance_km': 300,
        'package_weight_kg': 45,
        'delivery_rating': 3
    }
    
    delay_prob = ews.predict_delay_probability(test_route)
    print(f"\n⚠️ Delay Probability: {delay_prob*100:.1f}%")
    
    # Calculate risk score
    risk = ews.calculate_risk_score('pharmacy', delay_prob, test_route)
    print(f"\n🚨 Risk Score: {risk['risk_score']:.2f} ({risk['risk_level']})")
    print(f"   Alert Required: {'YES' if risk['alert_required'] else 'NO'}")
    if risk['risk_factors']:
        print(f"   Risk Factors: {', '.join(risk['risk_factors'])}")

"""
MODULE 1a: PREDICTIVE ANALYTICS
GradientBoostingRegressor for predicting Cost, Carbon, On-time probability
Identifies where forecasts fail (weather, partner, etc.)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class PredictiveAnalytics:
    """Predictive analytics for supply chain forecasting."""
    
    def __init__(self, data_path: str = 'data/datasets/Delivery_Logistics.csv'):
        """Initialize with delivery logistics data."""
        self.data_path = data_path
        self.df = None
        self.cost_model = None
        self.carbon_model = None
        self.on_time_model = None
        self.preprocessor = None
        self.feature_names = None
        
    def load_data(self):
        """Load and prepare data."""
        self.df = pd.read_csv(self.data_path)
        
        # Create on-time percentage from delayed column
        self.df['on_time_pct'] = (self.df['delayed'] == 'no').astype(int) * 100
        
        # Calculate carbon from distance and vehicle
        vehicle_emissions = {
            'bike': 50, 'ev bike': 20, 'scooter': 60,
            'ev van': 150, 'van': 800, 'truck': 1200
        }
        self.df['carbon_gco2'] = self.df.apply(
            lambda row: row['distance_km'] * vehicle_emissions.get(row['vehicle_type'], 800),
            axis=1
        )
        
        print(f"✓ Loaded {len(self.df)} records")
    
    def prepare_features(self):
        """Prepare features for ML models.
        
        IMPORTANT: Only uses features available BEFORE delivery occurs.
        Removed leaky features: 'delayed', 'delivery_status' (these are outcomes, not inputs).
        Note: 'delivery_rating' is kept as it represents historical/expected rating, not post-delivery rating.
        """
        # Drop unnecessary columns AND leaky features (outcomes that shouldn't be inputs)
        drop_cols = ['delivery_id', 'delivery_time_hours', 'expected_time_hours']
        leaky_features = ['delayed', 'delivery_status']  # These are outcomes, not inputs!
        X = self.df.drop(columns=drop_cols + leaky_features + ['delivery_cost', 'carbon_gco2', 'on_time_pct'])
        
        # Define feature types - ONLY features known BEFORE delivery
        categorical_features = [
            'delivery_partner', 'package_type', 'vehicle_type',
            'delivery_mode', 'region', 'weather_condition'
            # Removed: 'delayed', 'delivery_status' (data leakage)
        ]
        numerical_features = ['distance_km', 'package_weight_kg', 'delivery_rating']
        
        # Preprocessing pipeline
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            ]
        )
        
        X_processed = self.preprocessor.fit_transform(X)
        self.feature_names = X.columns.tolist()
        
        return X_processed
    
    def train_models(self):
        """Train GradientBoosting models for cost, carbon, and on-time prediction."""
        X_processed = self.prepare_features()
        
        # Prepare targets
        y_cost = self.df['delivery_cost'].values
        y_carbon = self.df['carbon_gco2'].values
        y_on_time = self.df['on_time_pct'].values
        
        # Split data
        X_train, X_test, y_cost_train, y_cost_test = train_test_split(
            X_processed, y_cost, test_size=0.2, random_state=42
        )
        _, _, y_carbon_train, y_carbon_test = train_test_split(
            X_processed, y_carbon, test_size=0.2, random_state=42
        )
        _, _, y_on_time_train, y_on_time_test = train_test_split(
            X_processed, y_on_time, test_size=0.2, random_state=42
        )
        
        # Train Cost Model
        print("\n📊 Training Cost Prediction Model...")
        self.cost_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.cost_model.fit(X_train, y_cost_train)
        cost_pred = self.cost_model.predict(X_test)
        print(f"  R² Score: {r2_score(y_cost_test, cost_pred):.4f}")
        print(f"  MAE: ₹{mean_absolute_error(y_cost_test, cost_pred):.2f}")
        
        # Train Carbon Model
        # NOTE: Carbon is deterministic (distance × vehicle_emission_factor), so this model
        # learns a trivial formula. High R² is expected but not meaningful for ML evaluation.
        # In production, carbon would be calculated directly, not predicted.
        print("\n📊 Training Carbon Prediction Model...")
        print("  ⚠️  Note: Carbon is deterministic (distance × vehicle_factor), so high R² is expected.")
        self.carbon_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.carbon_model.fit(X_train, y_carbon_train)
        carbon_pred = self.carbon_model.predict(X_test)
        print(f"  R² Score: {r2_score(y_carbon_test, carbon_pred):.4f}")
        print(f"  MAE: {mean_absolute_error(y_carbon_test, carbon_pred):.0f} gCO2")
        
        # Train On-Time Model
        print("\n📊 Training On-Time Prediction Model...")
        self.on_time_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.on_time_model.fit(X_train, y_on_time_train)
        on_time_pred = self.on_time_model.predict(X_test)
        print(f"  R² Score: {r2_score(y_on_time_test, on_time_pred):.4f}")
        print(f"  MAE: {mean_absolute_error(y_on_time_test, on_time_pred):.2f}%")
        
        print("\n✅ All models trained successfully!")
    
    def predict(self, route_dict: dict) -> dict:
        """
        Predict cost, carbon, and on-time for a route.
        
        Args:
            route_dict: Dictionary with route features
        
        Returns:
            Dictionary with predictions
        """
        # Convert to DataFrame
        route_df = pd.DataFrame([route_dict])
        
        # Ensure all required columns exist (removed leaky features)
        required_cols = ['delivery_partner', 'package_type', 'vehicle_type', 'delivery_mode',
                        'region', 'weather_condition',
                        'distance_km', 'package_weight_kg', 'delivery_rating']
        
        for col in required_cols:
            if col not in route_df.columns:
                # Use mode/mean from training data
                if col in ['distance_km', 'package_weight_kg', 'delivery_rating']:
                    route_df[col] = self.df[col].mean()
                else:
                    route_df[col] = self.df[col].mode()[0]
        
        # Preprocess
        X_processed = self.preprocessor.transform(route_df)
        
        # Predict
        cost_pred = self.cost_model.predict(X_processed)[0]
        carbon_pred = self.carbon_model.predict(X_processed)[0]
        on_time_pred = self.on_time_model.predict(X_processed)[0]
        
        return {
            'predicted_cost': float(cost_pred),
            'predicted_carbon_gco2': float(carbon_pred),
            'predicted_on_time_pct': float(on_time_pred)
        }
    
    def identify_forecast_failures(self):
        """Identify where forecasts fail (weather, partner, etc.)."""
        X_processed = self.prepare_features()
        
        # Get predictions
        cost_pred = self.cost_model.predict(X_processed)
        on_time_pred = self.on_time_model.predict(X_processed)
        
        # Calculate errors
        cost_error = np.abs(self.df['delivery_cost'].values - cost_pred)
        on_time_error = np.abs(self.df['on_time_pct'].values - on_time_pred)
        
        # Add to dataframe
        self.df['cost_error'] = cost_error
        self.df['on_time_error'] = on_time_error
        
        # Analyze failures by feature
        failure_analysis = {}
        
        for feature in ['weather_condition', 'delivery_partner', 'region', 'vehicle_type']:
            if feature in self.df.columns:
                failures = self.df.groupby(feature).agg({
                    'cost_error': 'mean',
                    'on_time_error': 'mean',
                    'delayed': lambda x: (x == 'yes').sum()
                }).sort_values('on_time_error', ascending=False)
                
                failure_analysis[feature] = failures
        
        return failure_analysis

def create_predictors(data_path: str = 'data/datasets/Delivery_Logistics.csv'):
    """Create and return predictor functions for use in optimization."""
    analytics = PredictiveAnalytics(data_path)
    analytics.load_data()
    analytics.train_models()
    
    def cost_predictor(route_dict: dict) -> float:
        return analytics.predict(route_dict)['predicted_cost']
    
    def on_time_predictor(route_dict: dict) -> float:
        return analytics.predict(route_dict)['predicted_on_time_pct']
    
    return cost_predictor, on_time_predictor, analytics

if __name__ == "__main__":
    # Test the module
    analytics = PredictiveAnalytics()
    analytics.load_data()
    analytics.train_models()
    
    # Test prediction (removed leaky features from test route)
    test_route = {
        'delivery_partner': 'delhivery',
        'package_type': 'pharmacy',
        'vehicle_type': 'ev van',
        'delivery_mode': 'express',
        'region': 'west',
        'weather_condition': 'clear',
        'distance_km': 150,
        'package_weight_kg': 25,
        'delivery_rating': 4
    }
    
    predictions = analytics.predict(test_route)
    print("\n📊 Test Prediction:")
    print(f"  Cost: ₹{predictions['predicted_cost']:.2f}")
    print(f"  Carbon: {predictions['predicted_carbon_gco2']:.0f} gCO2")
    print(f"  On-Time: {predictions['predicted_on_time_pct']:.1f}%")
    
    # Analyze failures
    print("\n🔍 Forecast Failure Analysis:")
    failures = analytics.identify_forecast_failures()
    print("\nTop failure factors by weather:")
    print(failures['weather_condition'].head())

"""
MODULE 1c: VENDOR MANAGEMENT (Unsupervised Segmentation)
K-Means clustering on delivery partners
Identify: Fast/Cheap/Reliable/Unreliable clusters
Map roles: Premium carriers vs budget carriers
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class VendorSegmentation:
    """K-Means clustering for vendor segmentation."""
    
    def __init__(self, data_path: str = 'data/datasets/Delivery_Logistics.csv'):
        """Initialize with delivery logistics data."""
        self.data_path = data_path
        self.df = None
        self.scaler = StandardScaler()
        self.kmeans = None
        self.cluster_labels = None
        self.vendor_features = None
        
    def load_data(self):
        """Load and prepare data."""
        self.df = pd.read_csv(self.data_path)
        print(f"✓ Loaded {len(self.df)} records")
    
    def prepare_vendor_features(self):
        """Calculate vendor performance metrics."""
        vendor_stats = self.df.groupby('delivery_partner').agg({
            'delivery_cost': 'mean',
            'distance_km': 'mean',
            'delayed': lambda x: (x == 'yes').mean(),  # Delay rate
            'delivery_rating': 'mean',
            'delivery_id': 'count'  # Volume
        }).rename(columns={
            'delivery_cost': 'avg_cost',
            'distance_km': 'avg_distance',
            'delayed': 'delay_rate',
            'delivery_rating': 'avg_rating',
            'delivery_id': 'volume'
        })
        
        # Calculate on-time percentage
        vendor_stats['on_time_pct'] = (1 - vendor_stats['delay_rate']) * 100
        
        # Calculate cost per km
        vendor_stats['cost_per_km'] = vendor_stats['avg_cost'] / vendor_stats['avg_distance']
        
        # Calculate reliability score (combination of on-time and rating)
        vendor_stats['reliability_score'] = (
            vendor_stats['on_time_pct'] * 0.6 + vendor_stats['avg_rating'] * 20 * 0.4
        )
        
        self.vendor_features = vendor_stats
        
        print(f"\n✓ Calculated features for {len(vendor_stats)} vendors")
        print("\nVendor Statistics:")
        print(vendor_stats)
        
        return vendor_stats
    
    def cluster_vendors(self, n_clusters: int = 4):
        """
        Perform K-Means clustering on vendors.
        
        Args:
            n_clusters: Number of clusters (default: 4 for Fast/Cheap/Reliable/Unreliable)
        """
        if self.vendor_features is None:
            self.prepare_vendor_features()
        
        # Select features for clustering
        features = ['avg_cost', 'delay_rate', 'avg_rating', 'on_time_pct', 'reliability_score']
        X = self.vendor_features[features].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # K-Means clustering
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.cluster_labels = self.kmeans.fit_predict(X_scaled)
        
        # Add cluster labels to vendor features
        self.vendor_features['cluster'] = self.cluster_labels
        
        print(f"\n✓ Clustered vendors into {n_clusters} groups")
        
        return self.cluster_labels
    
    def interpret_clusters(self) -> dict:
        """Interpret clusters and assign roles."""
        if self.vendor_features is None or 'cluster' not in self.vendor_features.columns:
            self.cluster_vendors()
        
        cluster_analysis = {}
        
        for cluster_id in sorted(self.vendor_features['cluster'].unique()):
            cluster_vendors = self.vendor_features[self.vendor_features['cluster'] == cluster_id]
            
            # Calculate cluster characteristics
            avg_cost = cluster_vendors['avg_cost'].mean()
            avg_on_time = cluster_vendors['on_time_pct'].mean()
            avg_rating = cluster_vendors['avg_rating'].mean()
            avg_reliability = cluster_vendors['reliability_score'].mean()
            
            # Determine cluster type
            if avg_on_time > 90 and avg_rating > 4.0:
                cluster_type = "Premium/Reliable"
            elif avg_cost < self.vendor_features['avg_cost'].median():
                cluster_type = "Budget/Cheap"
            elif avg_on_time < 85:
                cluster_type = "Unreliable"
            else:
                cluster_type = "Balanced"
            
            cluster_analysis[cluster_id] = {
                'type': cluster_type,
                'vendors': cluster_vendors.index.tolist(),
                'avg_cost': avg_cost,
                'avg_on_time': avg_on_time,
                'avg_rating': avg_rating,
                'avg_reliability': avg_reliability,
                'count': len(cluster_vendors)
            }
        
        return cluster_analysis
    
    def get_vendor_recommendation(self, package_type: str, priority: str = 'reliability') -> str:
        """
        Get vendor recommendation based on package type and priority.
        
        Args:
            package_type: Type of package
            priority: 'reliability', 'cost', or 'speed'
        
        Returns:
            Recommended vendor name
        """
        if self.vendor_features is None or 'cluster' not in self.vendor_features.columns:
            self.cluster_vendors()
            self.interpret_clusters()
        
        # Filter by package type performance if available
        package_vendors = self.df[self.df['package_type'] == package_type].groupby('delivery_partner').agg({
            'delayed': lambda x: (x == 'yes').mean(),
            'delivery_cost': 'mean',
            'delivery_rating': 'mean'
        })
        
        if priority == 'reliability':
            # Find vendor with best on-time for this package type
            package_vendors['on_time_pct'] = (1 - package_vendors['delayed']) * 100
            best_vendor = package_vendors['on_time_pct'].idxmax()
        elif priority == 'cost':
            best_vendor = package_vendors['delivery_cost'].idxmin()
        else:  # speed or default
            best_vendor = package_vendors['delivery_rating'].idxmax()
        
        return best_vendor

if __name__ == "__main__":
    # Test the module
    segmentation = VendorSegmentation()
    segmentation.load_data()
    segmentation.prepare_vendor_features()
    segmentation.cluster_vendors(n_clusters=4)
    
    # Interpret clusters
    print("\n📊 Vendor Cluster Analysis:")
    clusters = segmentation.interpret_clusters()
    for cluster_id, info in clusters.items():
        print(f"\nCluster {cluster_id}: {info['type']}")
        print(f"  Vendors: {', '.join(info['vendors'])}")
        print(f"  Avg Cost: ₹{info['avg_cost']:.2f}")
        print(f"  Avg On-Time: {info['avg_on_time']:.1f}%")
        print(f"  Avg Rating: {info['avg_rating']:.2f}")
    
    # Get recommendations
    print("\n💡 Vendor Recommendations:")
    print(f"  Pharmacy (reliability): {segmentation.get_vendor_recommendation('pharmacy', 'reliability')}")
    print(f"  Clothing (cost): {segmentation.get_vendor_recommendation('clothing', 'cost')}")

"""
Sourcing tools: wrap Module 01, 02, data/carriers, data/routes, config/agent_mapping for Sourcing Agent.
Cascade: Distance = routes.py → OpenRouteService → Web search → Haversine.
         Pricing = Web search → carriers.py (local fallback).
"""

from typing import Dict, List, Any, Optional, Tuple
import sys
import os
# Ensure code root is on path when running from project
_code_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _code_root not in sys.path:
    sys.path.insert(0, _code_root)

from config.routes import get_distance, get_route_info, is_metro_to_metro, get_city, _haversine_distance
from config.carriers import get_all_quotes, CARRIERS, CarrierProfile
from config.agent_mapping import get_agent_config

# Map carrier quote identifier to delivery_partner string expected by Module 01 (CSV column)
CARRIER_TO_PARTNER = {
    'DEL-EXP': 'delhivery', 'DEL-STD': 'delhivery',
    'BLU-EXP': 'blue dart', 'DHL-EXP': 'dhl', 'SHDW': 'shadowfax',
    'ECOM': 'ecom express', 'XBEES': 'xpressbees', 'EKART': 'ekart',
    'DTDC-STD': 'dtdc', 'GATI': 'gati',
}


def get_distance_cascade(origin: str, destination: str) -> Tuple[float, str]:
    """
    Get distance in km: Step 1 local routes.py → Step 2 OpenRouteService API
    → Step 3 Web search → Step 4 Haversine (if both cities in CITIES).
    Returns (distance_km, source).
    """
    origin = str(origin).lower().strip()
    destination = str(destination).lower().strip()
    # Step 1: Local routes.py (22 known cities)
    dist = get_distance(origin, destination)
    if dist is not None:
        return (float(dist), "local_routes")
    try:
        from config.api_config import API_KEYS, WEB_SEARCH_ENABLED
    except ImportError:
        API_KEYS = {}
        WEB_SEARCH_ENABLED = False
    # Step 2: OpenRouteService API
    key = (API_KEYS or {}).get("openrouteservice", "")
    if key:
        try:
            from data.apis.distance_api import get_distance_km as api_distance
            d = api_distance(origin, destination, key)
            if d is not None:
                return (float(d), "openrouteservice")
        except Exception:
            pass
    # Step 3: Web search
    if WEB_SEARCH_ENABLED:
        try:
            from tools.web_search import web_search, parse_distance_from_text
            raw = web_search(f"distance {origin} to {destination} km")
            d = parse_distance_from_text(raw) if raw else None
            if d is not None:
                return (float(d), "web_search")
        except Exception:
            pass
    # Step 4: Haversine (if both in CITIES)
    origin_city = get_city(origin)
    dest_city = get_city(destination)
    if origin_city and dest_city:
        d = _haversine_distance(
            origin_city.latitude, origin_city.longitude,
            dest_city.latitude, dest_city.longitude,
        ) * 1.3  # road factor
        return (round(float(d), 2), "haversine")
    return (150.0, "default")


def get_carrier_rates_cascade(carrier: str, origin: str, dest: str, weight_kg: float) -> Optional[Dict[str, Any]]:
    """
    Get carrier rates: Step 1 Web search for current pricing → Step 2 local carriers.py fallback.
    Returns dict with cost_inr, etc., or None; callers can fall back to get_all_quotes.
    """
    try:
        from config.api_config import WEB_SEARCH_ENABLED
        from tools.web_search import web_search
    except ImportError:
        return None
    if WEB_SEARCH_ENABLED:
        try:
            query = f"{carrier} shipping rates {origin} to {dest} {weight_kg}kg 2024"
            raw = web_search(query)
            if raw:
                import re
                m = re.search(r"[\d,]+(?:\.\d+)?", raw.replace(",", ""))
                if m:
                    try:
                        cost = float(m.group(0).replace(",", ""))
                        if 0 < cost < 1e6:
                            return {"cost_inr": cost, "source": "web_search"}
                    except (ValueError, TypeError):
                        pass
        except Exception:
            pass
    return None


def lookup_route(origin: str, destination: str) -> Optional[Dict]:
    """Get route info (distance_km, region, is_metro_to_metro). Uses get_distance_cascade when route unknown."""
    info = get_route_info(origin, destination)
    if info is not None:
        return info
    dist_km, _ = get_distance_cascade(origin, destination)
    origin_city = get_city(str(origin).lower())
    dest_city = get_city(str(destination).lower())
    region = "west"
    if origin_city:
        region = getattr(origin_city, "region", region)
    elif dest_city:
        region = getattr(dest_city, "region", region)
    is_metro = bool(origin_city and dest_city and getattr(origin_city, "is_metro", False) and getattr(dest_city, "is_metro", False))
    return {
        "distance_km": dist_km,
        "region": region,
        "is_metro_to_metro": is_metro,
    }


def get_package_rules(package_type: str) -> Dict:
    """Get SLA and rules for package type from agent_mapping."""
    config = get_agent_config(package_type)
    return {
        'on_time_threshold': config['on_time_threshold'],
        'on_time_threshold_pct': config['on_time_threshold'] * 100,
        'cold_chain_multiplier': config.get('cold_chain_multiplier', 1.0),
        'agent': config['agent'],
        'tier_name': config.get('tier_name', ''),
    }


def get_carrier_quotes_for_features(
    features: Dict,
    risk_assessment: Optional[Dict] = None,
) -> List[Dict]:
    """
    Get carrier quotes from data/carriers using features (distance, weight, package_type, region, etc.).
    Uses min_on_time_pct from package rules if not overridden by risk.
    """
    package_type = features.get('package_type') or 'clothing'
    rules = get_package_rules(package_type)
    min_on_time = rules['on_time_threshold_pct']
    if risk_assessment and risk_assessment.get('recommended_buffer_days', 0) > 0:
        min_on_time = max(min_on_time, 95.0)  # Stricter when risk is high
    
    distance_km = features.get('distance_km')
    if distance_km is None and features.get('origin') and features.get('destination'):
        route_info = lookup_route(str(features['origin']), str(features['destination']))
        distance_km = route_info['distance_km'] if route_info else 150
    
    distance_km = distance_km or 150
    weight_kg = features.get('package_weight_kg') or 25
    origin_region = _get_region(features)
    is_express = (features.get('delivery_mode') or '').lower() in ('same day', 'express')
    cold_chain = rules.get('cold_chain_multiplier', 1.0) > 1.0
    weather = (features.get('weather_condition') or 'clear').lower()
    if weather == 'stormy':
        weather = 'storm'
    
    is_metro = False
    if features.get('origin') and features.get('destination'):
        is_metro = is_metro_to_metro(str(features['origin']), str(features['destination']))
    
    quotes = get_all_quotes(
        distance_km=distance_km,
        weight_kg=weight_kg,
        package_type=package_type,
        origin_region=origin_region,
        is_express=is_express,
        requires_cold_chain=cold_chain,
        is_metro_to_metro=is_metro,
        weather_condition=weather,
        min_on_time_pct=min_on_time - 5,  # Allow slightly lower to get more options
    )
    return quotes


def _get_region(features: Dict) -> str:
    if features.get('region'):
        return str(features['region']).lower()
    if features.get('origin'):
        from config.routes import get_city
        c = get_city(str(features['origin']))
        if c:
            return c.region
    return 'west'


def predict_performance(analytics, route_dict: Dict) -> Dict[str, float]:
    """Use Module 01 to predict cost, carbon, on_time_pct for a route dict."""
    pred = analytics.predict(route_dict)
    return {
        'predicted_cost': pred['predicted_cost'],
        'predicted_carbon_gco2': pred['predicted_carbon_gco2'],
        'predicted_on_time_pct': pred['predicted_on_time_pct'],
    }


def get_carrier_cluster_name(vendor_segmentation, delivery_partner: str) -> Optional[str]:
    """Get cluster label for a delivery partner from Module 02."""
    if vendor_segmentation.vendor_features is None or 'cluster' not in vendor_segmentation.vendor_features.columns:
        return None
    if delivery_partner not in vendor_segmentation.vendor_features.index:
        return None
    cluster_id = vendor_segmentation.vendor_features.loc[delivery_partner, 'cluster']
    clusters = vendor_segmentation.interpret_clusters()
    for cid, info in clusters.items():
        if info.get('cluster_id') == cluster_id:
            return info.get('type', str(cluster_id))
    return str(cluster_id)


def filter_options_by_sla(options: List[Dict], on_time_threshold_pct: float) -> List[Dict]:
    """Filter options to those meeting SLA (on_time >= threshold)."""
    return [o for o in options if (o.get('predicted_on_time_pct') or o.get('estimated_on_time_pct') or 0) >= on_time_threshold_pct]


def build_route_options_from_quotes(
    quotes: List[Dict],
    features: Dict,
    distance_km: float,
) -> List[Dict]:
    """
    Build route option dicts for the orchestrator/optimizer from carrier quotes.
    Each route must have keys required by Module 01 predict(): delivery_partner, package_type,
    vehicle_type, delivery_mode, region, weather_condition,
    distance_km, package_weight_kg, delivery_rating.
    Note: 'delayed' and 'delivery_status' removed (data leakage - these are outcomes, not inputs).
    """
    package_type = (features.get('package_type') or 'clothing').lower()
    region = _get_region(features)
    weather = (features.get('weather_condition') or 'clear').lower()
    weight_kg = features.get('package_weight_kg') or 25
    delivery_mode = (features.get('delivery_mode') or 'two day').lower()
    
    # Vehicle type from carrier quote or default
    vehicle_by_partner = {
        'delhivery': 'ev van', 'blue dart': 'van', 'dhl': 'truck',
        'shadowfax': 'bike', 'ecom express': 'van', 'xpressbees': 'van',
        'ekart': 'ev van', 'dtdc': 'van', 'gati': 'truck',
    }
    
    routes = []
    for i, q in enumerate(quotes):
        partner = CARRIER_TO_PARTNER.get(q.get('carrier_code', ''), q.get('carrier_name', 'delhivery').lower().split()[0])
        if partner not in vehicle_by_partner:
            partner = 'delhivery'
        vehicle = vehicle_by_partner.get(partner, 'van')
        
        route = {
            'route_id': q.get('carrier_code') or f"Route_{i}",
            'distance_km': distance_km,
            'package_weight_kg': weight_kg,
            'delivery_rating': 4,  # Historical/expected rating (not post-delivery)
            'vehicle_type': vehicle,
            'delivery_partner': partner,
            'delivery_mode': delivery_mode,
            'region': region,
            'weather_condition': weather,
            'package_type': package_type,
            # Removed: 'delayed', 'delivery_status' (data leakage - these are outcomes)
        }
        routes.append(route)
    return routes

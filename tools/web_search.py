"""
Web search wrapper for Risk/Sourcing fallback.
Step 3 in cascades when APIs fail. Optional: Strands web_search or DuckDuckGo.
"""

from typing import Optional
import re


def web_search(query: str, enabled: bool = True) -> Optional[str]:
    """
    Run a web search and return a short snippet (for parsing weather/distance/pricing).
    When Strands is available, can use strands.tools web_search; otherwise try DuckDuckGo HTML or return None.
    """
    if not enabled or not query or not query.strip():
        return None
    try:
        from strands.tools import web_search as strands_search
        result = strands_search(query)
        return str(result) if result else None
    except (ImportError, Exception):
        return None


def parse_distance_from_text(text: Optional[str]) -> Optional[float]:
    """Extract distance in km from search snippet (e.g. 'Mumbai to Delhi 1400 km')."""
    if not text:
        return None
    m = re.search(r"\b(\d+(?:,\d{3})*(?:\.\d+)?)\s*km\b", text.replace(",", ""), re.IGNORECASE)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except (ValueError, TypeError):
            pass
    m = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:kilometers?|kms?)\b", text, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except (ValueError, TypeError):
            pass
    return None


def parse_cost_from_text(text: Optional[str]) -> Optional[float]:
    """Extract cost in INR from search snippet (e.g. '₹2100', 'Rs 1500', 'INR 2000')."""
    if not text:
        return None
    # ₹ or Rs or INR followed by optional space and number (with optional commas)
    for pattern in [
        r"₹\s*(\d+(?:,\d{3})*(?:\.\d+)?)",
        r"Rs?\.?\s*(\d+(?:,\d{3})*(?:\.\d+)?)",
        r"INR\s*(\d+(?:,\d{3})*(?:\.\d+)?)",
        r"(?:cost|rate|price)[:\s]*(\d+(?:,\d{3})*(?:\.\d+)?)",
    ]:
        m = re.search(pattern, text.replace(",", ""), re.IGNORECASE)
        if m:
            try:
                v = float(m.group(1).replace(",", ""))
                if 0 < v < 1e7:
                    return v
            except (ValueError, TypeError):
                pass
    return None


def parse_weather_from_text(text: Optional[str]) -> Optional[str]:
    """Map snippet to our weather_condition: clear, rainy, cold, hot, foggy, stormy."""
    if not text:
        return None
    t = text.lower()
    if any(x in t for x in ["storm", "thunder", "cyclone"]):
        return "stormy"
    if any(x in t for x in ["rain", "drizzle", "shower"]):
        return "rainy"
    if any(x in t for x in ["fog", "mist", "haze"]):
        return "foggy"
    if any(x in t for x in ["hot", "heat", "40", "45"]):
        return "hot"
    if any(x in t for x in ["cold", "snow", "freez"]):
        return "cold"
    if any(x in t for x in ["clear", "sunny", "partly"]):
        return "clear"
    return None

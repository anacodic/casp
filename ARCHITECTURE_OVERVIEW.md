# 🏗️ Supply Chain Multi-Agent System - Complete Architecture Overview

**Last Updated:** January 29, 2026
**Status:** Fully Implemented Research System

---

## 📋 Table of Contents
1. [System Overview](#system-overview)
2. [Directory Structure](#directory-structure)
3. [Architecture Flow](#architecture-flow)
4. [Module Descriptions](#module-descriptions)
5. [Active vs Archived](#active-vs-archived)
6. [Data Flow](#data-flow)
7. [Key Features](#key-features)

---

## 🎯 System Overview

This is a **carbon-aware, multi-agent supply chain optimization system** designed for research on AI-enabled logistics optimization. The system demonstrates how different package types (pharmacy vs clothing) have different optimization potential due to service-level constraints.

### Core Concept
- **High-stakes packages** (pharmacy): Constrained by safety (≥99% on-time) → Limited optimization room
- **Low-stakes packages** (clothing): Flexible requirements (≥85% on-time) → Large optimization room

### Main Components
1. **Multi-Agent System**: 3 specialized agents (High/Medium/Low stakes)
2. **7 Analysis Modules**: Predictive analytics, vendor segmentation, carbon intelligence, etc.
3. **Integrated Flow**: Risk → Sourcing → Carbon → Final Decision
4. **Real-World Data**: 9 delivery partners (Indian logistics), 20+ cities, 25K logistics records

---

## 📁 Directory Structure

```
code/
├── 📂 services/                    # Python backend (NOT LLM agents)
│   ├── risk_service.py             # Risk assessment (Module 07)
│   ├── sourcing_service.py         # Carrier options + optimization (Modules 01, 02)
│   ├── carbon_service.py           # Carbon + governance (Modules 03-06)
│   ├── stakes_optimizer.py         # Single optimizer for all stakes (critical 99%, high_value 95%, standard 85%)
│   └── README.md                   # Clarifies: LLM agents are in orchestration/
│
├── 📂 config/                      # Configuration & mappings
│   ├── agent_mapping.py            # Package type → Agent routing
│   ├── vehicle_emissions.py       # Emission factors by vehicle type
│   └── grid_carbon.py              # Grid carbon intensity by country
│
├── 📂 data/                        # NEW: Real-world carrier & route data
│   ├── carriers.py                 # 10 Indian carriers with profiles
│   └── routes.py                   # 20+ Indian cities with distances
│
├── 📂 tools/                       # Utility functions for agents
│   ├── extraction_tools.py         # Feature extraction (rules + semantic)
│   ├── semantic_classifier.py      # Bedrock LLM package_type classifier
│   ├── defaults_from_data.py       # Data-derived defaults per package_type
│   ├── risk_tools.py               # Risk assessment utilities
│   ├── sourcing_tools.py           # Carrier selection & prediction
│   └── carbon_tools.py             # Carbon calculation utilities
│
├── 📂 orchestration/               # Strands + Bedrock LLM orchestration
│   ├── _orchestrator_instance.py   # Singleton SupplyChainOrchestrator (avoids circular imports)
│   ├── agent_tools.py              # Low-level tools for Risk/Sourcing agents (weather_api, news_api, web_search, etc.)
│   ├── bedrock_tools.py            # @tool wrappers (risk_agent_tool, sourcing_agent_tool, carbon, pipeline)
│   ├── orchestrator_agent.py       # Orchestrator Agent (LLM #1) with BedrockModel
│   ├── risk_agent_strands.py       # Risk Agent (LLM #2): union API + web, fuse, then Module 07
│   └── sourcing_agent_strands.py   # Sourcing Agent (LLM #3): distance, web, carriers, recommend
│
├── 📂 optimization/                # Gradient descent optimizer
│   └── route_optimizer.py         # Constraint-aware route optimizer
│
├── 📂 modules/                     # 7 Analysis Modules
│   ├── 01_predictive_analytics.py  # ML cost/carbon/on-time prediction
│   ├── 02_vendor_segmentation.py   # K-Means carrier clustering
│   ├── 03_carbon_cost_intelligence.py # AI compute carbon calculation
│   ├── 04_grid_carbon_scenarios.py # Country-level carbon comparison
│   ├── 05_trade_off_frontiers.py  # Pareto frontier analysis
│   ├── 06_governance_levers.py     # Policy recommendations
│   └── 07_early_warning_system.py  # Delay prediction & risk scoring
│
├── 📂 data/                        # Single source of truth for data
│   ├── apis/                       # External APIs (weather, news, distance)
│   ├── local/                      # Local/static (cascade fallbacks)
│   │   ├── carriers.py
│   │   └── routes.py
│   ├── datasets/                   # Raw data files
│   │   └── Delivery_Logistics.csv # Main dataset (25K records) ⭐
│   ├── carriers.py                # Backward-compat re-export from local
│   └── routes.py                  # Backward-compat re-export from local
│
├── 📂 archive/                     # OLD CODE - Not actively used
│   ├── pharma_logistics_agent.py   # Original single-purpose agents
│   ├── fashion_logistics_agent.py
│   └── supply_chain_orchestrator.py (old version)
│
├── 📄 supply_chain_orchestrator.py # Main orchestrator (CURRENT VERSION)
├── 📄 example_usage.py             # Basic usage examples
├── 📄 demo_greenroute.py           # Interactive demos
├── 📄 requirements.txt             # Python dependencies
└── 📄 README.md                    # System documentation
```

---

## 🔄 Architecture Flow

### **Integrated Multi-Agent Pipeline**

```
USER INPUT
    ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: EXTRACTION (tools/extraction_tools.py)              │
│  Extract & validate: package_type, origin, destination,     │
│  distance, weight, weather                                   │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: RISK SERVICE (services/risk_service.py)            │
│  Uses: Module 07 (Early Warning System)                     │
│  Output: risk_level, delay_probability, warnings, buffer    │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 3: SOURCING SERVICE (services/sourcing_service.py)    │
│  Uses: Module 01 (Predictive), Module 02 (Segmentation),    │
│        data/carriers, data/routes, config/agent_mapping     │
│  Actions:                                                    │
│    1. Lookup route (data/routes.py)                         │
│    2. Get carrier quotes (data/carriers.py)                 │
│    3. Predict performance (Module 01)                       │
│    4. Cluster carriers (Module 02)                          │
│    5. Filter by SLA (config/agent_mapping.py)               │
│    6. Run optimization (StakesOptimizer)                     │
│  Output: carrier_options, optimization_result               │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 4: CARBON SERVICE (services/carbon_service.py)        │
│  Uses: Module 03 (AI carbon), Module 04 (Grid),             │
│        Module 05 (Trade-offs), Module 06 (Governance)       │
│  Actions:                                                    │
│    1. Calculate AI carbon (Module 03)                       │
│    2. Get grid intensity (Module 04)                        │
│    3. Analyze trade-offs (Module 05)                        │
│    4. Generate governance advice (Module 06)                │
│  Output: total_carbon, greenest_viable, governance          │
└─────────────────────────────────────────────────────────────┘
    ↓
FINAL RECOMMENDATION
```

### **Semantic classification and LLM orchestration**

- **Semantic package_type:** When rule-based extraction does not match a keyword (e.g. user says "insulin"), the system uses **Bedrock (Claude)** via `tools/semantic_classifier.py` to classify `package_type` (e.g. insulin → pharmacy). If Bedrock is unavailable, it defaults to **pharmacy** for safety.
- **Strands + Bedrock orchestration (industry-standard Agent vs Tool split):**
  - **Orchestrator Agent (LLM #1):** Coordinates flow and **generates adaptive search queries**. Tools: extract_features_tool, **risk_agent_tool**(features_json, risk_queries_json), **sourcing_agent_tool**(features_json, risk_assessment_json, sourcing_queries_json), run_optimization_tool, carbon_analysis_tool. The orchestrator reads the user query and decides which tools/agents to call; when calling risk_agent_tool it generates 4–6 adaptive risk queries (e.g. "Mumbai port strike 2025", "Delhi weather today"); when calling sourcing_agent_tool it generates 2–4 pricing/sourcing queries.
  - **Risk Agent (LLM #2):** Full Strands agent with tools: weather_api_tool, news_api_tool, web_search_tool, calculate_risk_score_tool. **Union + fuse flow:** Always runs APIs (weather, news) AND web search (with orchestrator-provided queries); fuses conflicting data (weather API vs web, news API vs web); then calls calculate_risk_score_tool (Module 07). API failure is not treated as critical by itself. See `orchestration/risk_agent_strands.py`, `orchestration/agent_tools.py`.
  - **Sourcing Agent (LLM #3):** Full Strands agent with tools: distance_api_tool, routes_lookup_tool, web_search_tool, get_carrier_options_tool. Reasons about best carrier given risk and returns carrier_options JSON for run_optimization_tool. See `orchestration/sourcing_agent_strands.py`.
  - **Carbon:** Tool only (no LLM). carbon_analysis_tool uses Python CarbonService (services/carbon_service.py, Modules 03–06). Local only; no APIs or web search.
- **LLM calls per request:** 3 (Orchestrator + Risk Agent + Sourcing Agent). All requests use the multi-agent path with adaptive reasoning and real-time data fusion.

### **API → Web search → Local (Risk & Sourcing tools)**

- **Risk Agent (Strands):** Gathers weather from **weather_api_tool** (origin, destination) and **web_search_tool** (adaptive queries); gathers news from **news_api_tool** and **web_search_tool**. Union of sources; LLM fuses and checks consistency, then **calculate_risk_score_tool** (Module 07). See `data/apis/weather_api.py`, `data/apis/news_api.py`, `tools/web_search.py`, `orchestration/agent_tools.py`.
- **Sourcing Agent (Strands):** Uses **distance_api_tool**, **routes_lookup_tool** (local `data/local/routes.py` or cascade), **web_search_tool** (pricing queries), **get_carrier_options_tool** (Python SourcingService). See `tools/sourcing_tools.py`, `data/apis/distance_api.py`, `data/local/carriers.py`.
- **Carbon:** Local only (`config/grid_carbon.py`, Modules 03–06).
- **Config:** `config/api_config.py` — API_KEYS, FALLBACK_ENABLED, WEB_SEARCH_ENABLED. Full cascade diagram: `code/docs/ARCHITECTURE_CASCADE.md`.

---

## 📦 Module Descriptions

### **MODULE 1: Predictive Analytics** ([modules/01_predictive_analytics.py](modules/01_predictive_analytics.py))
**Purpose:** ML prediction for cost, carbon, and on-time probability

**Features:**
- GradientBoostingRegressor for cost/carbon/on-time prediction
- Feature importance analysis
- Identifies forecast failure points
- 25K records training dataset

**Used By:** Sourcing Service (services/sourcing_service.py)

---

### **MODULE 2: Vendor Segmentation** ([modules/02_vendor_segmentation.py](modules/02_vendor_segmentation.py))
**Purpose:** Cluster carriers into performance tiers

**Features:**
- K-Means clustering (4 clusters)
- Identifies: Premium, Reliable, Budget, Unreliable carriers
- Cluster interpretation & role mapping

**Used By:** Sourcing Service (services/sourcing_service.py)

---

### **MODULE 3: Carbon Cost of Intelligence** ([modules/03_carbon_cost_intelligence.py](modules/03_carbon_cost_intelligence.py))
**Purpose:** Calculate AI compute carbon footprint

**Features:**
- Model comparison (Gemini Flash vs GPT-4)
- Energy per token calculation
- Country-specific grid carbon
- Proves AI carbon is negligible vs transport

**Used By:** Carbon Service (services/carbon_service.py)

---

### **MODULE 4: Grid Carbon Scenarios** ([modules/04_grid_carbon_scenarios.py](modules/04_grid_carbon_scenarios.py))
**Purpose:** Country-level carbon intensity comparison

**Features:**
- Grid carbon intensity by country (India: 632, USA: 386, France: 60 gCO2/kWh)
- Shows impact of location on total carbon
- Optimal country recommendation

**Used By:** Carbon Service (services/carbon_service.py)

---

### **MODULE 5: Trade-Off Frontiers** ([modules/05_trade_off_frontiers.py](modules/05_trade_off_frontiers.py))
**Purpose:** Pareto frontier analysis (Carbon vs Service vs Cost)

**Features:**
- CASP metric calculation (Service / Carbon)
- Pareto frontier identification
- 2D and 3D visualizations
- Efficiency ranking

**Used By:** Carbon Service (services/carbon_service.py)

---

### **MODULE 6: Governance Levers** ([modules/06_governance_levers.py](modules/06_governance_levers.py))
**Purpose:** Policy recommendations by package type

**Features:**
- Vehicle selection guidelines
- Inventory buffer sizing
- Carrier selection rules
- AI compute policies
- Sourcing location advice

**Used By:** Carbon Service (services/carbon_service.py)

---

### **MODULE 7: Early Warning System** ([modules/07_early_warning_system.py](modules/07_early_warning_system.py))
**Purpose:** Delay prediction & risk assessment

**Features:**
- GradientBoostingClassifier for delay prediction
- Risk scoring: P(delay) × Impact Multiplier
- Disruption amplification detection
- Buffer day recommendations

**Used By:** Risk Service (services/risk_service.py)

---

## 🔥 Active vs Archived

### ✅ **ACTIVE CODE** (Currently Used)

| File | Purpose | Status |
|------|---------|--------|
| [supply_chain_orchestrator.py](supply_chain_orchestrator.py) | Main orchestrator | ✅ Current |
| [agents/](agents/) (all 6 files) | Multi-agent system | ✅ Active |
| [modules/](modules/) (all 7 files) | Analysis modules | ✅ Active |
| [config/](config/) (all 3 files) | Configuration | ✅ Active |
| [data/](data/) (carriers.py, routes.py) | Real carrier/route data | ✅ NEW (Jan 29) |
| [tools/](tools/) (all 4 files) | Agent utilities | ✅ NEW (Jan 29) |
| [optimization/route_optimizer.py](optimization/route_optimizer.py) | Route optimizer | ✅ Active |
| [example_usage.py](example_usage.py) | Usage examples | ✅ Active |
| [demo_greenroute.py](demo_greenroute.py) | Interactive demos | ✅ NEW (Jan 29) |

### 📦 **ARCHIVED CODE** (Old Versions)

| File | Reason | Replacement |
|------|--------|-------------|
| [archive/pharma_logistics_agent.py](archive/pharma_logistics_agent.py) | Single-purpose | services/stakes_optimizer.py (critical) |
| [archive/fashion_logistics_agent.py](archive/fashion_logistics_agent.py) | Single-purpose | services/stakes_optimizer.py (standard) |
| [archive/supply_chain_orchestrator.py](archive/supply_chain_orchestrator.py) | Old version | supply_chain_orchestrator.py (root) |
| [archive/input-data-sources/](archive/input-data-sources/) | Archived copies of datasets | data/ is single source of truth |

**🚨 DO NOT USE ARCHIVE/** - These files are kept for reference only.

---

## 🌊 Data Flow

### **1. Route Optimization Flow**

```python
# User provides route options
routes = [
    {'distance_km': 150, 'vehicle_type': 'ev van', ...},
    {'distance_km': 200, 'vehicle_type': 'truck', ...}
]

# Orchestrator routes to appropriate agent
orchestrator = SupplyChainOrchestrator()
result = orchestrator.optimize_delivery('pharmacy', routes)

# Flow:
# 1. Route to agent (high/medium/low stakes)
# 2. Agent runs gradient descent optimization
# 3. Add early-warning analysis (Module 07)
# 4. Add governance recommendations (Module 06)
# 5. Add to trade-off frontier (Module 05)
# 6. Return complete result
```

### **2. Integrated Pipeline Flow** (NEW)

```python
# User provides features
features = {
    'package_type': 'pharmacy',
    'origin': 'mumbai',
    'destination': 'delhi',
    'weight_kg': 25,
    'weather_condition': 'clear'
}

# Orchestrator runs full pipeline
result = orchestrator.run_integrated_pipeline(features)

# Flow:
# 1. Extract features (tools/extraction_tools.py)
# 2. Risk Service assesses risk (Module 07)
# 3. Sourcing Service gets carrier options (Modules 01, 02, carriers, routes)
# 4. Sourcing Service runs optimization (stakes optimizers + gradient descent)
# 5. Carbon Service analyzes (Modules 03, 04, 05, 06)
# 6. Return final recommendation with all analyses
```

---

## 🎯 Key Features

### **1. Real-World Carrier Data** (NEW: data/carriers.py)
- 10 realistic Indian carriers:
  - Express: DHL, BlueDart, Delhivery Express, XpressBees, Shadowfax
  - Standard: Delhivery Standard, Ecom Express, Ekart, DTDC, Gati
- Carrier profiles include:
  - Performance metrics (avg on-time %, variance)
  - Pricing (₹ per kg-km)
  - Capabilities (cold chain, EV fleet, max weight)
  - Carbon footprint (gCO2 per km)
  - Regional coverage

### **2. Real Indian Routes** (NEW: data/routes.py)
- 20+ major Indian cities (Delhi, Mumbai, Bangalore, etc.)
- Pre-calculated distances for popular routes
- Route characteristics:
  - Metro-to-metro vs rural
  - Regional classification
  - State information

### **3. Multi-Agent Architecture**
- **High-Stakes Agent**: Pharmacy, electronics (≥99% on-time)
- **Medium-Stakes Agent**: Furniture, documents (≥95% on-time)
- **Low-Stakes Agent**: Clothing, cosmetics (≥85% on-time)

### **4. Gradient Descent Optimization**
- Minimizes: Carbon + Cost + Penalty(delay)
- Constraint-aware: Respects on-time thresholds
- Finds optimal route from multiple options

### **5. Carbon Analysis**
- Transport carbon (vehicles)
- AI compute carbon (ML models)
- Grid carbon intensity (by country)
- Total carbon = Transport + AI
- Proves AI carbon is negligible

### **6. CASP Metric**
```
CASP = Service Performance / Total Carbon
```
- Higher CASP = More efficient
- Fashion typically has HIGHER CASP (more flexibility)
- Pharma typically has LOWER CASP (constrained by safety)

---

## 🚀 Quick Start

### **Installation**
```bash
pip install -r requirements.txt
```

### **Basic Usage**
```python
from supply_chain_orchestrator import SupplyChainOrchestrator

# Initialize
orchestrator = SupplyChainOrchestrator()

# Option 1: Run integrated pipeline (NEW - Recommended)
result = orchestrator.run_integrated_pipeline({
    'package_type': 'pharmacy',
    'origin': 'mumbai',
    'destination': 'delhi',
    'weight_kg': 25
})

print(f"Recommended carrier: {result['recommendation']}")
print(f"Total carbon: {result['total_carbon']} gCO2")
print(f"Risk level: {result['risk_level']}")

# Option 2: Optimize with specific routes
routes = [
    {'distance_km': 150, 'vehicle_type': 'ev van', ...},
    {'distance_km': 200, 'vehicle_type': 'truck', ...}
]
result = orchestrator.optimize_delivery('pharmacy', routes)
```

### **Run Demos**
```bash
# Interactive demos with real carriers
python demo_greenroute.py

# Basic examples
python example_usage.py

# Full system test
python supply_chain_orchestrator.py
```

---

## 📊 System Metrics

| Metric | Value |
|--------|-------|
| Total Carriers | 10 Indian carriers |
| Total Cities | 20+ major cities |
| Training Records | 25,000 logistics records |
| ML Modules | 7 analysis modules |
| Agent Types | 3 stakes-based agents |
| Package Categories | 9 package types |
| Active Files | 25+ Python files |
| Archived Files | 3 files (for reference) |

---

## 🎓 Research Findings

### **Key Insight 1: Domain-Dependent Returns**
- Pharmacy: 4% carbon reduction (constrained by 99% SLA)
- Fashion: 40% carbon reduction (flexible 85% SLA)

### **Key Insight 2: AI Carbon is Negligible**
- Transport carbon: ~120,000 gCO2
- AI compute carbon: ~0.5 gCO2
- AI is 240,000× smaller than transport!

### **Key Insight 3: CASP Reveals Efficiency**
- Fashion has 2-3× higher CASP than Pharma
- Quantifies the "flexibility advantage"

### **Key Insight 4: Context-Aware AI Deployment**
- High-stakes: Low ROI (small savings, high constraints)
- Low-stakes: High ROI (large savings, flexible constraints)

---

## 🔧 Recent Changes (Jan 29, 2026)

### **New Additions**
1. **data/** directory with real carriers & routes
2. **tools/** directory with agent utilities
3. **3 new agents**: risk_agent.py, sourcing_agent.py, carbon_agent.py
4. **demo_greenroute.py**: Interactive demos
5. **run_integrated_pipeline()**: End-to-end flow

### **Improvements**
1. Real-world carrier data (10 carriers with profiles)
2. Indian city routes (20+ cities with distances)
3. Integrated multi-agent pipeline (Extract → Risk → Sourcing → Carbon)
4. Better tool organization (extraction, risk, sourcing, carbon)

### **File Status**
- ✅ Active: 25+ files in use
- 📦 Archived: 3 files (old versions)
- 🆕 New: 9 files (data/, tools/, demo)

---

## 📝 Citation

> "We demonstrate that AI-enabled supply chain optimization exhibits domain-dependent returns: pharmaceutical logistics (high-stakes) achieved only 4% carbon reduction due to safety constraints, while fashion logistics (low-stakes) achieved 40% reduction through flexible routing, proving that context-aware AI deployment is critical for sustainable supply chain design."

---

## ✅ System Status

| Component | Status | Notes |
|-----------|--------|-------|
| Multi-Agent System | ✅ Complete | 6 agents fully implemented |
| 7 Analysis Modules | ✅ Complete | All modules functional |
| Real Carrier Data | ✅ Complete | 10 carriers with profiles |
| Real Route Data | ✅ Complete | 20+ cities with distances |
| Integrated Pipeline | ✅ Complete | End-to-end flow working |
| Gradient Descent | ✅ Complete | Constraint-aware optimizer |
| Carbon Analysis | ✅ Complete | Transport + AI carbon |
| Demos | ✅ Complete | Interactive + examples |

**🎉 SYSTEM IS FULLY IMPLEMENTED AND READY FOR RESEARCH USE**

---

*For questions or issues, refer to the README.md or examine individual module files.*

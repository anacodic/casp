# Supply Chain Multi-Agent Optimization System

Complete implementation matching the submitted abstract for high-stakes logistics optimization with carbon-aware computation.

## 🏗️ System Architecture

### MODULE 1: SUPPLY-CHAIN DECISION MODULES

#### 1a. Predictive Analytics (`modules/01_predictive_analytics.py`)
- **GradientBoostingRegressor** for predicting:
  - Delivery cost
  - Carbon emissions
  - On-time probability
- Identifies where forecasts fail (weather, partner, etc.)

#### 1b. Multi-Agent Optimization (`orchestration/`)
- **High-Stakes Agent**: pharmacy, groceries (≥99% on-time)
- **Medium-Stakes Agent**: electronics, automobile parts, furniture, documents, fragile items (≥95% on-time)
- **Low-Stakes Agent**: clothing, cosmetics (≥85% on-time)
- **Gradient Descent Optimization**: Minimizes Carbon + Cost + Penalty(delay)

#### 1c. Vendor Segmentation (`modules/02_vendor_segmentation.py`)
- **K-Means clustering** on delivery partners
- Identifies: Fast/Cheap/Reliable/Unreliable clusters
- Maps roles: Premium carriers vs budget carriers

### MODULE 2: NATIONAL ENERGY-TRANSITION CONDITIONS (`modules/04_grid_carbon_scenarios.py`)
- Grid carbon intensity by country:
  - India: 632 gCO2/kWh
  - USA: 386 gCO2/kWh
  - France: 60 gCO2/kWh (nuclear)
  - China: 555 gCO2/kWh
- Shows impact on total carbon (transport + AI compute)

### MODULE 3: CARBON-AWARE COMPUTATION (`modules/03_carbon_cost_intelligence.py`)
- Carbon-cost-of-intelligence layer
- Model comparison: Gemini Flash (0.5 J/token) vs GPT-4 (2.0 J/token)
- Calculates AI compute carbon based on grid intensity

### OUTPUTS

#### OUTPUT 1: Trade-Off Frontiers (`modules/05_trade_off_frontiers.py`)
- Pareto frontier visualization (Carbon vs Service vs Cost)
- CASP metric calculation (Service / Carbon)
- 2D and 3D visualizations

#### OUTPUT 2: Governance Levers (`modules/06_governance_levers.py`)
- Policy recommendations by package type:
  - Vehicle selection
  - Inventory buffer sizing
  - Carrier selection
  - AI compute policies
  - Sourcing location

#### OUTPUT 3: Early-Warning Indicators (`modules/07_early_warning_system.py`)
- Delay prediction model (GradientBoostingClassifier)
- Risk scoring: P(delay) × Impact Multiplier
- Disruption amplification alerts

## 📁 Project Structure

```
casp/
├── services/                         # Python backend (not LLM agents)
│   ├── stakes_optimizer.py          # Single optimizer: critical 99%, high_value 95%, standard 85%
│   ├── risk_service.py              # Risk assessment service
│   ├── sourcing_service.py          # Sourcing recommendations
│   ├── carbon_service.py            # Carbon analysis service
│   └── __init__.py
├── config/                           # Configuration files
│   ├── agent_mapping.py             # Package type → Agent mapping
│   ├── api_config.py                # API configuration
│   ├── carriers.py                  # Carrier definitions
│   ├── routes.py                    # Route definitions
│   ├── vehicle_emissions.py         # Vehicle emission factors
│   ├── grid_carbon.py               # Grid carbon intensity by country
│   └── __init__.py
├── optimization/                     # Optimization algorithms
│   ├── route_optimizer.py           # Discrete route optimizer (evaluate all, pick best)
│   └── __init__.py
├── orchestration/                    # LLM multi-agent orchestration (Strands + Bedrock)
│   ├── orchestrator_agent.py        # Main orchestrator LLM agent
│   ├── risk_agent_strands.py        # Risk Agent (LLM #2)
│   ├── sourcing_agent_strands.py    # Sourcing Agent (LLM #3)
│   ├── agent_tools.py               # Low-level tools (weather, news, web search)
│   ├── bedrock_tools.py             # Bedrock tool definitions
│   ├── _orchestrator_instance.py    # Singleton orchestrator instance
│   └── __init__.py
├── modules/                          # Analysis modules
│   ├── 01_predictive_analytics.py   # ML prediction models
│   ├── 02_vendor_segmentation.py    # K-Means clustering
│   ├── 03_carbon_cost_intelligence.py # Carbon-cost-of-intelligence layer
│   ├── 04_grid_carbon_scenarios.py  # Country carbon scenarios
│   ├── 05_trade_off_frontiers.py    # Pareto frontier analysis
│   ├── 06_governance_levers.py      # Policy recommendations
│   ├── 07_early_warning_system.py   # Disruption prediction
│   └── __init__.py
├── tools/                            # Utility tools
│   ├── extraction_tools.py          # Natural language feature extraction
│   ├── semantic_classifier.py       # Bedrock-based package type classifier
│   ├── sourcing_tools.py            # Carrier sourcing tools
│   ├── risk_tools.py                # Risk assessment tools
│   ├── carbon_tools.py              # Carbon calculation tools
│   ├── defaults_from_data.py        # Data-derived defaults
│   ├── web_search.py                # Web search integration
│   └── __init__.py
├── data/                             # Single source of truth for data
│   ├── apis/                        # External API clients (weather, news, distance)
│   ├── datasets/
│   │   └── Delivery_Logistics.csv  # Main dataset (25K records)
│   └── reference/                   # Static reference data (JSON/CSV)
│       ├── carriers.json
│       ├── cities.json
│       ├── grid_carbon.json
│       ├── ai_model_energy.json
│       ├── routes.csv
│       └── vehicle_emissions.csv
├── frontend/                         # Optional web UI (standalone HTML)
│   └── index.html
├── app.py                            # FastAPI web app
├── supply_chain_orchestrator.py      # Main orchestrator (programmatic API)
├── generate_figure_data.py           # Paper figure generation
├── requirements.txt                  # Python dependencies
├── .env.example                      # Environment variable template
└── README.md                         # This file
```

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from supply_chain_orchestrator import SupplyChainOrchestrator

# Initialize orchestrator
orchestrator = SupplyChainOrchestrator()

# Define route options
routes = [
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
    # ... more routes
]

# Optimize for pharmacy (high-stakes)
report = orchestrator.generate_comprehensive_report('pharmacy', routes)
print(report)
```

## 📊 Package Type Mapping

| Package Type | Stakes Level | On-Time Threshold | StakesOptimizer |
|-------------|--------------|-------------------|------------------|
| pharmacy | critical | ≥99% | StakesOptimizer (high_stakes) |
| electronics | high_value | ≥95% | StakesOptimizer (medium_stakes) |
| groceries | critical | ≥99% | StakesOptimizer (high_stakes) |
| automobile parts | high_value | ≥95% | StakesOptimizer (medium_stakes) |
| furniture | high_value | ≥95% | StakesOptimizer (medium_stakes) |
| documents | high_value | ≥95% | StakesOptimizer (medium_stakes) |
| fragile items | high_value | ≥95% | StakesOptimizer (medium_stakes) |
| clothing | standard | ≥85% | StakesOptimizer (low_stakes) |
| cosmetics | standard | ≥85% | StakesOptimizer (low_stakes) |

## 🔬 Key Features

### 1. Gradient Descent Optimization
- Minimizes: Carbon + Cost + Penalty(delay)
- Constraint-aware: Respects on-time thresholds
- Finds optimal route from multiple options

### 2. Predictive Analytics
- Cost prediction (GradientBoosting)
- Carbon prediction
- On-time probability prediction
- Identifies forecast failure points

### 3. Vendor Segmentation
- K-Means clustering on delivery partners
- Identifies premium vs budget carriers
- Recommends vendors by package type

### 4. Grid Carbon Scenarios
- Compares AI compute carbon across countries
- Shows impact of location on total carbon
- Finds optimal country for lowest carbon

### 5. Trade-Off Frontiers
- Pareto frontier visualization
- CASP metric (Service / Carbon)
- 3D analysis (Carbon, Service, Cost)

### 6. Governance Levers
- Policy recommendations by package type
- Vehicle selection guidelines
- Buffer sizing recommendations
- AI compute policies

### 7. Early-Warning System
- Delay probability prediction
- Risk scoring: P(delay) × Impact
- Disruption alerts

## 📈 CASP Metric

**CASP = Service Performance / Total Carbon**

- Higher CASP = More efficient (more service per unit carbon)
- Fashion typically has HIGHER CASP (more flexibility)
- Pharma typically has LOWER CASP (constrained by safety)

## 🌍 Grid Carbon Intensity

| Country | Grid (gCO2/kWh) | AI Compute Carbon (500 tokens) |
|---------|-----------------|-------------------------------|
| France | 60 | 6 gCO2 |
| Canada | 110 | 11 gCO2 |
| UK | 193 | 19 gCO2 |
| USA | 386 | 39 gCO2 |
| China | 555 | 55 gCO2 |
| India | 632 | 63 gCO2 |

## 🎯 Key Findings (For Paper)

1. **Optimization Potential Differs**:
   - Pharma: Constrained by ≥99% on-time → Limited route choices
   - Fashion: Flexible at ≥85% on-time → Many route choices

2. **Carbon Savings Potential**:
   - Pharma: Can only optimize within narrow safety window (~4% savings)
   - Fashion: Can aggressively optimize for carbon (~40% savings)

3. **AI ROI Varies by Context**:
   - Pharma: AI compute carbon (50 gCO2) vs small physical savings = LOW ROI
   - Fashion: AI compute carbon (50 gCO2) vs large physical savings = HIGH ROI

4. **CASP Scores Reveal Efficiency**:
   - Fashion typically has 2-3× higher CASP (more service per carbon)
   - This quantifies the "flexibility advantage"

## 📚 Citation for Paper

> "We demonstrate that AI-enabled supply chain optimization exhibits domain-dependent returns: pharmaceutical logistics (high-stakes) achieved only 4% carbon reduction due to safety constraints, while fashion logistics (low-stakes) achieved 40% reduction through flexible routing, proving that context-aware AI deployment is critical for sustainable supply chain design."

## 🔧 Running Individual Modules

### Predictive Analytics
```python
import importlib

pa = importlib.import_module('modules.01_predictive_analytics')
analytics = pa.PredictiveAnalytics()
analytics.load_data()
analytics.train_models()
predictions = analytics.predict(route_dict)
```

### Vendor Segmentation
```python
import importlib

vs = importlib.import_module('modules.02_vendor_segmentation')
segmentation = vs.VendorSegmentation()
segmentation.load_data()
segmentation.cluster_vendors(n_clusters=4)
clusters = segmentation.interpret_clusters()
```

### Grid Carbon Scenarios
```python
import importlib

gc = importlib.import_module('modules.04_grid_carbon_scenarios')
scenarios = gc.GridCarbonScenarios()
comparison = scenarios.compare_countries(energy_joules=500)
optimal = scenarios.get_optimal_country(transport_carbon=120000)
```

## 🌐 Web UI

Query-based flow with data-derived defaults and edit-before-run:

1. **Run the API** (from the `casp/` directory):
   ```bash
   pip install fastapi uvicorn
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```
2. Open **http://localhost:8000** in a browser.
3. Enter a shipment description (e.g. "I want to ship 150km fashion"), click **Extract Details**.
4. Review "Your Shipment Details" and "We assumed (click to change)"; edit any value, then click **Run optimization**.
5. Results show recommended carrier, cost, carbon, on-time %, and risk.

- **Defaults** come from `data/datasets/Delivery_Logistics.csv` (mode for categorical, mean for numeric, per package type). See `tools/defaults_from_data.py` and `tools/extraction_tools.py`.
- **API:** `POST /api/extract` (query → features + defaults_used), `POST /api/optimize` (features → pipeline result).

#### API quick reference (server running on port 8000)

```bash
# Health
curl -s http://127.0.0.1:8000/api/health

# Extract features from natural language (optional first step)
curl -s -X POST http://127.0.0.1:8000/api/extract -H "Content-Type: application/json" -d '{"query": "Ship insulin Mumbai to Delhi, 150 km, 5 kg"}'

# Optimize with features (use the "features" object from extract, or build your own)
curl -s -X POST http://127.0.0.1:8000/api/optimize -H "Content-Type: application/json" -d '{"features": {"package_type": "pharmacy", "origin": "Mumbai", "destination": "Delhi", "distance_km": 150, "package_weight_kg": 5}}'

# LLM orchestration (Strands + Bedrock; adaptive search)
curl -s -X POST http://127.0.0.1:8000/api/chat -H "Content-Type: application/json" -d '{"message": "Optimize delivery for insulin Mumbai to Delhi"}'
```

Use the same host/port if you started the server with a different port (e.g. `--port 7244` → `http://127.0.0.1:7244`).

### Semantic classification (Bedrock) and insulin fix

- **Package type** is inferred from the query using rules plus an optional **Bedrock (Claude) classifier** when no rule matches. This avoids life-safety mistakes (e.g. "insulin" → **pharmacy**, not clothing).
- **Pharmacy synonyms** in `tools/extraction_tools.py` include: medicine, medical, pharma, **insulin**, vaccine, drugs, medication, prescription(s), refrigerated medicine.
- When no keyword matches, the system calls `tools/semantic_classifier.py` (Bedrock) to classify `package_type`; if Bedrock is unavailable it defaults to **pharmacy** (safe default). Requires `boto3` and AWS credentials for Bedrock.

### LLM orchestration (Strands + Bedrock)

- **Orchestrator (LLM #1)** coordinates and generates adaptive search queries. It calls **Risk Agent (LLM #2)** and **Sourcing Agent (LLM #3)** as tools; Carbon is a Python tool only. Risk Agent gathers weather/news from API + web search (union), fuses conflicting data, then uses Module 07. Sourcing Agent uses distance, web search, and carrier options to recommend a carrier.
- **Install:** `pip install strands-agents boto3` (see `requirements.txt`).
- **API:** `POST /api/chat` with body `{"message": "Optimize delivery for insulin Mumbai to Delhi"}`. The orchestrator uses the multi-agent path: `extract_features_tool` → `risk_agent_tool`(features, risk_queries) → `sourcing_agent_tool`(features, risk, sourcing_queries) → `run_optimization_tool` → `carbon_analysis_tool`.
- **CLI:** From `casp/`: `python -m orchestration.orchestrator_agent "Optimize delivery for insulin Mumbai to Delhi"`.
- **Tools** (in `orchestration/bedrock_tools.py`): `extract_features_tool`, `risk_agent_tool`, `sourcing_agent_tool`, `run_optimization_tool`, `carbon_analysis_tool`. See `orchestration/agent_tools.py` for low-level tools (weather_api, news_api, web_search, etc.) used by Risk and Sourcing agents.

## 📝 Notes

- All modules use the `data/datasets/Delivery_Logistics.csv` dataset (25,000 records; single source of truth)
- Models are trained on first run (may take a few minutes)
- Results are deterministic (random_state=42)
- System matches the submitted abstract architecture

## 🤝 Contributing

This is a research implementation. For questions or issues, please refer to the paper documentation.

"""
Option 2: Transparent Defaults – FastAPI backend.
Endpoints: POST /api/extract (query -> features + defaults_used), POST /api/optimize (features -> pipeline result).
Run from code/ directory: uvicorn app:app --reload
"""
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# Load .env before anything else reads os.environ.
# override=False means real env vars (set by prod infra) always win over .env values.
_ENV_FILE = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_ENV_FILE, override=False)

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Ensure we run from code/ so data/ and imports resolve
_CODE_DIR = Path(__file__).resolve().parent
if os.getcwd() != str(_CODE_DIR):
    os.chdir(_CODE_DIR)

from tools.extraction_tools import extract_from_query_and_merge_defaults, extract_features_from_dict
from tools.defaults_from_data import get_defaults_for_package, get_defaults_cache

app = FastAPI(
    title="Supply Chain Option 2 API",
    description="Query -> extract + data-derived defaults -> edit -> run optimization",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Return HTTPException as JSON (preserve status code)."""
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Ensure all unhandled errors return JSON so the frontend never gets non-JSON 500 bodies."""
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )

# Lazy singleton orchestrator (heavy init on first /optimize)
_orchestrator = None
_DATA_PATH = str(_CODE_DIR / "data" / "datasets" / "Delivery_Logistics.csv")


def get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        from supply_chain_orchestrator import SupplyChainOrchestrator
        _orchestrator = SupplyChainOrchestrator(data_path=_DATA_PATH)
    return _orchestrator


# --- Request/Response models ---

class ExtractRequest(BaseModel):
    query: str = Field(..., description="Natural language shipment description")


class ExtractResponse(BaseModel):
    features: Dict[str, Any] = Field(..., description="Full feature dict (extracted + defaults)")
    defaults_used: List[str] = Field(..., description="Keys that were filled from data-derived defaults")


class OptimizeRequest(BaseModel):
    features: Dict[str, Any] = Field(..., description="Feature dict (from extract or after user edit)")


# Serve frontend assets (e.g. supply_chain.jpg)
_FRONTEND_DIR = _CODE_DIR / "frontend"
if _FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static")

# --- Endpoints ---

@app.get("/")
async def root():
    """Serve the Option 2 UI (single page)."""
    from fastapi.responses import HTMLResponse, PlainTextResponse
    html_path = _CODE_DIR / "frontend" / "index.html"
    if html_path.exists():
        return HTMLResponse(content=open(html_path, encoding="utf-8").read())
    return PlainTextResponse(
        "Option 2 API. Use POST /api/extract and POST /api/optimize. Mount frontend at / if index.html exists."
    )


@app.post("/api/extract", response_model=ExtractResponse)
async def extract(request: ExtractRequest) -> ExtractResponse:
    """
    Extract features from natural-language query and merge with data-derived defaults (per package_type).
    Returns features + list of keys that were defaulted (for UI "We assumed" section).
    """
    try:
        features, defaults_used = extract_from_query_and_merge_defaults(request.query.strip())
        return ExtractResponse(features=features, defaults_used=defaults_used)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


def _json_safe(obj: Any) -> Any:
    """Convert numpy/pandas types to native Python for JSON serialization."""
    import math
    import numpy as np
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        f = float(obj)
        return None if math.isnan(f) else f
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    try:
        import pandas as pd
        if isinstance(obj, pd.DataFrame):
            try:
                clean = obj.where(pd.notnull(obj), None)
            except Exception:
                clean = obj.fillna(value=None)
            return _json_safe(clean.to_dict(orient="records"))
        if isinstance(obj, pd.Series):
            try:
                clean = obj.where(pd.notnull(obj), None)
            except Exception:
                clean = obj.fillna(value=None)
            return _json_safe(clean.to_dict())
    except ImportError:
        pass
    return obj


@app.post("/api/optimize")
async def optimize(request: OptimizeRequest) -> Dict[str, Any]:
    """
    Run the full pipeline (Risk -> Sourcing -> Carbon) with the given features.
    Expects the same feature shape as returned by /api/extract (after optional user edit).
    """
    try:
        features = extract_features_from_dict(request.features)
        orch = get_orchestrator()
        result = orch.run_integrated_pipeline(features)
        try:
            return _json_safe(result)
        except Exception as serr:
            raise HTTPException(status_code=500, detail=f"Response serialization failed: {serr}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health():
    return {"status": "ok"}


# --- Optional: Strands + Bedrock orchestration (POST /api/chat) ---
def _chat_available() -> bool:
    try:
        from strands import Agent  # noqa: F401
        return True
    except ImportError:
        return False


class ChatRequest(BaseModel):
    message: str = Field(..., description="Natural language message (e.g. optimize insulin Mumbai to Delhi)")


@app.post("/api/chat")
async def chat(request: ChatRequest) -> Dict[str, Any]:
    """
    LLM-orchestrated supply chain: Bedrock decides which tools to call (extract, risk, sourcing, carbon, full pipeline).
    Requires strands-agents and boto3; AWS credentials for Bedrock.
    """
    if not _chat_available():
        raise HTTPException(
            status_code=503,
            detail="Orchestration unavailable. Install: pip install strands-agents boto3 and configure AWS credentials.",
        )
    try:
        from orchestration.orchestrator_agent import run_orchestrator
        response_text = run_orchestrator(request.message.strip())
        return {"response": response_text}
    except Exception as e:
        error_msg = str(e)
        # Provide helpful error messages for common AWS credential issues
        if "NoCredentialsError" in error_msg or "Unable to locate credentials" in error_msg:
            error_msg = (
                "AWS credentials not found. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY "
                "environment variables or configure ~/.aws/credentials. See DEPLOYMENT.md for details."
            )
        elif "AccessDenied" in error_msg or "UnauthorizedOperation" in error_msg:
            error_msg = (
                "AWS access denied. Please ensure your AWS credentials have bedrock:InvokeModel permission. "
                "See DEPLOYMENT.md for IAM policy requirements."
            )
        elif "region" in error_msg.lower() or "Region" in error_msg:
            error_msg = (
                f"AWS region issue: {error_msg}. "
                "Please set AWS_REGION environment variable (e.g., us-east-1). "
                "See DEPLOYMENT.md for supported regions."
            )
        raise HTTPException(status_code=500, detail=error_msg)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

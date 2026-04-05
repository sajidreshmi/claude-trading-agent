"""
FastAPI Server — The Production API Layer

DOMAIN 3: CONFIG & WORKFLOWS

This file demonstrates:
1. Proper API design with typed request/response models
2. Background task execution (agents run async)
3. Health checks and monitoring endpoints
4. Tool registry health (circuit breakers, rate limits)
5. Structured error responses (never generic 500s)

WORKFLOW PATTERN:
Client → POST /analyze → returns job_id immediately
Client → GET /jobs/{job_id} → polls for result
This prevents HTTP timeouts on long-running agent orchestration.
"""

import uuid
import asyncio
import logging
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from agents.coordinator import CoordinatorAgent
from models.schemas import SubAgentTask, SubAgentResult, TaskStatus
from tools.market_tools import create_market_tools_registry
from config.settings import settings

logger = logging.getLogger(__name__)

# ─── In-Memory Job Store (production: use Redis or PostgreSQL) ───
jobs: dict[str, dict] = {}
tool_registry = create_market_tools_registry()


# ─── Request / Response Models ───────────────────────────────────

class AnalyzeRequest(BaseModel):
    """
    What the client sends. Minimal, validated.
    
    Notice: The client sends a SYMBOL, not instructions.
    The agent system decides HOW to analyze it.
    """
    symbol: str = Field(
        min_length=1, max_length=10,
        description="Stock ticker symbol (e.g., AAPL, TSLA)",
        examples=["AAPL", "TSLA", "GOOGL"],
    )
    context: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Optional analysis context",
        examples=["Focus on momentum indicators"],
    )


class JobResponse(BaseModel):
    """Immediate response — job accepted, here's your ID."""
    job_id: str
    status: str
    message: str
    created_at: str


class JobStatusResponse(BaseModel):
    """Polling response — current state of the analysis."""
    job_id: str
    status: str
    symbol: str
    created_at: str
    completed_at: Optional[str] = None
    result: Optional[dict] = None
    error: Optional[dict] = None
    iterations_used: Optional[int] = None


class HealthResponse(BaseModel):
    """System health check response."""
    status: str
    version: str
    llm_model: str
    tool_registry_health: dict
    active_jobs: int
    uptime_seconds: float


# ─── App Lifecycle ───────────────────────────────────────────────

start_time = datetime.utcnow()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    logger.info("🚀 Trading Agent API starting up")
    logger.info(f"   LLM Model: {settings.llm_model}")
    logger.info(f"   Risk Threshold: {settings.risk_threshold}")
    logger.info(f"   Max Position: ${settings.max_position_usd:,.0f}")
    logger.info(f"   Tools registered: {len(tool_registry._tools)}")
    yield
    logger.info("🛑 Trading Agent API shutting down")


app = FastAPI(
    title="Trading Agent API",
    description="Multi-agent trading analysis system with Claude",
    version="0.1.0",
    lifespan=lifespan,
)


# ─── Background Job Execution ───────────────────────────────────

async def _run_analysis_job(job_id: str, symbol: str, context: str | None):
    """
    Run the full agent orchestration in the background.
    
    WHY BACKGROUND?
    Agent orchestration can take 30-90 seconds (multiple LLM calls,
    tool executions, subagent dispatches). HTTP requests would timeout.
    
    PATTERN: Accept job → return ID → client polls → return result.
    This is how every production async system works.
    """
    try:
        jobs[job_id]["status"] = "running"

        coordinator = CoordinatorAgent()
        
        task = SubAgentTask(
            task_id=job_id,
            agent_type="coordinator",
            objective=f"Analyze {symbol} and produce a trade recommendation. "
                      f"First dispatch market analysis, then check risk, and produce a final recommendation.",
            input_data={
                "symbol": symbol,
                "context": context or "Standard analysis",
                "account_balance": 50000,
            },
            max_iterations=settings.coordinator_max_iterations,
        )

        result: SubAgentResult = await coordinator.run(task)

        jobs[job_id].update({
            "status": result.status.value,
            "result": result.result,
            "error": result.error.model_dump() if result.error else None,
            "iterations_used": result.iterations_used,
            "completed_at": datetime.utcnow().isoformat(),
        })

    except Exception as e:
        logger.error(f"Job {job_id} failed with exception: {e}", exc_info=True)
        jobs[job_id].update({
            "status": "failed",
            "error": {
                "code": "UNHANDLED_EXCEPTION",
                "message": str(e),
                "recoverable": False,
            },
            "completed_at": datetime.utcnow().isoformat(),
        })


# ─── API Endpoints ──────────────────────────────────────────────

@app.post("/analyze", response_model=JobResponse, status_code=202)
async def start_analysis(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    """
    Start a trading analysis job.
    
    HOOK: Concurrency Limit
    - Max 5 concurrent jobs to prevent resource exhaustion.
    - Deterministic backpressure (503 Service Unavailable).
    """
    MAX_CONCURRENT_JOBS = 5
    active_jobs = sum(1 for j in jobs.values() if j["status"] in ("pending", "running"))
    
    if active_jobs >= MAX_CONCURRENT_JOBS:
        raise HTTPException(
            status_code=503,
            detail={
                "error_code": "CONCURRENCY_LIMIT_REACHED",
                "message": f"Server is at max capacity ({MAX_CONCURRENT_JOBS} active jobs)",
                "suggestion": "Wait for current jobs to complete and try again.",
                "retry_after_seconds": 60,
            },
        )

    job_id = str(uuid.uuid4())
    
    jobs[job_id] = {
        "job_id": job_id,
        "symbol": request.symbol.upper(),
        "status": "pending",
        "created_at": datetime.utcnow().isoformat(),
        "result": None,
        "error": None,
    }

    background_tasks.add_task(
        _run_analysis_job, job_id, request.symbol.upper(), request.context
    )

    return JobResponse(
        job_id=job_id,
        status="pending",
        message=f"Analysis job for {request.symbol.upper()} accepted. Poll GET /jobs/{job_id} for results.",
        created_at=jobs[job_id]["created_at"],
    )


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get the status and result of an analysis job.
    
    Client polls this endpoint until status is "completed" or "failed".
    """
    if job_id not in jobs:
        raise HTTPException(
            status_code=404,
            detail={
                "error_code": "JOB_NOT_FOUND",
                "message": f"Job {job_id} not found",
                "suggestion": "Check the job_id or start a new analysis",
            },
        )

    job = jobs[job_id]
    return JobStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        symbol=job["symbol"],
        created_at=job["created_at"],
        completed_at=job.get("completed_at"),
        result=job.get("result"),
        error=job.get("error"),
        iterations_used=job.get("iterations_used"),
    )


@app.get("/jobs", response_model=list[JobStatusResponse])
async def list_jobs(status: Optional[str] = None, limit: int = 20):
    """List all jobs, optionally filtered by status."""
    result = []
    for job in list(jobs.values())[-limit:]:
        if status and job["status"] != status:
            continue
        result.append(JobStatusResponse(
            job_id=job["job_id"],
            status=job["status"],
            symbol=job["symbol"],
            created_at=job["created_at"],
            completed_at=job.get("completed_at"),
            result=job.get("result"),
            error=job.get("error"),
            iterations_used=job.get("iterations_used"),
        ))
    return result


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check — shows system status, tool health, active jobs.
    
    PRODUCTION MUST-HAVE:
    Every deployed service needs a health endpoint. Load balancers,
    k8s probes, monitoring systems all use this.
    """
    uptime = (datetime.utcnow() - start_time).total_seconds()
    active = sum(1 for j in jobs.values() if j["status"] in ("pending", "running"))

    return HealthResponse(
        status="healthy",
        version="0.1.0",
        llm_model=settings.llm_model,
        tool_registry_health=tool_registry.get_health(),
        active_jobs=active,
        uptime_seconds=round(uptime, 1),
    )


@app.get("/tools")
async def list_tools(category: Optional[str] = None):
    """
    List registered tools and their health status.
    
    Useful for debugging: "Why isn't the agent using X tool?"
    Check if it's rate-limited or circuit-broken.
    """
    tools = tool_registry.get_tools_for_llm(category=category)
    health = tool_registry.get_health()

    return {
        "tools": tools,
        "health": health,
        "total_count": len(tools),
    }


@app.get("/tools/audit")
async def tool_audit_log(limit: int = 50):
    """
    Tool execution audit trail.
    
    For compliance and debugging. In production, this feeds
    into your observability pipeline (Datadog, Grafana, etc.)
    """
    return {
        "audit_log": tool_registry.get_audit_log(limit=limit),
        "total_logged": len(tool_registry._call_log),
    }


@app.get("/config")
async def get_config():
    """
    Show current configuration (sanitized — no API keys).
    
    NEVER expose secrets. Show only operational config.
    """
    return {
        "llm_model": settings.llm_model,
        "coordinator_max_iterations": settings.coordinator_max_iterations,
        "subagent_max_iterations": settings.subagent_max_iterations,
        "risk_threshold": settings.risk_threshold,
        "min_confidence": settings.min_confidence,
        "max_position_usd": settings.max_position_usd,
        "max_portfolio_concentration": settings.max_portfolio_concentration,
        "max_sector_exposure": settings.max_sector_exposure,
        "tool_rate_limit_per_minute": settings.tool_rate_limit_per_minute,
        "api_key_configured": bool(settings.anthropic_api_key),
    }

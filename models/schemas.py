"""
Structured schemas for the trading agent system.

DESIGN PRINCIPLE: Hooks > Prompts
- Every agent output is validated through Pydantic models
- No "parse the LLM text and hope" patterns
- Structured errors with typed error codes
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime


# ─── Error Handling: Structured, Never Generic ───────────────────────

class ErrorCode(str, Enum):
    """Typed error codes — never raise generic 'Error occurred'."""
    TOOL_EXECUTION_FAILED = "TOOL_EXECUTION_FAILED"
    TOOL_NOT_FOUND = "TOOL_NOT_FOUND"
    INVALID_TOOL_INPUT = "INVALID_TOOL_INPUT"
    LLM_API_ERROR = "LLM_API_ERROR"
    MAX_ITERATIONS_EXCEEDED = "MAX_ITERATIONS_EXCEEDED"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    HUMAN_ESCALATION_REQUIRED = "HUMAN_ESCALATION_REQUIRED"
    RISK_LIMIT_EXCEEDED = "RISK_LIMIT_EXCEEDED"


class AgentError(BaseModel):
    """Structured error — the agent ALWAYS knows what went wrong and why."""
    code: ErrorCode
    message: str
    details: Optional[dict] = None
    recoverable: bool = True
    suggested_action: Optional[str] = None


# ─── Agent Communication Schemas ─────────────────────────────────────

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ESCALATED = "escalated"


class SubAgentTask(BaseModel):
    """
    What the coordinator sends to a subagent.
    Notice: structured JSON, NOT natural language instructions.
    """
    task_id: str
    agent_type: str  # "market_analyst", "risk_assessor", "report_writer"
    objective: str
    input_data: dict
    constraints: Optional[dict] = None
    max_iterations: int = Field(default=10, le=25)


class SubAgentResult(BaseModel):
    """
    What a subagent returns to the coordinator.
    Always structured — never free-text responses.
    """
    task_id: str
    status: TaskStatus
    result: Optional[dict] = None
    error: Optional[AgentError] = None
    iterations_used: int = 0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ─── Trading-Specific Schemas ────────────────────────────────────────

class MarketSignal(BaseModel):
    """Output from the Market Analyst agent."""
    symbol: str
    signal_type: str  # "bullish", "bearish", "neutral"
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    indicators: dict = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class RiskAssessment(BaseModel):
    """Output from the Risk Assessor agent."""
    symbol: str
    risk_score: float = Field(ge=0.0, le=1.0)
    max_position_size: float
    stop_loss_pct: float
    concerns: list[str] = Field(default_factory=list)
    approved: bool = False


class TradeDecision(BaseModel):
    """
    Final output from the Coordinator.
    
    CRITICAL: The 'approved' field is a HOOK, not a prompt.
    The LLM proposes, deterministic code decides.
    """
    symbol: str
    action: str  # "buy", "sell", "hold"
    signal: MarketSignal
    risk: RiskAssessment
    approved: bool = False  # Set by HOOK, never by LLM
    rejection_reason: Optional[str] = None

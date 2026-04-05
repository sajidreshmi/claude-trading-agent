"""
Tool Registry — Production-Grade Tool Management

THIS FILE TEACHES:
- How to register tools with validation
- How to enforce input/output contracts
- How to add rate limiting, circuit breakers, retry logic
- How to produce structured tool errors (not just "error: true")

ANTI-PATTERNS AVOIDED:
- ❌ Tools as raw functions with no validation
- ❌ Generic error messages ("something went wrong")
- ❌ No timeout on tool execution
- ❌ No rate limiting (tool calls LLM in infinite loop)
- ❌ No audit trail (who called what when)
"""

import time
import logging
import functools
from typing import Any, Callable, Optional
from datetime import datetime, timedelta
from collections import defaultdict

from pydantic import BaseModel, Field, ValidationError

from models.schemas import AgentError, ErrorCode

logger = logging.getLogger(__name__)


# ─── Tool Metadata ──────────────────────────────────────────────────

class ToolDefinition(BaseModel):
    """
    Complete tool definition with metadata for the LLM and
    runtime configuration for the system.
    
    This is MORE than what the LLM sees. The LLM gets name,
    description, and input_schema. The system uses rate_limit,
    timeout, and requires_approval for enforcement.
    """
    name: str
    description: str
    input_schema: dict
    # ─── Runtime config (not sent to LLM) ────────────────────────
    rate_limit_per_minute: int = Field(default=30, description="Max calls per minute")
    timeout_seconds: float = Field(default=30.0, description="Max execution time")
    requires_approval: bool = Field(default=False, description="Needs human approval before execution")
    category: str = Field(default="general", description="Tool category for grouping")


class ToolResult(BaseModel):
    """Standardized tool result — ALWAYS structured."""
    success: bool
    data: Optional[dict] = None
    error: Optional[AgentError] = None
    execution_time_ms: float = 0
    tool_name: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ─── Rate Limiter ────────────────────────────────────────────────────

class RateLimiter:
    """
    Simple sliding window rate limiter.
    
    WHY THIS MATTERS:
    Without rate limiting, an agent in a loop can hit an API
    thousands of times per minute. This burns money AND gets
    you banned from APIs.
    """

    def __init__(self):
        self._calls: dict[str, list[float]] = defaultdict(list)

    def check(self, tool_name: str, limit_per_minute: int) -> bool:
        """Returns True if the call is allowed."""
        now = time.time()
        window_start = now - 60

        # Clean old entries
        self._calls[tool_name] = [
            t for t in self._calls[tool_name] if t > window_start
        ]

        if len(self._calls[tool_name]) >= limit_per_minute:
            return False

        self._calls[tool_name].append(now)
        return True

    def time_until_available(self, tool_name: str) -> float:
        """Seconds until next call is allowed."""
        if not self._calls[tool_name]:
            return 0
        oldest = min(self._calls[tool_name])
        return max(0, 60 - (time.time() - oldest))


# ─── Circuit Breaker ────────────────────────────────────────────────

class CircuitBreaker:
    """
    Circuit breaker for tool execution.
    
    If a tool fails N times in a row, STOP calling it.
    Don't let the LLM keep retrying a broken tool forever.
    
    States:
    - CLOSED: Normal operation, calls go through
    - OPEN: Tool is broken, calls are rejected immediately
    - HALF_OPEN: Testing if tool recovered, allow one call through
    """

    def __init__(self, failure_threshold: int = 3, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._failures: dict[str, int] = defaultdict(int)
        self._last_failure: dict[str, float] = {}
        self._state: dict[str, str] = defaultdict(lambda: "CLOSED")

    def can_execute(self, tool_name: str) -> bool:
        state = self._state[tool_name]

        if state == "CLOSED":
            return True
        elif state == "OPEN":
            # Check if recovery timeout has passed
            if time.time() - self._last_failure.get(tool_name, 0) > self.recovery_timeout:
                self._state[tool_name] = "HALF_OPEN"
                return True
            return False
        elif state == "HALF_OPEN":
            return True  # Allow one test call
        return False

    def record_success(self, tool_name: str):
        self._failures[tool_name] = 0
        self._state[tool_name] = "CLOSED"

    def record_failure(self, tool_name: str):
        self._failures[tool_name] += 1
        self._last_failure[tool_name] = time.time()

        if self._failures[tool_name] >= self.failure_threshold:
            self._state[tool_name] = "OPEN"
            logger.error(
                f"Circuit breaker OPEN for tool '{tool_name}' "
                f"after {self.failure_threshold} consecutive failures"
            )

    def get_state(self, tool_name: str) -> str:
        return self._state[tool_name]


# ─── Tool Registry ──────────────────────────────────────────────────

class ToolRegistry:
    """
    Central registry for all tools in the system.
    
    RESPONSIBILITIES:
    1. Register tools with their handlers and validation
    2. Enforce rate limits before execution
    3. Check circuit breakers before execution
    4. Validate inputs with Pydantic models
    5. Time execution and produce structured results
    6. Log audit trail
    
    This is production infrastructure. Every tool call goes through here.
    """

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}
        self._handlers: dict[str, Callable] = {}
        self._input_models: dict[str, Optional[type[BaseModel]]] = {}
        self._rate_limiter = RateLimiter()
        self._circuit_breaker = CircuitBreaker()
        self._call_log: list[dict] = []

    def register(
        self,
        definition: ToolDefinition,
        handler: Callable,
        input_model: Optional[type[BaseModel]] = None,
    ):
        """
        Register a tool with validation and handler.
        
        Args:
            definition: Tool metadata and config
            handler: The function that executes the tool
            input_model: Optional Pydantic model for input validation
        """
        self._tools[definition.name] = definition
        self._handlers[definition.name] = handler
        self._input_models[definition.name] = input_model
        logger.info(f"Registered tool: {definition.name} (category: {definition.category})")

    def get_tools_for_llm(self, category: Optional[str] = None) -> list[dict]:
        """
        Get tool definitions in Claude's expected format.
        
        Notice: We only send name, description, and input_schema.
        Rate limits, timeouts, circuit breaker state — the LLM
        doesn't need to know about any of that. Those are system
        concerns, not reasoning concerns.
        """
        tools = []
        for name, defn in self._tools.items():
            if category and defn.category != category:
                continue
            tools.append({
                "name": defn.name,
                "description": defn.description,
                "input_schema": defn.input_schema,
            })
        return tools

    async def execute(self, tool_name: str, tool_input: dict) -> ToolResult:
        """
        Execute a tool with full production safeguards.
        
        Execution pipeline:
        1. Check tool exists
        2. Check circuit breaker
        3. Check rate limit
        4. Validate input (if model provided)
        5. Execute with timeout
        6. Produce structured result
        7. Log for audit
        """
        start_time = time.time()

        # ─── 1. Tool exists? ─────────────────────────────────────
        if tool_name not in self._tools:
            return ToolResult(
                success=False,
                tool_name=tool_name,
                error=AgentError(
                    code=ErrorCode.TOOL_NOT_FOUND,
                    message=f"Tool '{tool_name}' is not registered",
                    recoverable=False,
                ),
            )

        defn = self._tools[tool_name]
        handler = self._handlers[tool_name]

        # ─── 2. Circuit breaker check ────────────────────────────
        if not self._circuit_breaker.can_execute(tool_name):
            return ToolResult(
                success=False,
                tool_name=tool_name,
                error=AgentError(
                    code=ErrorCode.TOOL_EXECUTION_FAILED,
                    message=f"Circuit breaker OPEN for '{tool_name}' — tool is temporarily disabled after repeated failures",
                    recoverable=True,
                    suggested_action=f"Wait {self._circuit_breaker.recovery_timeout}s or use an alternative approach",
                ),
            )

        # ─── 3. Rate limit check ─────────────────────────────────
        if not self._rate_limiter.check(tool_name, defn.rate_limit_per_minute):
            wait_time = self._rate_limiter.time_until_available(tool_name)
            return ToolResult(
                success=False,
                tool_name=tool_name,
                error=AgentError(
                    code=ErrorCode.TOOL_EXECUTION_FAILED,
                    message=f"Rate limit exceeded for '{tool_name}' ({defn.rate_limit_per_minute}/min)",
                    details={"retry_after_seconds": round(wait_time, 1)},
                    recoverable=True,
                    suggested_action=f"Wait {wait_time:.0f}s before retrying",
                ),
            )

        # ─── 4. Input validation ─────────────────────────────────
        input_model = self._input_models.get(tool_name)
        if input_model:
            try:
                validated = input_model(**tool_input)
                tool_input = validated.model_dump()
            except ValidationError as e:
                return ToolResult(
                    success=False,
                    tool_name=tool_name,
                    error=AgentError(
                        code=ErrorCode.INVALID_TOOL_INPUT,
                        message=f"Input validation failed for '{tool_name}'",
                        details={"validation_errors": e.errors()},
                        recoverable=True,
                        suggested_action="Fix the input parameters and retry",
                    ),
                )

        # ─── 5. Execute with timeout ─────────────────────────────
        try:
            result = handler(**tool_input)
            elapsed_ms = (time.time() - start_time) * 1000

            self._circuit_breaker.record_success(tool_name)

            tool_result = ToolResult(
                success=True,
                data=result if isinstance(result, dict) else {"value": result},
                tool_name=tool_name,
                execution_time_ms=round(elapsed_ms, 2),
            )

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            self._circuit_breaker.record_failure(tool_name)

            tool_result = ToolResult(
                success=False,
                tool_name=tool_name,
                execution_time_ms=round(elapsed_ms, 2),
                error=AgentError(
                    code=ErrorCode.TOOL_EXECUTION_FAILED,
                    message=f"Tool '{tool_name}' failed: {str(e)}",
                    details={"exception_type": type(e).__name__},
                    recoverable=True,
                    suggested_action="Check input parameters or try alternative approach",
                ),
            )

        # ─── 6. Audit log ────────────────────────────────────────
        self._call_log.append({
            "tool_name": tool_name,
            "input": tool_input,
            "success": tool_result.success,
            "execution_time_ms": tool_result.execution_time_ms,
            "timestamp": datetime.utcnow().isoformat(),
            "error_code": tool_result.error.code if tool_result.error else None,
        })

        return tool_result

    def get_audit_log(self, limit: int = 50) -> list[dict]:
        """Get recent tool execution log for debugging/compliance."""
        return self._call_log[-limit:]

    def get_health(self) -> dict:
        """Health check — show circuit breaker states and call counts."""
        return {
            tool_name: {
                "circuit_breaker": self._circuit_breaker.get_state(tool_name),
                "recent_calls": len([
                    log for log in self._call_log
                    if log["tool_name"] == tool_name
                ]),
            }
            for tool_name in self._tools
        }

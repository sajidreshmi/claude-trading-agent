"""
Tests — Verify the System Without API Keys

These tests validate:
1. Schema validation (Pydantic models)
2. Tool registry (rate limiting, circuit breaker, validation)
3. Context management (summarization, key decisions)
4. Retry logic (specific error feedback)
5. Escalation flow (agent → human)

These run WITHOUT Claude API calls — testing the infrastructure
that WRAPS the LLM, not the LLM itself.
"""

import pytest
import asyncio
from datetime import datetime

from models.schemas import (
    ErrorCode, AgentError, TaskStatus,
    SubAgentTask, SubAgentResult,
    MarketSignal, RiskAssessment, TradeDecision,
)
from tools.registry import ToolRegistry, ToolDefinition, ToolResult, CircuitBreaker, RateLimiter
from tools.market_tools import (
    create_market_tools_registry,
    FetchPriceInput, CalculateVarInput,
    fetch_stock_price, calculate_value_at_risk,
)
from agents.context_manager import ContextManager, RetryManager, EscalationManager
from config.settings import Settings


# ═══════════════════════════════════════════════════════════════════
# DOMAIN 1 TESTS: Schema Validation
# ═══════════════════════════════════════════════════════════════════

class TestSchemas:
    """Test that Pydantic models enforce constraints."""

    def test_market_signal_valid(self):
        signal = MarketSignal(
            symbol="AAPL",
            signal_type="bullish",
            confidence=0.75,
            reasoning="RSI oversold, MACD bullish crossover",
            indicators={"rsi": 35.2},
        )
        assert signal.confidence == 0.75
        assert signal.symbol == "AAPL"

    def test_market_signal_confidence_bounds(self):
        """Confidence MUST be 0-1. Anything else → validation error."""
        with pytest.raises(Exception):
            MarketSignal(
                symbol="AAPL",
                signal_type="bullish",
                confidence=1.5,  # INVALID — must be <= 1.0
                reasoning="test",
            )

    def test_risk_assessment_valid(self):
        risk = RiskAssessment(
            symbol="AAPL",
            risk_score=0.45,
            max_position_size=7500,
            stop_loss_pct=2.5,
            concerns=["sector_concentration"],
            approved=True,
        )
        assert risk.approved is True

    def test_agent_error_structured(self):
        """Errors are ALWAYS structured with code + suggestion."""
        error = AgentError(
            code=ErrorCode.RISK_LIMIT_EXCEEDED,
            message="Risk score 0.85 exceeds threshold 0.7",
            recoverable=False,
            suggested_action="Reduce position size or choose different asset",
        )
        assert error.code == ErrorCode.RISK_LIMIT_EXCEEDED
        assert error.recoverable is False
        assert error.suggested_action is not None

    def test_subagent_task_max_iterations_capped(self):
        """Max iterations is capped at 25 — prevents runaway agents."""
        with pytest.raises(Exception):
            SubAgentTask(
                task_id="test",
                agent_type="market_analyst",
                objective="analyze AAPL",
                input_data={"symbol": "AAPL"},
                max_iterations=100,  # INVALID — max is 25
            )

    def test_trade_decision_hook_field(self):
        """approved field defaults to False — set by HOOK, not LLM."""
        decision = TradeDecision(
            symbol="AAPL",
            action="buy",
            signal=MarketSignal(
                symbol="AAPL", signal_type="bullish",
                confidence=0.8, reasoning="test",
            ),
            risk=RiskAssessment(
                symbol="AAPL", risk_score=0.3,
                max_position_size=8000, stop_loss_pct=2.0,
            ),
        )
        assert decision.approved is False  # Default — must be set by code


# ═══════════════════════════════════════════════════════════════════
# DOMAIN 2 TESTS: Tool Registry
# ═══════════════════════════════════════════════════════════════════

class TestToolRegistry:
    """Test tool infrastructure: validation, rate limit, circuit breaker."""

    def test_registry_creation(self):
        registry = create_market_tools_registry()
        assert len(registry._tools) == 6

    def test_tools_for_llm_format(self):
        """Output matches Claude's expected tool format."""
        registry = create_market_tools_registry()
        tools = registry.get_tools_for_llm()
        
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool
            # Runtime config (rate_limit, timeout) NOT exposed to LLM
            assert "rate_limit_per_minute" not in tool
            assert "timeout_seconds" not in tool

    def test_tools_filtered_by_category(self):
        registry = create_market_tools_registry()
        market_tools = registry.get_tools_for_llm(category="market_data")
        risk_tools = registry.get_tools_for_llm(category="risk")
        
        assert len(market_tools) == 3  # fetch, news, technical
        assert len(risk_tools) == 3    # var, portfolio, sector

    def test_input_validation_rejects_bad_input(self):
        """Pydantic validates tool inputs BEFORE execution."""
        with pytest.raises(Exception):
            FetchPriceInput(symbol="", period="invalid")

    def test_input_validation_accepts_good_input(self):
        validated = FetchPriceInput(symbol="AAPL", period="1d")
        assert validated.symbol == "AAPL"

    def test_var_input_position_limit(self):
        """Position size capped at $1M."""
        with pytest.raises(Exception):
            CalculateVarInput(symbol="AAPL", position_size_usd=2_000_000)

    def test_tool_execution_returns_data(self):
        result = fetch_stock_price(symbol="AAPL")
        assert "current_price" in result
        assert "volume" in result
        assert result["symbol"] == "AAPL"

    def test_var_calculation(self):
        result = calculate_value_at_risk(symbol="AAPL", position_size_usd=10000)
        assert "var_95_daily_usd" in result
        assert "within_limits" in result


class TestRateLimiter:
    def test_allows_within_limit(self):
        limiter = RateLimiter()
        for _ in range(5):
            assert limiter.check("test_tool", 10) is True

    def test_blocks_over_limit(self):
        limiter = RateLimiter()
        for _ in range(10):
            limiter.check("test_tool", 10)
        assert limiter.check("test_tool", 10) is False


class TestCircuitBreaker:
    def test_starts_closed(self):
        cb = CircuitBreaker(failure_threshold=3)
        assert cb.get_state("test") == "CLOSED"
        assert cb.can_execute("test") is True

    def test_opens_after_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure("test")
        cb.record_failure("test")
        cb.record_failure("test")  # 3rd failure → OPEN
        assert cb.get_state("test") == "OPEN"
        assert cb.can_execute("test") is False

    def test_resets_on_success(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure("test")
        cb.record_failure("test")
        cb.record_success("test")  # Reset
        assert cb.get_state("test") == "CLOSED"


# ═══════════════════════════════════════════════════════════════════
# DOMAIN 5 TESTS: Context Management & Reliability
# ═══════════════════════════════════════════════════════════════════

class TestContextManager:
    def test_adds_messages(self):
        cm = ContextManager(max_messages=20, summary_threshold=15)
        cm.add_message("user", "Analyze AAPL")
        cm.add_message("assistant", "I'll dispatch the analysis")
        assert len(cm.get_messages()) == 2

    def test_records_key_decisions(self):
        cm = ContextManager()
        cm.record_decision({"action": "buy", "symbol": "AAPL", "approved": True})
        assert len(cm.key_decisions) == 1
        assert cm.key_decisions[0]["action"] == "buy"

    def test_compression_triggers(self):
        cm = ContextManager(max_messages=20, summary_threshold=5)
        for i in range(10):
            cm.add_message("user", f"Message {i} " * 50)
        # After threshold, should have compressed
        assert cm.summary is not None

    def test_key_decisions_survive_compression(self):
        cm = ContextManager(summary_threshold=5)
        cm.record_decision({"action": "reject", "symbol": "TSLA", "reason": "high risk"})
        for i in range(10):
            cm.add_message("user", f"Message {i}")
        # Key decision still there after compression
        assert len(cm.key_decisions) == 1
        assert cm.key_decisions[0]["symbol"] == "TSLA"

    def test_stats_reporting(self):
        cm = ContextManager()
        cm.add_message("user", "test")
        stats = cm.get_stats()
        assert "total_messages" in stats
        assert "capacity_pct" in stats


class TestRetryManager:
    def test_allows_first_retry(self):
        rm = RetryManager(max_retries=3)
        assert rm.should_retry("task-1") is True

    def test_blocks_after_max_retries(self):
        rm = RetryManager(max_retries=2)
        rm.record_attempt("task-1", "TOOL_EXECUTION_FAILED", "API timeout", "retry")
        rm.record_attempt("task-1", "TOOL_EXECUTION_FAILED", "API timeout", "retry")
        assert rm.should_retry("task-1") is False

    def test_retry_context_has_history(self):
        rm = RetryManager(max_retries=3)
        rm.record_attempt("task-1", "INVALID_TOOL_INPUT", "bad symbol", "fix input")
        ctx = rm.get_retry_context("task-1")
        assert ctx["total_attempts"] == 1
        assert ctx["remaining"] == 2
        assert len(ctx["history"]) == 1

    def test_suggestion_changes_with_failures(self):
        rm = RetryManager(max_retries=3)
        rm.record_attempt("t1", "TOOL_EXECUTION_FAILED", "err1", "retry")
        rm.record_attempt("t1", "TOOL_EXECUTION_FAILED", "err2", "retry")
        ctx = rm.get_retry_context("t1")
        assert "escalating" in ctx["suggestion"].lower() or "fundamental" in ctx["suggestion"].lower()


class TestEscalationManager:
    def test_create_escalation(self):
        em = EscalationManager()
        esc = em.escalate(
            reason="Risk score 0.85 exceeds threshold",
            task_context={"symbol": "TSLA", "risk_score": 0.85},
            severity="high",
        )
        assert esc["severity"] == "high"
        assert esc["status"] == "pending_review"

    def test_resolve_escalation(self):
        em = EscalationManager()
        esc = em.escalate(reason="test", task_context={})
        result = em.resolve(esc["escalation_id"], "Approved after review", approved=True)
        assert result["status"] == "approved"

    def test_pending_escalations(self):
        em = EscalationManager()
        em.escalate(reason="test1", task_context={})
        em.escalate(reason="test2", task_context={})
        assert len(em.get_pending()) == 2


# ═══════════════════════════════════════════════════════════════════
# DOMAIN 3 TESTS: Configuration
# ═══════════════════════════════════════════════════════════════════

class TestConfig:
    def test_defaults_loaded(self):
        s = Settings()
        assert s.risk_threshold == 0.7
        assert s.max_position_usd == 10000.0
        assert s.coordinator_max_iterations == 15

    def test_risk_threshold_bounds(self):
        """Risk threshold must be 0-1."""
        with pytest.raises(Exception):
            Settings(risk_threshold=1.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

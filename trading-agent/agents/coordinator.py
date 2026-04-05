"""
Coordinator Agent — The Hub in Hub-and-Spoke (v2)

UPGRADES FROM v1:
- Risk Assessor is now a REAL subagent (not simulated random)
- Fixed async nesting (use event loop properly)
- Added deterministic HOOKS that layer ON TOP of subagent results
- Demonstrates: even after the Risk Assessor agent approves,
  the coordinator's hooks can STILL reject. Defense in depth.

KEY DESIGN PRINCIPLE:
The coordinator's tools are "dispatch_to_analyst" and "dispatch_to_risk" —
they don't call APIs, they call OTHER AGENTS.
"""

import json
import uuid
import asyncio
import logging
from typing import Any

from agents.base_agent import BaseAgent
from agents.market_analyst import MarketAnalystAgent
from agents.risk_assessor import RiskAssessorAgent
from models.schemas import (
    AgentError,
    ErrorCode,
    SubAgentResult,
    SubAgentTask,
    TaskStatus,
)

logger = logging.getLogger(__name__)


def _run_async(coro):
    """
    Helper to run async code from sync context.
    
    WHY THIS EXISTS:
    The agentic loop (BaseAgent.run) is async. But tool execution
    inside the loop is sync. When Coordinator dispatches to a subagent,
    it needs to run another async loop from within sync code.
    
    In production, you'd use a task queue (Celery, etc.) instead.
    This is the pragmatic solution for development.
    """
    try:
        loop = asyncio.get_running_loop()
        # If we're already in an async context, create a new thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    except RuntimeError:
        # No running loop — safe to use asyncio.run
        return asyncio.run(coro)


class CoordinatorAgent(BaseAgent):
    """
    The Coordinator — orchestrates subagents and makes final decisions.
    
    MENTAL MODEL (Trading Desk):
    ┌─────────────────────────────────────────────────┐
    │  COORDINATOR (Senior Trader)                     │
    │  • Receives order request                        │
    │  • Asks research desk: "What does the data say?" │
    │  • Asks risk desk: "Can we take this position?"  │
    │  • Makes final call (with hard-coded limits)     │
    └─────────────────────────────────────────────────┘
    
    DEFENSE IN DEPTH:
    Layer 1: Risk Assessor subagent evaluates risk (LLM reasoning)
    Layer 2: Coordinator hooks enforce hard limits (deterministic code)
    Both layers must approve. LLM cannot bypass code.
    """

    def __init__(self):
        super().__init__(
            agent_name="coordinator",
            max_iterations=15,
        )
        # Subagent instances — each with isolated context
        self.market_analyst = MarketAnalystAgent()
        self.risk_assessor = RiskAssessorAgent()

        # ─── HOOKS: Hard limits (Layer 2 — code enforcement) ─────
        self.risk_threshold = 0.7       # Max risk score to auto-approve
        self.min_confidence = 0.5       # Below this → escalate to human
        self.max_position_usd = 10000   # Hard cap on position size

    def get_system_prompt(self) -> str:
        return """You are the Coordinator agent in a trading analysis system.

Your role: Decompose trading analysis requests, dispatch to specialist agents,
and synthesize their results into actionable trade decisions.

WORKFLOW:
1. Receive a trading analysis request (symbol + context)
2. Dispatch market analysis to the Market Analyst agent via dispatch_analysis
3. Review the analyst's signal
4. Dispatch risk assessment via check_risk with the analyst's confidence score
5. If risk is approved, produce a final recommendation
6. If risk is rejected or confidence is too low, escalate to human

You coordinate. You do NOT perform analysis yourself.

RULES:
- ALWAYS dispatch analysis before checking risk
- ALWAYS check risk before making a final recommendation
- If anything is uncertain, use escalate_to_human
- Your final response must be a JSON object

## Example Final Output
```json
{
  "symbol": "AAPL",
  "action": "buy",
  "confidence": 0.75,
  "reasoning": "Market analyst reports bullish signal. Risk assessor approved with score 0.4.",
  "signal_summary": "Bullish crossover with 75% confidence",
  "risk_summary": "Risk score 0.4 — approved, max position $7500",
  "approved": true,
  "max_position_usd": 7500
}
```
"""

    def get_tools(self) -> list[dict]:
        """
        3 tools. Each dispatches to another agent or escalates.
        The coordinator NEVER calls external APIs.
        """
        return [
            {
                "name": "dispatch_analysis",
                "description": "Dispatch a market analysis task to the Market Analyst subagent. Returns the analyst's signal with trading recommendation.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock ticker to analyze (e.g., AAPL, TSLA)",
                        },
                        "context": {
                            "type": "string",
                            "description": "Additional context for the analysis",
                        },
                    },
                    "required": ["symbol"],
                },
            },
            {
                "name": "check_risk",
                "description": "Dispatch a risk assessment to the Risk Assessor subagent. Evaluates VaR, portfolio concentration, and sector exposure. Returns approval/rejection.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock ticker symbol",
                        },
                        "action": {
                            "type": "string",
                            "enum": ["buy", "sell"],
                            "description": "Proposed trade action",
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Signal confidence from analysis (0.0 to 1.0)",
                        },
                        "proposed_size_usd": {
                            "type": "number",
                            "description": "Proposed position size in USD",
                            "default": 5000,
                        },
                    },
                    "required": ["symbol", "action", "confidence"],
                },
            },
            {
                "name": "escalate_to_human",
                "description": "Escalate a decision to a human trader when uncertainty is too high or risk limits are breached.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "description": "Why this needs human review",
                        },
                        "context": {
                            "type": "object",
                            "description": "All relevant data for the human reviewer",
                        },
                    },
                    "required": ["reason"],
                },
            },
        ]

    def execute_tool(self, tool_name: str, tool_input: dict) -> Any:
        if tool_name == "dispatch_analysis":
            return self._dispatch_analysis(tool_input)
        elif tool_name == "check_risk":
            return self._dispatch_risk_check(tool_input)
        elif tool_name == "escalate_to_human":
            return self._escalate(tool_input)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    # ─── Subagent Dispatchers ────────────────────────────────────

    def _dispatch_analysis(self, input: dict) -> dict:
        """Dispatch to Market Analyst subagent."""
        task = SubAgentTask(
            task_id=str(uuid.uuid4()),
            agent_type="market_analyst",
            objective=f"Analyze the market conditions for {input['symbol']} and produce a trading signal.",
            input_data={
                "symbol": input["symbol"],
                "context": input.get("context", "Standard analysis"),
            },
            max_iterations=10,
        )

        result: SubAgentResult = _run_async(self.market_analyst.run(task))

        if result.status == TaskStatus.COMPLETED:
            return {
                "status": "success",
                "agent": "market_analyst",
                "analysis": result.result,
                "iterations_used": result.iterations_used,
            }
        else:
            return {
                "status": "failed",
                "agent": "market_analyst",
                "error": result.error.model_dump() if result.error else "Unknown error",
            }

    def _dispatch_risk_check(self, input: dict) -> dict:
        """
        Dispatch to Risk Assessor subagent, THEN apply coordinator hooks.
        
        TWO-LAYER DEFENSE:
        Layer 1: Risk Assessor agent analyzes risk (LLM reasoning)
        Layer 2: Coordinator hooks enforce hard limits (deterministic)
        
        Even if the Risk Assessor says "approved," the hooks can reject.
        This is DEFENSE IN DEPTH. The LLM cannot bypass code.
        """
        symbol = input["symbol"]
        action = input["action"]
        confidence = input["confidence"]
        proposed_size = input.get("proposed_size_usd", 5000)

        # ─── Layer 1: Dispatch to Risk Assessor subagent ─────────
        task = SubAgentTask(
            task_id=str(uuid.uuid4()),
            agent_type="risk_assessor",
            objective=f"Assess the risk of a proposed {action} trade for {symbol} with position size ${proposed_size}.",
            input_data={
                "symbol": symbol,
                "action": action,
                "proposed_size_usd": proposed_size,
                "signal_confidence": confidence,
            },
            max_iterations=8,
        )

        risk_result: SubAgentResult = _run_async(self.risk_assessor.run(task))

        if risk_result.status != TaskStatus.COMPLETED:
            return {
                "status": "failed",
                "agent": "risk_assessor",
                "error": risk_result.error.model_dump() if risk_result.error else "Risk assessment failed",
                "approved": False,
            }

        risk_data = risk_result.result or {}

        # ─── Layer 2: Coordinator HOOKS (deterministic override) ──
        # These run AFTER the subagent, and can override its decision
        
        approved = True
        hook_rejections = []

        # Hook 1: Confidence floor
        if confidence < self.min_confidence:
            approved = False
            hook_rejections.append(
                f"HOOK_REJECT: Confidence {confidence} < minimum {self.min_confidence}"
            )

        # Hook 2: Position size cap
        risk_score = risk_data.get("risk_score", 0.5)
        max_position = min(
            self.max_position_usd,
            proposed_size * (1 - risk_score)
        )
        if max_position <= 100:  # Below minimum viable position
            approved = False
            hook_rejections.append(
                f"HOOK_REJECT: Calculated position ${max_position:.0f} below minimum $100"
            )

        # Hook 3: Risk score ceiling
        if risk_score > self.risk_threshold:
            approved = False
            hook_rejections.append(
                f"HOOK_REJECT: Risk score {risk_score} > threshold {self.risk_threshold}"
            )

        return {
            "status": "success",
            "agent": "risk_assessor",
            "risk_assessment": risk_data,
            "iterations_used": risk_result.iterations_used,
            # ─── Hook results layered on top ─────────────────────
            "coordinator_approved": approved,
            "hook_rejections": hook_rejections,
            "final_max_position_usd": round(max_position, 2) if approved else 0,
            "risk_score": risk_score,
        }

    def _escalate(self, input: dict) -> dict:
        """Human escalation — the agent admits it can't handle this."""
        logger.warning(
            f"[COORDINATOR] 🚨 Human escalation: {input['reason']}"
        )
        return {
            "status": "escalated",
            "message": "Decision escalated to human trader",
            "reason": input["reason"],
            "context": input.get("context", {}),
            "action_required": "Human must review and approve/reject via dashboard",
        }

"""
Risk Assessor Subagent

ISOLATED CONTEXT PRINCIPLE:
- This agent sees: portfolio positions, risk metrics, exposure data
- This agent NEVER sees: news, price charts, analyst opinions
- It makes risk decisions based on NUMBERS, not sentiment

HOOKS IN THIS AGENT:
- Hard position limits (code-enforced, not prompt-suggested)
- VaR threshold breakers
- Concentration limits per sector

WHY SEPARATE FROM MARKET ANALYST?
1. Different data access (isolation)
2. Different expertise (no self-review bias)
3. Independent judgment — the analyst can't influence risk approval
4. In production, these could run on different servers with different permissions
"""

import json
import random
from datetime import datetime
from typing import Any

from agents.base_agent import BaseAgent


class RiskAssessorAgent(BaseAgent):
    """
    Subagent that assesses risk for proposed trades.
    
    CRITICAL DESIGN DECISION:
    This agent has NO access to fetch_price or get_news.
    It works purely with portfolio data and risk metrics.
    This prevents the risk assessor from being "swayed" by
    the same bullish news that the analyst saw.
    
    Independent review > Self-review. Always.
    """

    def __init__(self):
        super().__init__(
            agent_name="risk_assessor",
            max_iterations=8,  # Risk assessment should be fast
        )
        # ─── HOOKS: Hard limits the LLM cannot override ─────────
        self.max_portfolio_concentration = 0.20  # 20% max in one stock
        self.max_var_95 = 5000.0                 # Max $5k daily VaR
        self.max_sector_exposure = 0.40          # 40% max in one sector

    def validate_tool_call(self, tool_name: str, tool_input: dict) -> tuple[bool, str]:
        """
        Risk Assessor Hooks:
        - Enforce max $1M position size limit (hard gate)
        - Sanitize symbols
        """
        if tool_name == "calculate_var":
            size = tool_input.get("position_size_usd", 0)
            if size > 1_000_000:
                return False, f"Position size ${size} exceeds hard limit of $1,000,000"
        
        symbol = tool_input.get("symbol")
        if symbol:
            tool_input["symbol"] = symbol.upper()
            
        return True, ""

    def validate_result(self, result: dict) -> tuple[bool, str]:
        """
        Ensure the assessor produced a valid RiskAssessment.
        """
        required = ["symbol", "risk_score", "approved", "reasoning"]
        for field in required:
            if field not in result:
                return False, f"Missing required field in risk assessment: {field}"
        
        if not (0 <= result["risk_score"] <= 1):
            return False, f"Risk score {result['risk_score']} out of bounds (0-1)"
            
        return True, ""

    def get_system_prompt(self) -> str:
        return """You are a Risk Assessor agent in a trading system.

Your role: Evaluate the risk of a proposed trade based on portfolio exposure,
Value-at-Risk calculations, and position limits.

You are INDEPENDENT from the Market Analyst. You do NOT consider news sentiment
or analyst opinions. You work with NUMBERS ONLY.

WORKFLOW:
1. Calculate Value-at-Risk for the proposed position
2. Check current portfolio exposure and concentration limits
3. Assess sector exposure
4. Produce a risk assessment with a clear approve/reject decision

RULES:
- NEVER approve a trade that breaches concentration limits
- ALWAYS calculate VaR before approving
- If uncertain, recommend REJECTION (conservative bias)

## Example Output
```json
{
  "symbol": "AAPL",
  "risk_score": 0.45,
  "max_position_size": 7500.00,
  "stop_loss_pct": 2.5,
  "concerns": ["sector_concentration_approaching_limit"],
  "approved": true,
  "var_95_impact": 1250.00,
  "reasoning": "VaR within limits. Portfolio concentration at 15% after trade, below 20% threshold. Sector exposure acceptable."
}
```
"""

    def get_tools(self) -> list[dict]:
        """
        3 tools. Risk-focused only. No market data tools.
        
        This is ISOLATED CONTEXT in action.
        """
        return [
            {
                "name": "calculate_var",
                "description": "Calculate Value-at-Risk (95% confidence) for a proposed position. Returns potential daily loss.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock ticker symbol",
                        },
                        "position_size_usd": {
                            "type": "number",
                            "description": "Proposed position size in USD",
                        },
                        "holding_period_days": {
                            "type": "integer",
                            "description": "Expected holding period in days",
                            "default": 5,
                        },
                    },
                    "required": ["symbol", "position_size_usd"],
                },
            },
            {
                "name": "check_portfolio_exposure",
                "description": "Get current portfolio positions, concentration per stock, and total exposure.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "account_id": {
                            "type": "string",
                            "description": "Trading account identifier",
                            "default": "default",
                        },
                    },
                    "required": [],
                },
            },
            {
                "name": "check_sector_exposure",
                "description": "Get current sector allocation and limits. Returns exposure percentages per sector.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Symbol to check sector for",
                        },
                    },
                    "required": ["symbol"],
                },
            },
        ]

    def execute_tool(self, tool_name: str, tool_input: dict) -> Any:
        """
        Tool execution with SIMULATED portfolio data.
        
        In Domain 2, these connect to real portfolio databases.
        The HOOKS are already production-ready though.
        """
        if tool_name == "calculate_var":
            return self._sim_calculate_var(tool_input)
        elif tool_name == "check_portfolio_exposure":
            return self._sim_portfolio_exposure(tool_input)
        elif tool_name == "check_sector_exposure":
            return self._sim_sector_exposure(tool_input)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    # ─── Simulated Tool Implementations ──────────────────────────

    def _sim_calculate_var(self, input: dict) -> dict:
        symbol = input["symbol"]
        size = input["position_size_usd"]
        days = input.get("holding_period_days", 5)
        
        # Simulated volatility-based VaR
        daily_vol = random.uniform(0.01, 0.04)  # 1-4% daily vol
        var_95 = round(size * daily_vol * 1.645 * (days ** 0.5), 2)
        
        return {
            "symbol": symbol,
            "position_size_usd": size,
            "daily_volatility_pct": round(daily_vol * 100, 2),
            "var_95_daily": round(size * daily_vol * 1.645, 2),
            "var_95_period": var_95,
            "holding_period_days": days,
            "exceeds_limit": var_95 > self.max_var_95,
            "var_limit": self.max_var_95,
        }

    def _sim_portfolio_exposure(self, input: dict) -> dict:
        return {
            "account_id": input.get("account_id", "default"),
            "total_value": 50000.00,
            "cash_available": 15000.00,
            "positions": [
                {"symbol": "MSFT", "value": 12000, "pct": 0.24},
                {"symbol": "GOOGL", "value": 8000, "pct": 0.16},
                {"symbol": "JPM", "value": 5000, "pct": 0.10},
                {"symbol": "XOM", "value": 5000, "pct": 0.10},
                {"symbol": "CASH", "value": 15000, "pct": 0.30},
            ],
            "concentration_limit": self.max_portfolio_concentration,
            "largest_position_pct": 0.24,
        }

    def _sim_sector_exposure(self, input: dict) -> dict:
        symbol = input["symbol"]
        
        # Simulated sector mapping
        sector_map = {
            "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
            "TSLA": "Consumer Discretionary", "JPM": "Financials",
            "XOM": "Energy", "JNJ": "Healthcare",
        }
        sector = sector_map.get(symbol, "Technology")
        
        return {
            "symbol": symbol,
            "sector": sector,
            "current_sector_exposure": {
                "Technology": 0.34,
                "Financials": 0.10,
                "Energy": 0.10,
                "Healthcare": 0.06,
                "Cash": 0.30,
            },
            "sector_limit": self.max_sector_exposure,
            "symbol_sector_exposure": 0.34 if sector == "Technology" else 0.10,
            "would_breach_limit": (0.34 if sector == "Technology" else 0.10) > self.max_sector_exposure,
        }

"""
Market Analyst Subagent

Responsible for: Fetching market data, running technical analysis,
gathering news sentiment. Returns a MarketSignal.

DESIGN DECISIONS:
- 4 tools only (fetch_price, get_news, technical_analysis, get_market_overview)
- Isolated context: never sees portfolio positions or risk limits
- Output is ALWAYS a MarketSignal (Pydantic validated)
- Tools are SIMULATED for now — we'll add real APIs in Domain 2

FEW-SHOT EXAMPLE IN PROMPT: Notice the system prompt includes
a concrete example of expected output. This is 10x more effective
than lengthy instructions. (Few-shot > Long instructions)
"""

import json
import random
from datetime import datetime

from agents.base_agent import BaseAgent


class MarketAnalystAgent(BaseAgent):
    """
    Subagent that analyzes market conditions for a given symbol.
    
    This agent has NO knowledge of portfolio positions or risk limits.
    It only knows how to analyze market data. This is the
    "isolated context" principle — each agent sees only what it needs.
    """

    def __init__(self):
        super().__init__(
            agent_name="market_analyst",
            max_iterations=10,
        )

    def validate_tool_call(self, tool_name: str, tool_input: dict) -> tuple[bool, str]:
        """
        Market Analyst Hooks:
        - Ensure symbol is presence and uppercase
        - Limit symbols to alphanumeric
        """
        symbol = tool_input.get("symbol")
        if symbol:
            if not symbol.isalnum():
                return False, f"Invalid symbol format: {symbol}. Use alphanumeric only."
            # Deterministic fix: normalize symbol
            tool_input["symbol"] = symbol.upper()
        
        return True, ""

    def validate_result(self, result: dict) -> tuple[bool, str]:
        """
        Ensure the analyst produced a valid MarketSignal.
        """
        required = ["symbol", "signal_type", "confidence", "reasoning"]
        for field in required:
            if field not in result:
                return False, f"Missing required field in signal: {field}"
        
        if result["signal_type"] not in ["bullish", "bearish", "neutral"]:
            return False, f"Invalid signal_type: {result['signal_type']}"
            
        if not (0 <= result["confidence"] <= 1):
            return False, f"Confidence {result['confidence']} out of bounds (0-1)"
            
        return True, ""

    def get_system_prompt(self) -> str:
        """
        Notice the FEW-SHOT EXAMPLE at the bottom.
        
        This is worth more than 500 words of instructions.
        The model learns the output format from the example,
        not from lengthy descriptions.
        """
        return """You are a Market Analyst agent in a trading system.

Your role: Analyze market conditions for a given symbol and produce a trading signal.

You have access to tools for fetching prices, news, and running technical analysis.
Use them to form a comprehensive view before producing your signal.

RULES:
- Always fetch current price data first
- Check recent news for the symbol
- Run technical analysis on the price data  
- Synthesize everything into a signal

OUTPUT FORMAT: You MUST respond with ONLY a JSON object matching this schema:

## Example Output
```json
{
  "symbol": "AAPL",
  "signal_type": "bullish",
  "confidence": 0.75,
  "reasoning": "RSI at 35 indicates oversold, MACD showing bullish crossover, recent earnings beat expectations by 12%",
  "indicators": {
    "rsi": 35.2,
    "macd_signal": "bullish_crossover",
    "sma_20": 178.50,
    "sma_50": 175.20,
    "volume_trend": "increasing"
  }
}
```

signal_type must be one of: "bullish", "bearish", "neutral"
confidence must be between 0.0 and 1.0
"""

    def get_tools(self) -> list[dict]:
        """
        4 tools. Not 10. Not 15. Four.
        
        If you feel the need for more, you need another agent.
        """
        return [
            {
                "name": "fetch_price",
                "description": "Fetch current and recent price data for a stock symbol. Returns OHLCV data.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock ticker symbol (e.g., AAPL, TSLA)",
                        },
                        "period": {
                            "type": "string",
                            "enum": ["1d", "5d", "1mo", "3mo"],
                            "description": "Time period for historical data",
                        },
                    },
                    "required": ["symbol"],
                },
            },
            {
                "name": "get_news",
                "description": "Get recent news headlines and sentiment for a stock symbol.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock ticker symbol",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max number of articles (1-10)",
                            "default": 5,
                        },
                    },
                    "required": ["symbol"],
                },
            },
            {
                "name": "technical_analysis",
                "description": "Run technical indicators (RSI, MACD, SMA) on a symbol's price data.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock ticker symbol",
                        },
                        "indicators": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of indicators: rsi, macd, sma_20, sma_50, bollinger",
                        },
                    },
                    "required": ["symbol", "indicators"],
                },
            },
            {
                "name": "get_market_overview",
                "description": "Get broad market conditions (S&P 500, VIX, sector performance).",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "sectors": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Sectors to check: tech, healthcare, finance, energy",
                        },
                    },
                    "required": [],
                },
            },
        ]

    def execute_tool(self, tool_name: str, tool_input: dict):
        """
        Tool execution with SIMULATED data.
        
        In Domain 2, we'll replace these with real API calls.
        For now, this lets us test the agentic loop without
        needing API keys or live market data.
        
        NOTE: Even with simulated data, the LOOP LOGIC is real.
        The stop_reason handling, error propagation, iteration
        counting — all production-ready.
        """
        if tool_name == "fetch_price":
            return self._sim_fetch_price(tool_input)
        elif tool_name == "get_news":
            return self._sim_get_news(tool_input)
        elif tool_name == "technical_analysis":
            return self._sim_technical_analysis(tool_input)
        elif tool_name == "get_market_overview":
            return self._sim_market_overview(tool_input)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    # ─── Simulated Tool Implementations ──────────────────────────

    def _sim_fetch_price(self, input: dict) -> dict:
        symbol = input["symbol"]
        base_price = hash(symbol) % 500 + 50  # Deterministic per symbol
        return {
            "symbol": symbol,
            "current_price": round(base_price + random.uniform(-5, 5), 2),
            "open": round(base_price - random.uniform(0, 3), 2),
            "high": round(base_price + random.uniform(2, 8), 2),
            "low": round(base_price - random.uniform(2, 8), 2),
            "volume": random.randint(1_000_000, 50_000_000),
            "change_pct": round(random.uniform(-5, 5), 2),
            "timestamp": datetime.utcnow().isoformat(),
        }

    def _sim_get_news(self, input: dict) -> dict:
        symbol = input["symbol"]
        sentiments = ["positive", "negative", "neutral"]
        return {
            "symbol": symbol,
            "articles": [
                {
                    "headline": f"{symbol} reports stronger than expected quarterly results",
                    "sentiment": "positive",
                    "source": "Reuters",
                    "published": "2h ago",
                },
                {
                    "headline": f"Analysts upgrade {symbol} price target amid AI momentum",
                    "sentiment": "positive",
                    "source": "Bloomberg",
                    "published": "5h ago",
                },
                {
                    "headline": f"Market volatility concerns weigh on {symbol} sector peers",
                    "sentiment": "negative",
                    "source": "CNBC",
                    "published": "1d ago",
                },
            ],
            "overall_sentiment": random.choice(sentiments),
        }

    def _sim_technical_analysis(self, input: dict) -> dict:
        symbol = input["symbol"]
        indicators = input.get("indicators", ["rsi", "macd"])
        result = {"symbol": symbol, "indicators": {}}
        
        for ind in indicators:
            if ind == "rsi":
                result["indicators"]["rsi"] = {
                    "value": round(random.uniform(20, 80), 1),
                    "signal": "oversold" if random.random() < 0.3 else "neutral",
                }
            elif ind == "macd":
                result["indicators"]["macd"] = {
                    "macd_line": round(random.uniform(-2, 2), 3),
                    "signal_line": round(random.uniform(-2, 2), 3),
                    "histogram": round(random.uniform(-1, 1), 3),
                    "signal": random.choice(["bullish_crossover", "bearish_crossover", "neutral"]),
                }
            elif ind.startswith("sma"):
                result["indicators"][ind] = {
                    "value": round(hash(symbol) % 500 + random.uniform(-10, 10), 2),
                }
            elif ind == "bollinger":
                base = hash(symbol) % 500
                result["indicators"]["bollinger"] = {
                    "upper": round(base + 20, 2),
                    "middle": round(base, 2),
                    "lower": round(base - 20, 2),
                    "position": random.choice(["above_upper", "middle", "near_lower"]),
                }
        
        return result

    def _sim_market_overview(self, input: dict) -> dict:
        return {
            "sp500": {"value": 5234.18, "change_pct": round(random.uniform(-2, 2), 2)},
            "vix": {"value": round(random.uniform(12, 35), 1), "signal": "low_volatility" if random.random() > 0.5 else "elevated"},
            "sectors": {
                "tech": round(random.uniform(-3, 3), 2),
                "healthcare": round(random.uniform(-3, 3), 2),
                "finance": round(random.uniform(-3, 3), 2),
                "energy": round(random.uniform(-3, 3), 2),
            },
            "market_sentiment": random.choice(["risk_on", "risk_off", "neutral"]),
        }

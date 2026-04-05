"""
Market Data Tools — Production-Ready Tool Implementations

This file demonstrates CORRECT tool design:

1. CLEAR NAMES: fetch_stock_price, not get_data
2. VALIDATED INPUTS: Pydantic models, not raw dicts
3. STRUCTURED ERRORS: Typed error codes with recovery suggestions
4. DOCUMENTED: Description tells the LLM exactly what to expect
5. BOUNDED: Each tool does ONE thing well

In production, these would call real APIs:
- Alpha Vantage / Yahoo Finance for prices
- Financial Modeling Prep for fundamentals
- NewsAPI / Benzinga for news
- Custom calculation engine for technicals

For now, they return realistic simulated data so we can
test the full pipeline without API keys.
"""

import random
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from tools.registry import ToolDefinition, ToolRegistry


# ─── Input Validation Models ────────────────────────────────────────
# These ensure the LLM sends correct data. If it doesn't,
# the ToolRegistry catches the error and sends a structured
# rejection BACK to the LLM with instructions to fix it.


class FetchPriceInput(BaseModel):
    """Validated input for price fetching."""
    symbol: str = Field(min_length=1, max_length=10, description="Stock ticker")
    period: str = Field(default="1d", pattern="^(1d|5d|1mo|3mo)$")


class GetNewsInput(BaseModel):
    """Validated input for news fetching."""
    symbol: str = Field(min_length=1, max_length=10)
    limit: int = Field(default=5, ge=1, le=20)


class TechnicalAnalysisInput(BaseModel):
    """Validated input for technical analysis."""
    symbol: str = Field(min_length=1, max_length=10)
    indicators: list[str] = Field(
        default=["rsi", "macd"],
        description="Available: rsi, macd, sma_20, sma_50, bollinger, vwap"
    )


class CalculateVarInput(BaseModel):
    """Validated input for VaR calculation."""
    symbol: str = Field(min_length=1, max_length=10)
    position_size_usd: float = Field(gt=0, le=1_000_000, description="Max $1M")
    holding_period_days: int = Field(default=5, ge=1, le=252)


class PortfolioExposureInput(BaseModel):
    """Validated input for portfolio exposure check."""
    account_id: str = Field(default="default")


class SectorExposureInput(BaseModel):
    """Validated input for sector exposure check."""
    symbol: str = Field(min_length=1, max_length=10)


# ─── Tool Handler Functions ─────────────────────────────────────────
# These are the ACTUAL tool implementations.
# Clean, focused, one-thing-only functions.


def fetch_stock_price(symbol: str, period: str = "1d") -> dict:
    """Fetch OHLCV price data for a stock."""
    base = abs(hash(symbol)) % 500 + 50
    return {
        "symbol": symbol.upper(),
        "current_price": round(base + random.uniform(-5, 5), 2),
        "open": round(base - random.uniform(0, 3), 2),
        "high": round(base + random.uniform(2, 8), 2),
        "low": round(base - random.uniform(2, 8), 2),
        "close": round(base + random.uniform(-3, 3), 2),
        "volume": random.randint(1_000_000, 50_000_000),
        "change_pct": round(random.uniform(-5, 5), 2),
        "period": period,
        "timestamp": datetime.utcnow().isoformat(),
    }


def get_stock_news(symbol: str, limit: int = 5) -> dict:
    """Get recent news with sentiment for a stock."""
    headlines = [
        (f"{symbol} reports stronger than expected quarterly results", "positive"),
        (f"Analysts upgrade {symbol} price target amid growth momentum", "positive"),
        (f"Market volatility concerns weigh on {symbol} sector", "negative"),
        (f"{symbol} announces strategic partnership in AI space", "positive"),
        (f"Regulatory concerns loom for {symbol} industry", "negative"),
        (f"{symbol} trading volume surges on institutional buying", "neutral"),
    ]
    selected = random.sample(headlines, min(limit, len(headlines)))
    
    positive = sum(1 for _, s in selected if s == "positive")
    negative = sum(1 for _, s in selected if s == "negative")
    
    return {
        "symbol": symbol.upper(),
        "articles": [
            {"headline": h, "sentiment": s, "source": random.choice(["Reuters", "Bloomberg", "CNBC", "WSJ"])}
            for h, s in selected
        ],
        "sentiment_summary": {
            "positive": positive,
            "negative": negative,
            "neutral": len(selected) - positive - negative,
        },
        "overall_sentiment": "positive" if positive > negative else "negative" if negative > positive else "neutral",
    }


def run_technical_analysis(symbol: str, indicators: list[str] = None) -> dict:
    """Run technical indicators on a symbol."""
    indicators = indicators or ["rsi", "macd"]
    result = {"symbol": symbol.upper(), "indicators": {}, "timestamp": datetime.utcnow().isoformat()}
    
    for ind in indicators:
        if ind == "rsi":
            val = round(random.uniform(20, 80), 1)
            result["indicators"]["rsi"] = {
                "value": val,
                "signal": "oversold" if val < 30 else "overbought" if val > 70 else "neutral",
            }
        elif ind == "macd":
            result["indicators"]["macd"] = {
                "macd_line": round(random.uniform(-2, 2), 3),
                "signal_line": round(random.uniform(-2, 2), 3),
                "histogram": round(random.uniform(-1, 1), 3),
                "signal": random.choice(["bullish_crossover", "bearish_crossover", "neutral"]),
            }
        elif ind in ("sma_20", "sma_50"):
            base = abs(hash(symbol)) % 500 + 50
            result["indicators"][ind] = {
                "value": round(base + random.uniform(-15, 15), 2),
            }
        elif ind == "bollinger":
            base = abs(hash(symbol)) % 500 + 50
            result["indicators"]["bollinger"] = {
                "upper": round(base + 20, 2),
                "middle": round(base, 2),
                "lower": round(base - 20, 2),
                "position": random.choice(["above_upper", "middle", "near_lower"]),
            }
        elif ind == "vwap":
            base = abs(hash(symbol)) % 500 + 50
            result["indicators"]["vwap"] = {
                "value": round(base + random.uniform(-5, 5), 2),
                "price_vs_vwap": random.choice(["above", "below", "at"]),
            }
    
    return result


def calculate_value_at_risk(
    symbol: str,
    position_size_usd: float,
    holding_period_days: int = 5,
) -> dict:
    """Calculate VaR (95% confidence) for a position."""
    daily_vol = random.uniform(0.01, 0.04)
    var_95 = round(position_size_usd * daily_vol * 1.645 * (holding_period_days ** 0.5), 2)
    
    return {
        "symbol": symbol.upper(),
        "position_size_usd": position_size_usd,
        "daily_volatility_pct": round(daily_vol * 100, 2),
        "var_95_daily_usd": round(position_size_usd * daily_vol * 1.645, 2),
        "var_95_period_usd": var_95,
        "holding_period_days": holding_period_days,
        "max_var_limit_usd": 5000.0,
        "within_limits": var_95 <= 5000.0,
    }


def check_portfolio_exposure(account_id: str = "default") -> dict:
    """Get current portfolio positions and concentration."""
    return {
        "account_id": account_id,
        "total_value_usd": 50000.0,
        "cash_available_usd": 15000.0,
        "positions": [
            {"symbol": "MSFT", "value_usd": 12000, "weight_pct": 24.0},
            {"symbol": "GOOGL", "value_usd": 8000, "weight_pct": 16.0},
            {"symbol": "JPM", "value_usd": 5000, "weight_pct": 10.0},
            {"symbol": "XOM", "value_usd": 5000, "weight_pct": 10.0},
        ],
        "concentration_limit_pct": 20.0,
        "diversification_score": 0.72,
    }


def check_sector_exposure(symbol: str) -> dict:
    """Check sector allocation and concentration limits."""
    sector_map = {
        "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
        "NVDA": "Technology", "TSLA": "Consumer Discretionary",
        "JPM": "Financials", "GS": "Financials",
        "XOM": "Energy", "CVX": "Energy",
        "JNJ": "Healthcare", "PFE": "Healthcare",
    }
    sector = sector_map.get(symbol.upper(), "Technology")
    
    exposures = {
        "Technology": 34.0, "Financials": 10.0,
        "Energy": 10.0, "Healthcare": 6.0, "Cash": 30.0,
    }
    
    return {
        "symbol": symbol.upper(),
        "sector": sector,
        "current_exposure_pct": exposures,
        "symbol_sector_pct": exposures.get(sector, 0),
        "sector_limit_pct": 40.0,
        "within_limits": exposures.get(sector, 0) <= 40.0,
    }


# ─── Registry Setup ─────────────────────────────────────────────────

def create_market_tools_registry() -> ToolRegistry:
    """
    Create and populate the tool registry for market analysis tools.
    
    Notice the SEPARATION:
    - ToolDefinition = what the LLM sees + runtime config
    - Handler = the actual function
    - Input model = Pydantic validation
    
    All three registered together. All three enforced by the registry.
    """
    registry = ToolRegistry()

    # ─── Market Analyst tools ────────────────────────────────────
    registry.register(
        definition=ToolDefinition(
            name="fetch_stock_price",
            description="Fetch current and recent OHLCV price data for a stock symbol. Returns current price, open, high, low, close, volume, and percentage change.",
            input_schema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker (e.g., AAPL)"},
                    "period": {"type": "string", "enum": ["1d", "5d", "1mo", "3mo"], "default": "1d"},
                },
                "required": ["symbol"],
            },
            rate_limit_per_minute=30,
            timeout_seconds=10.0,
            category="market_data",
        ),
        handler=fetch_stock_price,
        input_model=FetchPriceInput,
    )

    registry.register(
        definition=ToolDefinition(
            name="get_stock_news",
            description="Get recent news headlines with sentiment analysis for a stock. Returns articles with positive/negative/neutral classification.",
            input_schema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker"},
                    "limit": {"type": "integer", "default": 5, "description": "Max articles (1-20)"},
                },
                "required": ["symbol"],
            },
            rate_limit_per_minute=20,
            timeout_seconds=15.0,
            category="market_data",
        ),
        handler=get_stock_news,
        input_model=GetNewsInput,
    )

    registry.register(
        definition=ToolDefinition(
            name="run_technical_analysis",
            description="Run technical indicators (RSI, MACD, SMA, Bollinger, VWAP) on a stock. Returns indicator values and signals.",
            input_schema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker"},
                    "indicators": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Indicators: rsi, macd, sma_20, sma_50, bollinger, vwap",
                    },
                },
                "required": ["symbol", "indicators"],
            },
            rate_limit_per_minute=30,
            timeout_seconds=10.0,
            category="market_data",
        ),
        handler=run_technical_analysis,
        input_model=TechnicalAnalysisInput,
    )

    # ─── Risk Assessment tools ───────────────────────────────────
    registry.register(
        definition=ToolDefinition(
            name="calculate_value_at_risk",
            description="Calculate Value-at-Risk (95% confidence) for a proposed position. Returns potential daily and period loss.",
            input_schema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "position_size_usd": {"type": "number", "description": "Position size in USD (max $1M)"},
                    "holding_period_days": {"type": "integer", "default": 5},
                },
                "required": ["symbol", "position_size_usd"],
            },
            rate_limit_per_minute=30,
            timeout_seconds=5.0,
            category="risk",
        ),
        handler=calculate_value_at_risk,
        input_model=CalculateVarInput,
    )

    registry.register(
        definition=ToolDefinition(
            name="check_portfolio_exposure",
            description="Get current portfolio positions, concentration percentages, and available cash.",
            input_schema={
                "type": "object",
                "properties": {
                    "account_id": {"type": "string", "default": "default"},
                },
                "required": [],
            },
            rate_limit_per_minute=60,
            timeout_seconds=5.0,
            category="risk",
        ),
        handler=check_portfolio_exposure,
        input_model=PortfolioExposureInput,
    )

    registry.register(
        definition=ToolDefinition(
            name="check_sector_exposure",
            description="Check current sector allocation and whether adding a position would breach concentration limits.",
            input_schema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                },
                "required": ["symbol"],
            },
            rate_limit_per_minute=60,
            timeout_seconds=5.0,
            category="risk",
        ),
        handler=check_sector_exposure,
        input_model=SectorExposureInput,
    )

    return registry

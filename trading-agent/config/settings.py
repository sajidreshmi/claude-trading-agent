"""
Application Configuration — Environment-Driven, Never Hardcoded

PRODUCTION PRINCIPLE:
Configuration lives OUTSIDE the code. The same Docker image
runs in dev, staging, and prod — only config changes.

Pydantic Settings handles:
- Environment variable loading
- Type validation
- Default values
- .env file support
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    """
    All configuration in one place. No magic strings scattered
    across the codebase. No "TODO: move to config" comments.
    """
    
    # ─── LLM Configuration ───────────────────────────────────────
    anthropic_api_key: str = Field(default="", description="Claude API key")
    llm_model: str = Field(default="claude-sonnet-4-20250514", description="Claude model to use")
    llm_max_tokens: int = Field(default=4096, ge=256, le=8192)
    
    # ─── Agent Configuration ─────────────────────────────────────
    coordinator_max_iterations: int = Field(default=15, ge=1, le=50)
    subagent_max_iterations: int = Field(default=10, ge=1, le=25)
    
    # ─── Risk Hooks (deterministic, NOT LLM-configurable) ────────
    risk_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    max_position_usd: float = Field(default=10000.0, gt=0)
    max_portfolio_concentration: float = Field(default=0.20, ge=0.0, le=1.0)
    max_sector_exposure: float = Field(default=0.40, ge=0.0, le=1.0)
    
    # ─── Rate Limiting ───────────────────────────────────────────
    tool_rate_limit_per_minute: int = Field(default=30, ge=1)
    circuit_breaker_threshold: int = Field(default=3, ge=1)
    circuit_breaker_recovery_sec: float = Field(default=60.0, gt=0)
    
    # ─── Server ──────────────────────────────────────────────────
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1024, le=65535)
    debug: bool = Field(default=False)
    
    # ─── Database (Domain 5 — placeholder) ───────────────────────
    database_url: Optional[str] = Field(default=None)
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


# Singleton — import this everywhere
settings = Settings()

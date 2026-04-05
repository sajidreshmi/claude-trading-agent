# CLAUDE.md — Trading Agent Project Guide

## Overview
A production-grade, multi-agent trading analysis system built on the Claude Architect foundations (Hub-and-Spoke, Tool Registry, Context Management).

## Building & Running
- **Setup**: `pip install -r requirements.txt`
- **Environment**: Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY`.
- **Run CLI**: `python run.py <SYMBOL>`
- **Run Server**: `uvicorn server:app --reload` (UI at http://localhost:8000/docs)
- **Docker**: `docker compose up --build`

## Testing
- **Suite**: `pytest tests/test_system.py -v` (33 tests covering all domains)
- **Coverage**: Includes Schema validation, Tool Registry, Context Management, Config, and Escalation.

## Code Style & Patterns
- **Hooks > Prompts**: Never rely on LLM prompts for safety. Use deterministic code hooks (e.g., `validate_tool_call`, `validate_result`).
- **Structured Errors**: Use `ErrorCode` enum and `AgentError` schema for ALL failures.
- **Agentic Loop**: Use `stop_reason` from the Anthropic SDK. Do NOT parse text for "finished" or "tool_use".
- **Isolated Context**: Subagents only see data relevant to their role (Market Analyst has no portfolio data).
- **Few-Shot Examples**: Always include valid JSON output examples in system prompts.
- **Tool Limits**: Keep tools per agent to 4-5 max. Use `ToolRegistry` for execution.

## System Architecture (5 Domains)
1. **Agentic Architecture**: Hub-and-Spoke (`Coordinator` + subagents).
2. **Tool Design**: Central `ToolRegistry` with rate limiting and circuit breakers.
3. **Architecting Workflows**: Async background job pattern (`202 Accepted` + polling).
4. **Prompt Engineering**: Versioned templates, few-shot, role-based constraints.
5. **Context Management**: Sliding window summarization, retry manager, human escalation.

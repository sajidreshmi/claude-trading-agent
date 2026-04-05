---
context: fork
allowed-tools: [view_file, list_dir, grep_search]
argument-hint: "Which agent or component of the trading system do you want to explore?"
---
# Agent Skills — Trading System Intelligence

## Overview
This skill implements the intelligence layer of the trading system, featuring a **Hub-and-Spoke** orchestration model that separates market analysis from risk assessment.

### Core Agent Capabilities
1. **Coordinator Agent (The Hub)**: Orchestrates tasks, dispatches to subagents, synthesizes final decisions, and implements deterministic hooks for risk limits and confidence.
2. **Market Analyst Agent (The Specialist)**: Focused on symbols and sentiment. Has access to price, news, and technical analysis tools.
3. **Risk Assessor Agent (The Auditor)**: Independent reviewer. Has access to portfolio, sector, and VaR tools. **Isolated from market analyst context** to prevent self-review bias.

---

## Skills Breakdown

### Skill 1: Deterministic Agentic Loop (`BaseAgent`)
Every agent in this system follows a standardized loop that handles tool use and task completion using Claude's `stop_reason` API.

- **Deterministic**: No text parsing. Relies on API status codes.
- **Resilient**: Implements `max_iterations` and structured error handling for ALL failures.
- **Extensible**: Supports pre-execution (`validate_tool_call`) and post-completion (`validate_result`) hooks.

### Skill 2: Context Management & Reliability
Handles long-running tasks and conversation growth.

- **ContextManager**: Implements sliding window summarization to stay within the 200k token limit while preserving **Key Decisions**.
- **RetryManager**: Provides specific feedback for failed attempts (e.g., "Invalid symbol format") instead of generic "try again".
- **EscalationManager**: Deterministically hand-offs tasks to humans for review based on risk scores, confidence, or repeated failures.

### Skill 3: Orchestration & Delegation
The Hub-and-Spoke model ensures a single point of responsibility (the Coordinator) while delegating specialized tasks to isolated subagents.

- **Delegation**: `Coordinator` never fetches data directly. It only dispatches.
- **Synthesis**: Combines conflicting signals (e.g., Bullish Analyst + High Risk Assessor) into a final, safe "REJECT" or "ADJUSTED_BUY" decision.

---

## Instructions for Agent Training (Domain 4)
All agent system prompts follow the **Claude Certified Architect** pattern:
- **Role**: Clear persona (e.g., "You are a Risk Assessor agent...").
- **Constraints**: Hard rules (e.g., "NEVER approve a trade that breaches limits").
- **Few-Shot Examples**: Valid JSON output blocks to ensure zero-shot success.
- **Hooks > Prompts**: The agents are told about their limits, but the code *enforces* them.

---

## Skill Files
- [base_agent.py](file:///Users/sajid/Documents/Tech-LnD/AI/claude-architect-exam/trading-agent/agents/base_agent.py)
- [coordinator.py](file:///Users/sajid/Documents/Tech-LnD/AI/claude-architect-exam/trading-agent/agents/coordinator.py)
- [market_analyst.py](file:///Users/sajid/Documents/Tech-LnD/AI/claude-architect-exam/trading-agent/agents/market_analyst.py)
- [risk_assessor.py](file:///Users/sajid/Documents/Tech-LnD/AI/claude-architect-exam/trading-agent/agents/risk_assessor.py)
- [context_manager.py](file:///Users/sajid/Documents/Tech-LnD/AI/claude-architect-exam/trading-agent/agents/context_manager.py)

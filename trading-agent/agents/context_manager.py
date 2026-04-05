"""
Context Manager — Long-Running Workflow Context Strategy

DOMAIN 5: CONTEXT MANAGEMENT & RELIABILITY

THE PROBLEM:
Agent conversations grow with every tool call. After 5-10 iterations,
you hit the context window limit. Without a strategy, the agent either:
- Crashes (context too long)
- Forgets earlier decisions (truncated history)
- Gets confused (too much noise)

THE SOLUTION: Sliding window + summarization
- Keep the last N messages in full
- Summarize everything before that
- Preserve KEY DECISIONS even in summaries
- Track token usage and trigger summarization proactively

PRODUCTION CONCERN:
In trading, forgetting a previous decision is DANGEROUS.
"I already rejected AAPL due to high risk" must survive summarization.
"""

import json
import logging
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ContextManager:
    """
    Manages conversation context for long-running agent workflows.
    
    Strategy:
    ┌───────────────────────────────────────────────────────┐
    │                  CONTEXT WINDOW                        │
    │                                                        │
    │  [SUMMARY of old messages]  [Recent messages in full]  │
    │  ◄── compressed ──►         ◄── current window ──►     │
    │                                                        │
    │  KEY DECISIONS preserved    Full detail preserved       │
    │  Raw data dropped          Tool calls + results shown  │
    └───────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        max_messages: int = 20,
        summary_threshold: int = 15,
        max_chars_estimate: int = 100_000,  # ~25k tokens
    ):
        self.max_messages = max_messages
        self.summary_threshold = summary_threshold
        self.max_chars_estimate = max_chars_estimate
        
        self.messages: list[dict] = []
        self.summary: Optional[str] = None
        self.key_decisions: list[dict] = []
        self._total_chars = 0

    def add_message(self, role: str, content) -> None:
        """Add a message and check if summarization is needed."""
        msg = {"role": role, "content": content}
        self.messages.append(msg)
        
        # Estimate character count
        if isinstance(content, str):
            self._total_chars += len(content)
        elif isinstance(content, list):
            self._total_chars += sum(
                len(json.dumps(block)) if isinstance(block, dict) else len(str(block))
                for block in content
            )

        # Check if we need to summarize
        if self._should_summarize():
            self._compress()

    def record_decision(self, decision: dict) -> None:
        """
        Record a KEY DECISION that must survive summarization.
        
        These are NEVER dropped. Even when we compress 50 messages
        into a summary, every decision recorded here stays intact.
        """
        self.key_decisions.append({
            **decision,
            "timestamp": datetime.utcnow().isoformat(),
        })

    def get_messages(self) -> list[dict]:
        """
        Get the current message list for the LLM.
        
        If we have a summary, prepend it as a system-like context.
        """
        if self.summary:
            summary_msg = {
                "role": "user",
                "content": f"""## Previous Context (summarized)
{self.summary}

## Key Decisions Made
{json.dumps(self.key_decisions, indent=2) if self.key_decisions else "None yet."}

## Continue from here
Continue with the current task based on the above context.
""",
            }
            return [summary_msg] + self.messages
        return self.messages

    def get_stats(self) -> dict:
        """Context usage statistics for monitoring."""
        return {
            "total_messages": len(self.messages),
            "has_summary": self.summary is not None,
            "key_decisions_count": len(self.key_decisions),
            "estimated_chars": self._total_chars,
            "estimated_tokens": self._total_chars // 4,  # Rough estimate
            "capacity_pct": round(
                self._total_chars / self.max_chars_estimate * 100, 1
            ),
        }

    def _should_summarize(self) -> bool:
        """Check if we need to compress the context."""
        return (
            len(self.messages) > self.summary_threshold
            or self._total_chars > self.max_chars_estimate * 0.8
        )

    def _compress(self) -> None:
        """
        Compress older messages into a summary.
        
        In production, this would call the LLM to summarize.
        Here we do a deterministic extraction as a fallback.
        
        The LLM-based summarization uses prompts/templates.py
        context_summary_prompt() — but we want a fallback that
        works WITHOUT an LLM call for reliability.
        """
        if len(self.messages) <= 5:
            return  # Not enough to compress

        # Keep the last 5 messages in full
        old_messages = self.messages[:-5]
        self.messages = self.messages[-5:]

        # Extract key information from old messages
        extracted = []
        for msg in old_messages:
            content = msg.get("content", "")
            if isinstance(content, str) and len(content) > 0:
                # Keep first 200 chars of each message
                extracted.append(f"[{msg['role']}]: {content[:200]}...")
            elif isinstance(content, list):
                # Extract tool results
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "tool_result":
                            result_preview = str(block.get("content", ""))[:150]
                            extracted.append(f"[tool_result]: {result_preview}...")

        # Build summary
        self.summary = "\n".join(extracted[-10:])  # Keep last 10 extracts
        
        # Recalculate char count
        self._total_chars = sum(
            len(json.dumps(msg)) if isinstance(msg.get("content"), list) 
            else len(str(msg.get("content", "")))
            for msg in self.messages
        )
        
        logger.info(
            f"Context compressed: {len(old_messages)} old messages → summary. "
            f"Keeping {len(self.messages)} recent + {len(self.key_decisions)} decisions"
        )


class RetryManager:
    """
    Manages retry logic with SPECIFIC error feedback.
    
    PRODUCTION PRINCIPLE:
    "Try again" is useless. "This failed because the symbol was
    invalid, try with a different format" is actionable.
    
    Each retry gets:
    - What failed
    - Why it failed
    - What to try differently
    - How many attempts remain
    """

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self._attempts: dict[str, list[dict]] = {}

    def should_retry(self, task_id: str) -> bool:
        """Check if we have retries remaining."""
        attempts = self._attempts.get(task_id, [])
        return len(attempts) < self.max_retries

    def record_attempt(
        self,
        task_id: str,
        error_code: str,
        error_message: str,
        action_taken: str,
    ) -> None:
        """Record a failed attempt with details."""
        if task_id not in self._attempts:
            self._attempts[task_id] = []

        self._attempts[task_id].append({
            "attempt": len(self._attempts[task_id]) + 1,
            "error_code": error_code,
            "error_message": error_message,
            "action_taken": action_taken,
            "timestamp": datetime.utcnow().isoformat(),
        })

    def get_retry_context(self, task_id: str) -> dict:
        """
        Get full retry context for the next attempt.
        
        This is fed back to the agent so it knows:
        - What was tried before
        - What failed each time
        - What to try differently
        """
        attempts = self._attempts.get(task_id, [])
        return {
            "total_attempts": len(attempts),
            "max_retries": self.max_retries,
            "remaining": self.max_retries - len(attempts),
            "history": attempts,
            "suggestion": self._suggest_action(attempts),
        }

    def _suggest_action(self, attempts: list[dict]) -> str:
        """Generate a specific suggestion based on failure history."""
        if not attempts:
            return "First attempt — proceed normally"

        last_error = attempts[-1]["error_code"]
        count = len(attempts)

        suggestions = {
            "TOOL_EXECUTION_FAILED": "The tool had an execution error. Try with different parameters.",
            "INVALID_TOOL_INPUT": "Input validation failed. Check parameter types and ranges.",
            "LLM_API_ERROR": "API call failed. This may be transient — retry is appropriate.",
            "VALIDATION_FAILED": "Output validation failed. Ensure response matches the expected JSON schema.",
            "RATE_LIMITED": "Rate limit hit. Wait before retrying.",
        }

        base = suggestions.get(last_error, "Review the error details and adjust approach")

        if count >= 2:
            base += ". Multiple failures suggest a fundamental issue — consider escalating to human."

        return base


class EscalationManager:
    """
    Manages escalation from agent → human.
    
    WHEN TO ESCALATE (deterministic hooks, not LLM judgment):
    1. Confidence below threshold (min_confidence)
    2. Risk score above threshold (risk_threshold)  
    3. Max retries exhausted
    4. Contradictory signals from subagents
    5. Novel situation not covered by training data
    
    ESCALATION DATA:
    The human reviewer gets EVERYTHING needed to decide:
    - Original task
    - Agent's analysis
    - Why the agent couldn't decide
    - Recommended action (if any)
    """

    def __init__(self):
        self._escalations: list[dict] = []

    def escalate(
        self,
        reason: str,
        task_context: dict,
        agent_recommendation: Optional[str] = None,
        severity: str = "medium",  # low, medium, high, critical
    ) -> dict:
        """
        Create an escalation record.
        
        In production, this would:
        - Push to a message queue
        - Send a Slack/email alert
        - Create a ticket in the escalation dashboard
        - Log to the audit trail
        """
        escalation = {
            "escalation_id": f"ESC-{len(self._escalations) + 1:04d}",
            "reason": reason,
            "severity": severity,
            "task_context": task_context,
            "agent_recommendation": agent_recommendation,
            "status": "pending_review",
            "created_at": datetime.utcnow().isoformat(),
            "resolved_at": None,
            "resolution": None,
        }

        self._escalations.append(escalation)

        logger.warning(
            f"🚨 ESCALATION {escalation['escalation_id']}: "
            f"[{severity.upper()}] {reason}"
        )

        return escalation

    def resolve(
        self,
        escalation_id: str,
        resolution: str,
        approved: bool,
    ) -> Optional[dict]:
        """Human resolves an escalation."""
        for esc in self._escalations:
            if esc["escalation_id"] == escalation_id:
                esc["status"] = "approved" if approved else "rejected"
                esc["resolution"] = resolution
                esc["resolved_at"] = datetime.utcnow().isoformat()
                return esc
        return None

    def get_pending(self) -> list[dict]:
        """Get all pending escalations for human review."""
        return [e for e in self._escalations if e["status"] == "pending_review"]

    def get_all(self) -> list[dict]:
        """Get full escalation history."""
        return self._escalations

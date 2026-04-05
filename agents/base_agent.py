"""
Base Agent — The Production Agentic Loop

THIS IS THE MOST IMPORTANT FILE IN THE SYSTEM.

The agentic loop pattern:
1. Send messages + tool definitions to Claude
2. Check stop_reason (DETERMINISTIC, not text parsing)
3. If stop_reason == "tool_use" → execute tools, append results, loop
4. If stop_reason == "end_turn" → agent is done, extract result
5. If iterations exceeded → structured error, not silent failure

ANTI-PATTERNS THIS AVOIDS:
- ❌ Parsing "DONE" or "FINISHED" from response text
- ❌ Unbounded loops (always has max_iterations)
- ❌ Silent failures (every error is structured)
- ❌ Generic exceptions (typed ErrorCode enum)
- ❌ Self-review (agents don't review their own output)
"""

import json
import logging
from typing import Any
from abc import ABC, abstractmethod

import anthropic

from models.schemas import (
    AgentError,
    ErrorCode,
    SubAgentResult,
    SubAgentTask,
    TaskStatus,
)

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base for all agents in the system.
    
    Every agent — coordinator and subagents — inherits this loop.
    The loop is DETERMINISTIC: it uses stop_reason from the API,
    never parses text to decide what to do next.
    """

    def __init__(
        self,
        agent_name: str,
        model: str = "claude-sonnet-4-20250514",
        max_iterations: int = 10,
    ):
        self.agent_name = agent_name
        self.model = model
        self.max_iterations = max_iterations
        self.client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Each agent defines its own identity and constraints."""
        ...

    @abstractmethod
    def get_tools(self) -> list[dict]:
        """
        Each agent defines 4-5 tools MAX.
        
        Why 4-5? More tools = worse tool selection accuracy.
        Anthropic's own benchmarks show degradation above 5-6 tools.
        If you need more, you need another agent.
        """
        ...

    @abstractmethod
    def execute_tool(self, tool_name: str, tool_input: dict) -> Any:
        """
        Execute a tool call. This is where HOOKS live.
        
        Example hook: if tool_name == "place_order" and risk > threshold:
            return error, don't execute. This is DETERMINISTIC.
            The LLM doesn't decide — code does.
        """
        ...

    def validate_tool_call(self, tool_name: str, tool_input: dict) -> tuple[bool, str]:
        """
        PRE-EXECUTION HOOK — runs BEFORE every tool call.
        
        Override in subagents to add deterministic gates:
        - Symbol allowlists
        - Position size caps
        - Blocked operations during market hours
        
        Returns:
            (allowed: bool, reason: str)
            If allowed is False, the tool call is BLOCKED and the
            reason is sent back to the LLM as an error.
        """
        return True, ""

    def pre_tool_use(self, tool_name: str, tool_input: dict) -> dict:
        """
        PRE-TOOL-USE HOOK — runs AFTER validation but BEFORE tool execution.
        
        Override in subagents to modify tool inputs before execution 
        (e.g., injecting hidden auth tokens, forcing canonical date formats).
        """
        return tool_input

    def post_tool_use(self, tool_name: str, tool_input: dict, result: Any) -> Any:
        """
        POST-TOOL-USE HOOK — runs AFTER tool execution but BEFORE sending to LLM.
        
        Override in subagents to manipulate the output:
        - Redact Personally Identifiable Information (PII)
        - Compress extremely large payloads to save token limits
        - Transform raw SQL objects into readable summaries
        """
        return result

    def validate_result(self, result: dict) -> tuple[bool, str]:
        """
        POST-COMPLETION HOOK — runs AFTER the agent produces final output.
        
        Override in subagents to enforce output contracts:
        - Required fields present
        - Values within expected ranges
        - No hallucinated data
        
        Returns:
            (valid: bool, reason: str)
            If invalid, the agent loops and tries again.
        """
        return True, ""

    async def run(self, task: SubAgentTask) -> SubAgentResult:
        """
        THE AGENTIC LOOP — study this carefully.
        
        This is NOT a simple request-response. It's a loop that:
        1. Sends the task to Claude
        2. Claude either responds (end_turn) or requests tools (tool_use)
        3. If tools requested: execute them, feed results back, loop
        4. If response: validate and return structured result
        5. If max iterations hit: return structured error
        
        The stop_reason drives EVERYTHING. No text parsing.
        """
        
        # Initialize conversation with the task
        messages = [
            {
                "role": "user",
                "content": self._format_task_prompt(task),
            }
        ]

        iterations = 0

        while iterations < self.max_iterations:
            iterations += 1
            logger.info(
                f"[{self.agent_name}] Iteration {iterations}/{self.max_iterations}"
            )

            try:
                # ─── Step 1: Call Claude ──────────────────────────
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=self.get_system_prompt(),
                    tools=self.get_tools(),
                    messages=messages,
                )

                # ─── Step 2: Check stop_reason (THE KEY DECISION) ─
                #
                # This is DETERMINISTIC. The API tells us exactly
                # what happened. We never guess.
                #
                # stop_reason == "end_turn"  → Agent is done thinking
                # stop_reason == "tool_use"  → Agent wants to use a tool
                # stop_reason == "max_tokens" → Response was truncated
                #

                if response.stop_reason == "end_turn":
                    # ─── Agent is done. Extract the text result. ──
                    result_text = self._extract_text(response)
                    parsed = self._parse_result(result_text)
                    
                    # ─── POST-COMPLETION HOOK ─────────────────────
                    # Validate the result BEFORE returning it.
                    # This catches malformed output, missing fields,
                    # out-of-range values, etc.
                    valid, reason = self.validate_result(parsed)
                    if not valid and iterations < self.max_iterations:
                        logger.warning(
                            f"[{self.agent_name}] Result validation failed: {reason}"
                        )
                        messages.append({"role": "assistant", "content": response.content})
                        messages.append({
                            "role": "user",
                            "content": f"Your output failed validation: {reason}. "
                                       f"Please fix and respond with corrected JSON.",
                        })
                        continue  # Let the agent try again
                    
                    return SubAgentResult(
                        task_id=task.task_id,
                        status=TaskStatus.COMPLETED,
                        result=parsed,
                        iterations_used=iterations,
                    )

                elif response.stop_reason == "tool_use":
                    # ─── Agent wants to use tools. Execute them. ──
                    
                    # Append assistant's response (includes tool_use blocks)
                    messages.append({"role": "assistant", "content": response.content})

                    # Process EACH tool call in the response
                    tool_results = []
                    for block in response.content:
                        if block.type == "tool_use":
                            logger.info(
                                f"[{self.agent_name}] Tool call: {block.name}"
                            )
                            
                            # ─── PRE-EXECUTION HOOK ──────────────
                            # Check if this tool call is ALLOWED
                            # before running it. Deterministic gate.
                            allowed, block_reason = self.validate_tool_call(
                                block.name, block.input
                            )
                            if not allowed:
                                logger.warning(
                                    f"[{self.agent_name}] Tool call BLOCKED: "
                                    f"{block.name} — {block_reason}"
                                )
                                tool_results.append(
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": block.id,
                                        "content": json.dumps({
                                            "error": f"Tool call blocked by safety hook: {block_reason}",
                                            "error_code": ErrorCode.VALIDATION_FAILED,
                                            "blocked_by": "pre_execution_hook",
                                        }),
                                        "is_error": True,
                                    }
                                )
                                continue
                            
                            # Execute the tool (with HOOKS for safety)
                            try:
                                # ─── PRE-TOOL USE (Modify Input) ──
                                modified_input = self.pre_tool_use(block.name, block.input)
                                
                                result = self.execute_tool(
                                    block.name, modified_input
                                )
                                
                                # ─── POST-TOOL USE (Modify Output) ──
                                modified_result = self.post_tool_use(block.name, modified_input, result)
                                
                                tool_results.append(
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": block.id,
                                        "content": json.dumps(modified_result)
                                        if not isinstance(modified_result, str)
                                        else modified_result,
                                    }
                                )
                            except Exception as e:
                                # Structured tool error — tell the agent
                                # WHAT failed and HOW to recover
                                tool_results.append(
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": block.id,
                                        "content": json.dumps(
                                            {
                                                "error": str(e),
                                                "error_code": ErrorCode.TOOL_EXECUTION_FAILED,
                                                "suggestion": "Try with different parameters or skip this step",
                                            }
                                        ),
                                        "is_error": True,
                                    }
                                )

                    # Feed tool results back into the conversation
                    messages.append({"role": "user", "content": tool_results})

                elif response.stop_reason == "max_tokens":
                    # Response was truncated — this is a real production issue.
                    # Don't silently continue. Either increase max_tokens or 
                    # tell the agent to be more concise.
                    logger.warning(
                        f"[{self.agent_name}] Response truncated at max_tokens"
                    )
                    messages.append({"role": "assistant", "content": response.content})
                    messages.append(
                        {
                            "role": "user",
                            "content": "Your response was truncated. Please provide a shorter, complete response.",
                        }
                    )

            except anthropic.APIError as e:
                # API-level errors: rate limits, server errors, etc.
                return SubAgentResult(
                    task_id=task.task_id,
                    status=TaskStatus.FAILED,
                    error=AgentError(
                        code=ErrorCode.LLM_API_ERROR,
                        message=f"Claude API error: {str(e)}",
                        details={"status_code": getattr(e, "status_code", None)},
                        recoverable=True,
                        suggested_action="Retry with exponential backoff",
                    ),
                    iterations_used=iterations,
                )

        # ─── Max iterations reached — NEVER silently stop ────────
        return SubAgentResult(
            task_id=task.task_id,
            status=TaskStatus.FAILED,
            error=AgentError(
                code=ErrorCode.MAX_ITERATIONS_EXCEEDED,
                message=f"Agent {self.agent_name} exceeded {self.max_iterations} iterations",
                recoverable=False,
                suggested_action="Review task complexity or increase max_iterations",
            ),
            iterations_used=iterations,
        )

    def _format_task_prompt(self, task: SubAgentTask) -> str:
        """
        Format the task as a structured prompt.
        
        Notice: we include the constraints and input data as JSON,
        not as natural language. This reduces ambiguity.
        """
        return f"""## Task
{task.objective}

## Input Data
```json
{json.dumps(task.input_data, indent=2)}
```

## Constraints
```json
{json.dumps(task.constraints or {}, indent=2)}
```

## Output Format
Respond with a valid JSON object containing your analysis.
Do not include any text outside the JSON object.
"""

    def _extract_text(self, response) -> str:
        """Extract text content from Claude's response."""
        for block in response.content:
            if hasattr(block, "text"):
                return block.text
        return ""

    def _parse_result(self, text: str) -> dict:
        """
        Try to parse the result as JSON.
        
        If parsing fails, wrap in a structured dict anyway.
        We NEVER return unstructured data to the coordinator.
        """
        try:
            # Try to extract JSON from the response
            # Handle cases where the model wraps JSON in markdown code blocks
            cleaned = text.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                # Remove first and last lines (``` markers)
                cleaned = "\n".join(lines[1:-1])
            return json.loads(cleaned)
        except (json.JSONDecodeError, IndexError):
            return {"raw_response": text, "parse_warning": "Response was not valid JSON"}

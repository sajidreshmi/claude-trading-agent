"""
Runner — Execute the trading agent system.

This is your entry point to test the agentic loop end-to-end.
"""

import asyncio
import json
import logging
import uuid
import sys

from agents.coordinator import CoordinatorAgent
from models.schemas import SubAgentTask

# Configure logging to see the agentic loop in action
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


async def run_trading_analysis(symbol: str):
    """
    Run a complete trading analysis pipeline.
    
    Watch the logs — you'll see:
    1. Coordinator receives the task
    2. Coordinator dispatches to Market Analyst (tool_use)
    3. Market Analyst uses its tools (fetch_price, get_news, etc.)
    4. Market Analyst returns signal (end_turn)
    5. Coordinator receives signal, checks risk
    6. Coordinator produces final decision
    
    All driven by stop_reason, not text parsing.
    """
    
    print("=" * 60)
    print(f"  TRADING ANALYSIS: {symbol}")
    print("=" * 60)

    # Initialize the coordinator
    coordinator = CoordinatorAgent()

    # Create the task
    task = SubAgentTask(
        task_id=str(uuid.uuid4()),
        agent_type="coordinator",
        objective=f"Analyze {symbol} and produce a trade recommendation. First dispatch market analysis, then check risk, and produce a final recommendation.",
        input_data={
            "symbol": symbol,
            "account_balance": 50000,
            "current_positions": [],  # No existing positions
        },
        max_iterations=15,
    )

    # Run the coordinator's agentic loop
    result = await coordinator.run(task)

    # Display results
    print("\n" + "=" * 60)
    print("  RESULT")
    print("=" * 60)
    print(f"Status:      {result.status}")
    print(f"Iterations:  {result.iterations_used}")

    if result.result:
        print(f"Decision:\n{json.dumps(result.result, indent=2)}")
    
    if result.error:
        print(f"Error:       {result.error.code}")
        print(f"Message:     {result.error.message}")
        print(f"Recoverable: {result.error.recoverable}")
        print(f"Suggestion:  {result.error.suggested_action}")

    return result


if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    asyncio.run(run_trading_analysis(symbol))

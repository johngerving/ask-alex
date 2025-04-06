from typing import List
from agno.agent import Agent, AgentMemory
from agno.run.response import RunEvent, RunResponse
from agno.memory.agent import AgentRun
from agno.utils.log import logger
from agno.models.google import Gemini
from agno.models.message import Message
from agno.workflow import Workflow


def agent_with_history(agent: Agent, history: List[Message]) -> Agent:
    """
    Create a new agent with the given history.

    Args:
        agent: The agent to use.
        history: The history of messages to use.

    Returns:
        A new agent with the given history.
    """
    agent.memory.add_run(
        AgentRun(
            response=RunResponse(
                messages=history,
            )
        )
    )

    return agent

import json
import logging
import os
from textwrap import dedent
from typing import Optional
from agno.agent import Agent
from agno.models.google import Gemini
from agno.storage.agent.postgres import PostgresAgentStorage


def get_chat_agent(user_id: str) -> Agent:
    logger = logging.getLogger("ray.serve")
    logger.info(f"Connection: {os.getenv('PG_CONN_STR')}")
    storage = PostgresAgentStorage(
        table_name="chat_agent",
        db_url=os.getenv("PG_CONN_STR"),
        schema="public",
        auto_upgrade_schema=True,
    )

    session_id: Optional[str] = None

    existing_sessions = storage.get_all_session_ids(user_id)
    if len(existing_sessions) > 0:
        session_id = existing_sessions[0]

    agent = Agent(
        model=Gemini(id="gemini-2.0-flash"),
        storage=storage,
        session_id=session_id,
        user_id=user_id,
        instructions=dedent(
            """\
            You are ALEX, a helpful AI assistant designed to provide information about Humboldt. Respond to the input as a friendly AI assistant, generating human-like text, and follow the instructions in the input if applicable. Keep the response concise and engaging, using Markdown when appropriate. Use a conversational tone and provide helpful and informative responses, utilizing only the context provided to formulate answers.
        """
        ),
        markdown=True,
        show_tool_calls=True,
        add_history_to_messages=True,
        debug_mode=True,
        monitoring=True,
    )

    return agent

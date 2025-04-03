import json
import logging
import os
from textwrap import dedent
from typing import Optional, Literal, List
from agno.agent import Agent, AgentMemory
from agno.run.response import RunEvent, RunResponse
from agno.utils.log import logger
from agno.models.google import Gemini
from agno.models.message import Message
from agno.storage.agent.postgres import PostgresAgentStorage
from agno.workflow import Workflow
from agno.tools import tool
from pydantic import BaseModel, Field


@tool(
    stop_after_tool_call=True,
)
def retrieval_route():
    """Use this function to initiate the retrieval workflow if you need more information to answer a query."""


class ChatWorkflow(Workflow):
    def run(self, message: str) -> RunResponse:
        storage = PostgresAgentStorage(
            table_name="chat_agent",
            db_url=os.getenv("PG_CONN_STR"),
            schema="public",
            auto_upgrade_schema=True,
        )

        chat_agent = self.get_chat_agent(storage=storage)

        response = chat_agent.run(message)

        logger = logging.getLogger("ray.serve")
        logger.info(response)

        return response

    def get_chat_agent(self, storage: PostgresAgentStorage) -> Agent:

        session_id: Optional[str] = None

        existing_sessions = storage.get_all_session_ids(self.user_id)
        if len(existing_sessions) > 0:
            session_id = existing_sessions[0]

        agent = Agent(
            model=Gemini(id="gemini-2.0-flash"),
            storage=storage,
            session_id=session_id,
            user_id=self.user_id,
            instructions=dedent(
                """\
                You are ALEX, a helpful AI assistant designed to provide information about Humboldt. Respond to the input as a friendly AI assistant, generating human-like text, and follow the instructions in the input if applicable. Keep the response concise and engaging, using Markdown when appropriate. Use a conversational tone and provide helpful and informative responses.
                Use the retrieval_route tool to route the query to a retrieval step if you do not have enough information to answer the question.
                Do not ask permission to use a tool.

                <examples>
                    In response to the message "Who are you?", you should respond normally.

                    In response to the message "What are the symptoms of COVID-19 according to the latest research, you should call the retrieval_route tool.
                </examples>
                """
            ),
            tools=[retrieval_route],
            show_tool_calls=True,
            add_history_to_messages=True,
            debug_mode=True,
            monitoring=True,
        )

        return agent

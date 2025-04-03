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
from pydantic import BaseModel, Field


class Action(BaseModel):
    route: Literal["chat", "retrieval"] = Field(
        ...,
        description="The action to take. Choose chat if the user message does not require additional information to answer. Choose retrieval if you need information that cannot be found within the available context, or if the user specifically requests it.",
    )


class ChatWorkflow(Workflow):
    def run(self, message: str) -> RunResponse:
        storage = PostgresAgentStorage(
            table_name="chat_agent",
            db_url=os.getenv("PG_CONN_STR"),
            schema="public",
            auto_upgrade_schema=True,
        )

        router = self.get_router_agent(storage=storage)

        response = router.run(message)
        if not isinstance(response.content, Action):
            raise Exception(f"Invalid Action response: {response.content}")
        action: Action = response.content

        if action.route == "chat":
            messages = router.memory.get_messages_from_last_n_runs(3)
            chat_agent = self.get_chat_agent(messages=messages)
            response = chat_agent.run(message)
            return response

        return response

    def get_router_agent(self, storage: PostgresAgentStorage) -> Agent:

        session_id: Optional[str] = None

        existing_sessions = storage.get_all_session_ids(self.user_id)
        if len(existing_sessions) > 0:
            session_id = existing_sessions[0]

        agent = Agent(
            model=Gemini(id="gemini-2.0-flash-lite"),
            storage=storage,
            session_id=session_id,
            user_id=self.user_id,
            instructions=dedent(
                """\
                You are a router agent designed to decide which action to take based on the current user message and the chat history.
                """
            ),
            response_model=Action,
            reasoning=True,
            show_tool_calls=True,
            add_history_to_messages=True,
            debug_mode=True,
            monitoring=True,
        )

        return agent

    def get_chat_agent(self, messages: List[Message]) -> Agent:
        agent = Agent(
            model=Gemini(id="gemini-2.0-flash"),
            instructions=dedent(
                """\
                You are ALEX, a helpful AI assistant designed to provide information about Humboldt. Respond to the input as a friendly AI assistant, generating human-like text, and follow the instructions in the input if applicable. Keep the response concise and engaging, using Markdown when appropriate. Use a conversational tone and provide helpful and informative responses.
                """
            ),
            memory=AgentMemory(messages=messages),
            markdown=True,
            show_tool_calls=True,
            add_history_to_messages=True,
            debug_mode=True,
            monitoring=True,
        )

        return agent

from textwrap import dedent
from typing import List
from llama_index.llms.openrouter import OpenRouter
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import FunctionTool
import os

import psycopg

from app.agent.prompts import METADATA_AGENT_PROMPT


async def call_metadata_agent(prompt: str) -> str:
    """Call the metadata agent to get metadata for a given prompt. Use this for determining the proper metadata to use to search for documents.

    Args:
        prompt (str): The prompt to send to the metadata agent. This should be a detailed, standalone prompt requesting a kind of document and should reflect the user's query.

    Returns:
        str: The response from the metadata agent, which should be a JSON string containing the metadata.
    """

    llm = OpenRouter(
        model="mistralai/mistral-small-3.2-24b-instruct",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        context_window=128000,
        max_tokens=4000,
        is_chat_model=True,
        is_function_calling_model=True,
    )

    async with await psycopg.AsyncConnection.connect(os.getenv("PG_CONN_STR")) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT collection FROM collections_distinct ORDER BY collection"
            )
            rows = await cur.fetchall()
            collections: List[str] = [row[0] for row in rows]
            collections_str = "\n".join(collections)

            await cur.execute(
                "SELECT department FROM departments_distinct ORDER BY department"
            )
            rows = await cur.fetchall()
            departments: List[str] = [row[0] for row in rows]
            departments_str = "\n".join(departments)

    response = await llm.achat(
        messages=[
            ChatMessage(
                role="system",
                content=METADATA_AGENT_PROMPT % (collections_str, departments_str),
            ),
            ChatMessage(role="user", content=prompt),
        ]
    )

    return response.message.content


tool = FunctionTool.from_defaults(
    async_fn=call_metadata_agent,
)

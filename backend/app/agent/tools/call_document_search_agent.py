from textwrap import dedent
from typing import List
from llama_index.llms.openrouter import OpenRouter
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context
import os

import psycopg

from app.agent.agents.document_search_agent import get_document_search_agent
from app.agent.tools import search_documents
from app.agent.prompts import DOCUMENT_SEARCH_AGENT_PROMPT


async def call_document_search_agent(ctx: Context, prompt: str) -> str:
    """Call the document search agent to get a list of documents.

    Example user messages to use this tool for:
    - "What documents talk about ...?"
    - "Summarize documents about ..."
    - "Find documents that mention ..."

    Args:
        prompt (str): The prompt to send to the document search agent. This should be a detailed, standalone prompt requesting a kind of document and should reflect the user's query.

    Returns:
        str: The response from the document search agent, which should be a JSON string containing the metadata.
    """

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

    # agent = get_document_search_agent(
    #     collections_str,
    #     departments_str,
    # )

    llm = OpenRouter(
        model="qwen/qwen3-32b",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        context_window=128000,
        max_tokens=4000,
        is_chat_model=True,
        is_function_calling_model=True,
    )

    response = await llm.achat_with_tools(
        tools=[search_documents],
        chat_history=[
            ChatMessage(
                role="system",
                content=DOCUMENT_SEARCH_AGENT_PROMPT
                % (collections_str, departments_str),
            ),
            ChatMessage(role="user", content=prompt),
        ],
    )

    tool_calls = llm.get_tool_calls_from_response(response, error_on_no_tool_call=False)

    if not tool_calls:
        return response.message.content

    for tool_call in tool_calls:
        if tool_call.tool_name == search_documents.metadata.name:
            new_tool_input = {**tool_call.tool_kwargs}
            new_tool_input[search_documents.ctx_param_name] = ctx
            tool_output = await search_documents.acall(**new_tool_input)
            return tool_output

    return "No valid tool call found in response."


tool = FunctionTool.from_defaults(
    async_fn=call_document_search_agent,
)

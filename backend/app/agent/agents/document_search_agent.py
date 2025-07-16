from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openrouter import OpenRouter
import os
from app.agent.prompts import DOCUMENT_SEARCH_AGENT_PROMPT
from app.agent.tools import search_documents


def get_document_search_agent(collections: str, departments: str) -> FunctionAgent:
    """Returns a document search agent that can be used to search for documents based on user queries."""

    llm = OpenRouter(
        model="qwen/qwen3-32b",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        context_window=128000,
        max_tokens=4000,
        temperature=0.7,
        is_chat_model=True,
        is_function_calling_model=True,
    )

    agent = FunctionAgent(
        system_prompt=DOCUMENT_SEARCH_AGENT_PROMPT % (collections, departments),
        tools=[search_documents],
        llm=llm,
    )

    return agent

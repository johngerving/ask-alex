import asyncio
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from reflection_agent import ReflectionAgent
from llama_index.llms.openrouter import OpenRouter
import os
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from langfuse import get_client

# Initialize LlamaIndex instrumentation
LlamaIndexInstrumentor().instrument()

from dotenv import load_dotenv

load_dotenv()


def web_search(query: str) -> str:
    """Search the web for a query and return a list of results."""
    return "No results found."


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


tools = [
    FunctionTool.from_defaults(web_search),
    FunctionTool.from_defaults(add),
]

agent = ReflectionAgent(
    tools=tools,
    llm=OpenRouter(
        model="deepseek/deepseek-chat-v3-0324",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        context_window=41000,
        max_tokens=4000,
        is_chat_model=True,
        is_function_calling_model=True,
    ),
    small_llm=OpenRouter(
        model="mistralai/mistral-small-24b-instruct-2501",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        context_window=41000,
        max_tokens=4000,
        is_chat_model=True,
        is_function_calling_model=True,
    ),
    timeout=60,
)


async def main():
    langfuse = get_client()
    with langfuse.start_as_current_span(name="reflection_agent_trace"):
        ret = await agent.run(input="Write a report on microglia.")

    print(ret)
    langfuse.flush()


asyncio.run(main())

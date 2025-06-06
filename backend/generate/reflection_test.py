import asyncio
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from reflection_agent import ReflectionAgent
from llama_index.llms.openrouter import OpenRouter
import os

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
    ret = await agent.run(
        input="What is the sum of 1924 and 298093? Also, when was the Redwood National Park established?"
    )

    print(ret)


asyncio.run(main())

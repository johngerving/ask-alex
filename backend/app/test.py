import logging
import os
from typing import Annotated

from litellm import api_base
from llama_index.core.llms import ChatMessage
import asyncio
from dotenv import load_dotenv


load_dotenv()

from llama_index.llms.openai_like import OpenAILike
from llama_index.core.agent.workflow import FunctionAgent

from llama_index.llms.mistralai import MistralAI


import llama_index.core

# llm = OpenAILike(
#     model="nvidia/Llama-3_3-Nemotron-Super-49B-v1",
#     api_base="http://0.0.0.0:5000/v1",
#     api_key="ouykUUGoJrhTBgOkpHkAStohxdVBoUeV",
#     context_window=32768,
#     is_chat_model=True,
#     is_function_calling_model=True,
# )

# llm = OpenAILike(
#     model="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
#     api_base="http://0.0.0.0:8000/v1",
#     api_key="fake",
#     is_chat_model=True,
#     is_function_calling_model=True,
# )

llm = MistralAI(
    model="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    endpoint="http://0.0.0.0:8000",
    api_key="fake",
)

# llm = LiteLLM(
#     model="openai/gemma3",
#     api_base="https://llm.nrp-nautilus.io",
#     api_key=os.getenv("LITELLM_API_KEY"),
#     is_chat_model=True,
#     is_function_calling_model=True,
# )


from llama_index.core.tools import FunctionTool


def add(
    a: Annotated[int, "The first number to add."],
    b: Annotated[int, "The second number to add."],
) -> int:
    """Add two numbers."""
    print(f"Adding {a} and {b}")
    return a + b


tool = FunctionTool.from_defaults(fn=add)


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


agent = FunctionAgent(
    llm=llm,
    system_prompt="You are a helpful assistant that can perform calculations.",
    tools=[tool],
)


async def main():
    gen = await llm.astream_chat_with_tools(
        tools=[tool],
        user_msg="What is 593 + 325?",
    )
    async for response in gen:
        if (
            hasattr(response.raw.choices[0].delta, "reasoning_content")
            and response.raw.choices[0].delta.reasoning_content is not None
        ):
            print(
                f"{bcolors.OKCYAN}{response.raw.choices[0].delta.reasoning_content}",
                end="",
                flush=True,
            )
        elif (
            hasattr(response.raw.choices[0].delta, "content")
            and response.raw.choices[0].delta.content is not None
        ):
            print(
                f"{bcolors.OKGREEN}{response.raw.choices[0].delta.content}",
                end="",
                flush=True,
            )
        elif (
            hasattr(response.raw.choices[0].delta, "tool_calls")
            and response.raw.choices[0].delta.tool_calls is not None
        ):
            tool_call = response.raw.choices[0].delta.tool_calls[0].function
            if tool_call.name is not None:
                print(
                    f"{bcolors.OKCYAN}{tool_call.name}",
                    end="",
                    flush=True,
                )
            if tool_call.arguments is not None:
                print(
                    f"{bcolors.OKCYAN}{tool_call.arguments}",
                    end="",
                    flush=True,
                )


asyncio.run(main())

# async def main():
#     gen = await llm.astream_chat(
#         [
#             ChatMessage(role="user", content="How many R's are in strawberry?"),
#         ],
#     )
#     async for response in gen:
#         if (
#             hasattr(response.raw.choices[0].delta, "reasoning_content")
#             and response.raw.choices[0].delta.reasoning_content is not None
#         ):
#             print(
#                 f"{bcolors.OKCYAN}{response.raw.choices[0].delta.reasoning_content}",
#                 end="",
#                 flush=True,
#             )
#         elif hasattr(response.raw.choices[0].delta, "content"):
#             print(
#                 f"{bcolors.OKGREEN}{response.raw.choices[0].delta.content}",
#                 end="",
#                 flush=True,
#             )

#     print()


# asyncio.run(main())

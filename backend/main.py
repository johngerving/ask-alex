import os
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.base.llms.types import ChatMessage

llm = OpenAILike(model="allenai/OLMo-2-1124-13B-Instruct", api_base="http://localhost:8000/v1", api_key="fake", is_chat_model=True)

message = input("> ")

while message != "exit":
    curr_resp = ""
    # resp = llm.complete(f"<|endoftext|><|system|>You are ALEX, a helpful AI assistant designed to provide information about Humboldt-related documents.<|endoftext|><|user|>\n{message}\n<|assistant|>\n")
    resp = llm.stream_complete(f"<|endoftext|><|system|>You are ALEX, a helpful AI assistant designed to provide information about Humboldt-related documents.<|endoftext|><|user|>\n{message}\n<|assistant|>\n")
    for c in resp:
        print(str(c)[len(curr_resp):], end="")
        curr_resp = str(c)

    print()
    message = input("> ")
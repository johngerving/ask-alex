from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.base.response.schema import Response
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from llama_index.core import PromptTemplate

qa_prompt = PromptTemplate(
    "Below is a list of retrieved chunks.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query. Your answer should be detailed and comprehensive.\n"
    "Always provide inline citations for any information you use to formulate your answer, citing the id field of the chunk or document you used. DO NOT hallucinate a chunk id.\n"
    '- Example 1: If you are citing a document with the id "asdfgh", you should write something like, "Apples fall to the ground in autum [asdfgh].\n"'
    '- Example 2: If you are citing two documents with the ids "asdfgh" and "qwerty", you should write something like, "The sun rises in the east and sets in the west [asdfgh][qwerty]."\n'
    "Query: {query_str}\n"
)


class ChunkQueryEngine(CustomQueryEngine):
    """Query engine that retrieves chunks from a retriever and synthesizes a response using an LLM, providing inline citations for the chunks used in the response."""

    retriever: BaseRetriever
    llm: OpenAI

    def custom_query(self, query_str: str) -> Response:
        raise NotImplementedError()

    async def acustom_query(self, query_str: str):
        nodes = await self.retriever.aretrieve(query_str)

        context_str = ""
        for node in nodes:
            context_str += f"<chunk id={node.node_id}>\n"
            context_str += node.node.get_content() + "\n"
            context_str += "</chunk>\n"

        response = await self.llm.achat(
            [
                ChatMessage(
                    role="system",
                    content=qa_prompt.format(
                        context_str=context_str, query_str=query_str
                    )
                    + " /no_think",
                )
            ]
        )

        return Response(
            response=response.message.content,
            source_nodes=nodes,
        )

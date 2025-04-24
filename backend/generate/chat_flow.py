import functools
import json
import os
from tabnanny import verbose
from textwrap import dedent
from typing import Annotated, Dict, Iterator, Optional, Literal, List
from urllib.parse import urlparse
import haystack
from pydantic import BaseModel, Field
from logging import Logger

from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Context,
    Workflow,
    step,
)
from llama_index.core.agent.workflow import (
    AgentInput,
    AgentOutput,
    ToolCall,
    ToolCallResult,
    AgentStream,
)
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.agent.workflow import ReActAgent, FunctionAgent
from llama_index.core.tools import FunctionTool, RetrieverTool
from llama_index.core.llms import ChatMessage
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.postprocessor.colbert_rerank import ColbertRerank
from llama_index.core.agent.workflow import AgentStream
from llama_index.core.schema import TextNode
from llama_index.core import set_global_handler


from utils import validate_brackets


class WorkflowStartEvent(StartEvent):
    """Event to start the workflow."""

    message: ChatMessage
    history: List[ChatMessage]


class ChatRouteEvent(Event):
    pass


class RetrievalRouteEvent(Event):
    pass


class ChatOrRetrieval(BaseModel):
    """Data model for routing between chat and retrieval."""

    reasoning: str = Field(
        ..., description="The reasoning behind the routing decision."
    )
    route: Literal["chat", "retrieval"] = Field(
        ..., description="The route to take: either 'chat' or 'retrieval'."
    )


class WorkflowResponse(BaseModel):
    delta: str
    response: str


class ChatFlow(Workflow):
    """The main workflow for Ask Alex."""

    def __init__(self, logger: Logger, **kwargs):
        super().__init__(**kwargs)
        self.logger = logger

    llm = GoogleGenAI(model="gemini-2.0-flash", temperature=0)

    small_llm = GoogleGenAI(model="gemini-2.0-flash-lite", temperature=0)

    pg_conn_str = os.getenv("PG_CONN_STR")
    if not pg_conn_str:
        raise ValueError("PG_CONN_STR environment variable not set")

    # Get Postgres credentials from connection string
    pg_url = urlparse(pg_conn_str)
    host = pg_url.hostname
    port = pg_url.port
    database = pg_url.path[1:]
    user = pg_url.username
    password = pg_url.password

    # Vector store to store chunks + embeddings in

    vector_store = PGVectorStore.from_params(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password,
        table_name="llamaindex_docs",
        schema_name="public",
        hybrid_search=True,
        embed_dim=768,
        hnsw_kwargs={
            "hnsw_m": 16,
            "hnsw_ef_construction": 64,
            "hnsw_ef_search": 40,
            "hnsw_dist_method": "vector_cosine_ops",
        },
    )

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=HuggingFaceEmbedding(
            model_name="sentence-transformers/all-mpnet-base-v2"
        ),
    )

    vector_retriever = index.as_retriever(
        vector_store_query_mode="default",
        similarity_top_k=50,
        verbose=True,
    )

    text_retriever = index.as_retriever(
        vector_store_query_mode="sparse", similarity_top_k=50
    )

    retriever = QueryFusionRetriever(
        [vector_retriever, text_retriever],
        similarity_top_k=10,
        llm=small_llm,
        num_queries=1,
        mode="relative_score",
        use_async=False,
    )

    # reranker = ColbertRerank(
    #     top_n=10,
    #     model="colbert-ir/colbertv2.0",
    #     tokenizer="colbert-ir/colbertv2.0",
    #     keep_retrieval_score=True,
    # )

    @step
    async def route_chat_or_retrieval(
        self, ctx: Context, ev: StartEvent
    ) -> ChatRouteEvent | RetrievalRouteEvent:
        """Route the message to either chat or retrieval based on the message and history."""
        self.logger.info("Routing chat or retrieval")

        sllm = self.small_llm.as_structured_llm(output_cls=ChatOrRetrieval)

        # Set global context variables for use in the entire workflow
        await ctx.set("message", ev.message)
        await ctx.set("history", ev.history)

        sources: List[TextNode] = [TextNode(text="test")]
        await ctx.set("sources", sources)

        # Call the LLM to determine the route
        json_obj: Dict[str, str] = sllm.chat(
            messages=[
                ChatMessage(
                    role="system",
                    content=dedent(
                        """\
                        You are a router agent tasked with deciding whether to route a user message to a chat agent or a retrieval agent.
                        You will receive a list of previous messages and the current user message.
                        Use the following steps:
                        1. Output a thought in which you reason through whether to route the message to the chat agent or the retrieval agent.
                        2. Output the route you have chosen: either "chat" or "retrieval".
                        """
                    ),
                ),
                ev.message,
            ]
        ).raw.dict()

        # Parse the JSON response from the LLM as a ChatOrRetrieval object
        response = ChatOrRetrieval(**json_obj)

        # Route query based on the LLM's response
        if response.route == "chat":
            return ChatRouteEvent()
        else:
            return RetrievalRouteEvent()

    @step
    async def chat(self, ctx: Context, ev: ChatRouteEvent) -> StopEvent:
        """Handle the chat route by generating a response using the LLM."""
        self.logger.info("Running step chat")

        message = await ctx.get("message")
        history = await ctx.get("history")

        agent = FunctionAgent(
            llm=self.llm,
            system_prompt=dedent(
                """\
                You are ALEX, a helpful AI assistant designed to provide information about Humboldt. Respond to the input as a friendly AI assistant, generating human-like text, and follow the instructions in the input if applicable. Keep the response concise and engaging, using Markdown when appropriate. Use a conversational tone and provide helpful and informative responses. Do not provide information beyond what you are given in the context.
                """
            ),
        )

        handler = agent.run(message, chat_history=history)

        response = ""

        async for ev in handler.stream_events():
            if isinstance(ev, AgentStream) and ev.response:
                ctx.write_event_to_stream(
                    WorkflowResponse(delta=ev.delta, response=ev.response)
                )

        response = handler

        return StopEvent(result=response)

    @step
    async def retrieve(self, ctx: Context, ev: RetrievalRouteEvent) -> StopEvent:
        """Handle the retrieval route by searching the knowledge base for information."""
        self.logger.info("Running retrieval step")

        message: ChatMessage = await ctx.get("message")
        history: List[ChatMessage] = await ctx.get("history")

        tools = [
            FunctionTool.from_defaults(
                fn=self._think,
                name="think",
            ),
            FunctionTool.from_defaults(
                async_fn=functools.partial(self._search_knowledge_base, ctx),
                name="search_knowledge_base",
            ),
        ]

        agent = FunctionAgent(
            llm=self.llm,
            system_prompt=dedent(
                """\
                You are ALEX, a helpful AI assistant designed to provide information about Humboldt. Respond to the input as a friendly AI assistant, generating human-like text, and follow the instructions in the input if applicable. Make your responses comprehensive, informative, and thorough, using Markdown when appropriate. Use a conversational tone and provide helpful and informative responses.

                Formulate an answer to user queries.

                Follow these steps:
                1. Think step-by-step about the query with the `think(thought)` tool, expanding with more context if necessary.
                2. ALWAYS use the `search_knowledge_base(query)` tool to get relevant documents.
                3. Search the knowledge base as many times as you need to obtain relevant documents.
                4. Use the retrieved documents to write a comprehensive answer to the query, discarding irrelevant documents. Provide inline citations of each document you use.

                ## Using the think tool

                Before taking any action or responding to the user after receiving tool results, use the think tool as a scratchpad to:
                - Create a plan of action
                - List the specific rules that apply to the current request
                - Check if all required information is collected
                - Verify that the planned action complies with all policies
                - Iterate over tool results for correctness

                Here is an example of what to iterate over inside the think tool:
                <think_tool_example>
                I need to search the knowledge base for information about redwood trees.
                I should verify that the context is sufficient to answer a question about redwood trees comprehensively once I'm done.
                - Plan: collect missing info, verify relevancy, formulate answer
                </think_tool_example>

                Finally, here are a set of rules that you MUST follow:
                <rules>
                - Use the `think(thought)` tool to reason about the current request and develop a plan of action.
                - Use the `search_knowledge_base(query)` tool to retrieve documents from your knowledge base before answering the query.
                - Do not use phrases like "based on the information provided" or "from the knowledge base".
                - Always provide inline citations for any information you use to formulate your answer. These should be in the format [doc_id], where doc_id is the "doc_id" field of the document you are citing. Multiple citations should be written as [id_1][id_2].
                    - Example: If you are citing a document with the doc_id "12345", you should write something like, "Apples fall to the ground in autum [12345]."
                </rules>
                """
            ),
            tools=tools,
        )

        handler = agent.run(message, chat_history=history)

        curr_response = ""

        prev_formatted_response = ""
        curr_formatted_response = ""

        response = str(await handler)

        batch_len = 32
        for start in range(0, len(response), batch_len):
            end = min(start + batch_len, len(response))

            curr_response = response[:end]

            if validate_brackets(curr_response):
                curr_formatted_response = await self._generate_citations(
                    ctx, curr_response
                )
                delta = curr_formatted_response[len(prev_formatted_response) :]
                prev_formatted_response = curr_formatted_response

                ctx.write_event_to_stream(
                    WorkflowResponse(delta=delta, response=curr_formatted_response)
                )

        # async for ev in handler.stream_events():
        #     if isinstance(ev, AgentStream) and ev.response:
        #         print(ev.delta, end="", flush=True)
        #         curr_response = ev.response.strip()

        #         if validate_brackets(curr_response):
        #             curr_formatted_response = await self._generate_citations(
        #                 ctx, curr_response
        #             )
        #             delta = curr_formatted_response[len(prev_formatted_response) :]
        #             prev_formatted_response = curr_formatted_response

        #             ctx.write_event_to_stream(
        #                 WorkflowResponse(delta=delta, response=curr_formatted_response)
        #             )

        curr_formatted_response = await self._generate_citations(ctx, curr_response)
        delta = curr_formatted_response[len(prev_formatted_response) :]

        if len(delta) > 0:
            ctx.write_event_to_stream(
                WorkflowResponse(delta=delta, response=curr_formatted_response)
            )

        final_response = await self._generate_citations(ctx, response)

        self.logger.info(f"Final response: {final_response}")

        return StopEvent(result=final_response)

    async def _search_knowledge_base(
        self,
        ctx: Context,
        query: Annotated[str, "The query to search the knowledge base for"],
    ) -> str:
        """Search the knowledge base for relevant documents."""
        self.logger.info(f"Running search_knowledge_base with query: {query}")
        # Use the retriever to get relevant nodes
        try:
            nodes = self.retriever.retrieve(query)
            self.logger.info(f"Retrieved {len(nodes)} nodes")
            # nodes = self.reranker.postprocess_nodes(nodes, query_str=query)
            # self.logger.info(f"Postprocessed {len(nodes)} nodes")

            sources: List[TextNode] = await ctx.get("sources")
            sources = sources + nodes
            await ctx.set("sources", sources)

            json_obj = [
                {"doc_id": node.node_id[:8], "content": node.text} for node in nodes
            ]

        except Exception as e:
            print(e)
            raise

        return json.dumps(json_obj, indent=2)

    def _think(self, thought: Annotated[str, "A thought to think about."]):
        """Use the tool to think about something. It will not obtain new information or change the database, but just append the thought to the log. Use it when complex reasoning or some cache memory is needed."""
        self.logger.info(f"Thought: {thought}")

    async def _generate_citations(self, ctx: Context, text: str) -> str:
        """Generate citations for a text response based on the sources used and replace them in the text.

        Args:
            ctx (Context): The context object containing the sources.
            text (str): The text response to generate citations for.

        Returns:
            str: The text response with citations generated.
        """
        sources: List[TextNode] = await ctx.get("sources")

        sources_used: List[TextNode] = []
        for source in sources:
            if source.node_id[:8] in text and not any(
                [s.node_id == source.node_id for s in sources_used]
            ):
                sources_used.append(source)

        for i, source in enumerate(sources_used):
            download_link = source.metadata.get("download_link")
            node_id = source.node_id[:8]
            if download_link is None:
                text = text.replace(f"[{node_id}]", "")
            else:
                text = text.replace(
                    f"[{node_id}]", f"[[{i+1}]]({source.metadata.get('download_link')})"
                )

        return text

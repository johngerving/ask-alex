import os
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from haystack.components.builders import ChatPromptBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
from haystack_integrations.components.retrievers.pgvector import PgvectorEmbeddingRetriever
from haystack import Pipeline

from ray import serve
from starlette.requests import Request

# from dotenv import load_dotenv
# load_dotenv()

@serve.deployment
class HaystackQA:
    def __init__(self):
        # Create a pipeline so that it can be reused every time __call__ is invoked
        generator = OpenAIChatGenerator(
            api_key=Secret.from_token("VLLM-PLACEHOLDER-API-KEY"), # api_key is needed for compatibility for OpenAI API
            model="allenai/OLMo-2-1124-13B-Instruct",
            api_base_url="http://localhost:8000/v1",
            generation_kwargs={"max_tokens": 512}
        )

        self.messages = [
            ChatMessage.from_system("You are ALEX, a helpful AI assistant designed to provide information about Humboldt. Respond to the input as a friendly AI assistant, generating human-like text, and follow the instructions in the input if applicable. Keep the response concise and engaging, using Markdown when appropriate. Use a conversational tone and provide helpful and informative responses, utilizing only the context provided to formulate answers.")
        ]

        # Get the document store in Postgres
        document_store = PgvectorDocumentStore(
            connection_string=Secret.from_token(os.getenv("PG_CONN_STR")),
            table_name="haystack_docs",
            embedding_dimension=768,
            vector_function="cosine_similarity",
            recreate_table=False,
            search_strategy="hnsw"
        )

        # Retriever to get embedding vectors based on query
        retriever = PgvectorEmbeddingRetriever(document_store=document_store, top_k=8)

        rag_template = [
            ChatMessage.from_system(
                """
                Answer the questions based on the given context.

                Context:
                {% for document in documents %}
                    {{ document.content }}
                {% endfor %}
                Question: {{ question }}
                Answer:
                """
            )
        ]

        rag_prompt_builder = ChatPromptBuilder(template=rag_template)

        self.rag_pipeline = Pipeline()
        self.rag_pipeline.add_component("embedder", SentenceTransformersTextEmbedder()) # Embed the user's query to compare to vector DB
        self.rag_pipeline.add_component("retriever", retriever) # Retrieve similar embeddings to get relevant documents
        self.rag_pipeline.add_component("prompt_builder", prompt_builder)
        self.rag_pipeline.add_component("llm", generator)

        self.rag_pipeline.connect("embedder.embedding", "retriever.query_embedding")
        self.rag_pipeline.connect("retriever", "prompt_builder.documents")
        self.rag_pipeline.connect("prompt_builder.prompt", "llm.messages")


        self.chat_pipeline = Pipeline()
        self.chat_pipeline.add_component("llm", generator) # Pass prompt to LLM

    async def __call__(self, request: Request) -> str:
        query: str = str(await request.body())

        # Run the pipeline with the user's query
        # res = self.chat_pipeline.run({"embedder": {"text": query}, "prompt_builder": {"query": query}}, include_outputs_from={"retriever"})
        res = self.chat_pipeline.run({"llm": {"messages": self.messages + [ChatMessage.from_user(query)]}})
        replies = res["llm"]["replies"]

        # replies = ""
        if replies:
            return replies[0].text

        return ""

haystack_deployment = HaystackQA.bind()
# query = "What are the impacts of ammonium phosphate-based fire retardants on cyanobacteria growth?"

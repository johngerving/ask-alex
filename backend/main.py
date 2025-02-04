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

        # Prompt template used for every chat - insert documents from query pipeline
        prompt_builder = ChatPromptBuilder(
            template = [
                ChatMessage.from_user(
                    """
                    You are ALEX, a helpful AI assistant designed to provide information about Humboldt-related documents.

                    Create a concise and informative answer (no more than 50 words) for a given query based solely on the given documents.
                    You must only use information from the given documents. Use an unbiased and journalistic tone. Do not repeat text. If the question is not related to any documents, respond to the question without referring to the documents.

                    Given the following information, answer the query.

                    Context:
                    {% for document in documents %}
                        {{ document.content }} 
                    {% endfor %}

                    Query: {{query}}
                    Answer:
                    """
                )
            ]
        )

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

        self.chat_pipeline = Pipeline()
        self.chat_pipeline.add_component("embedder", SentenceTransformersTextEmbedder()) # Embed the user's query to compare to vector DB
        self.chat_pipeline.add_component("retriever", retriever) # Retrieve similar embeddings to get relevant documents
        self.chat_pipeline.add_component("prompt_builder", prompt_builder) # Build the prompts using the user query and the documents retrieved
        self.chat_pipeline.add_component("llm", generator) # Pass prompt to LLM

        self.chat_pipeline.connect("embedder.embedding", "retriever.query_embedding")
        self.chat_pipeline.connect("retriever", "prompt_builder.documents")
        self.chat_pipeline.connect("prompt_builder.prompt", "llm.messages")

    async def __call__(self, request: Request) -> str:
        query: str = str(await request.body())

        # Run the pipeline with the user's query
        res = self.chat_pipeline.run({"embedder": {"text": query}, "prompt_builder": {"query": query}}, include_outputs_from={"retriever"})
        replies = res["llm"]["replies"]
        # replies = ""
        if replies:
            return replies[0].text

        return ""

haystack_deployment = HaystackQA.bind()
# query = "What are the impacts of ammonium phosphate-based fire retardants on cyanobacteria growth?"

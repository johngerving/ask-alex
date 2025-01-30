import os
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from haystack.components.builders import ChatPromptBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
from haystack_integrations.components.retrievers.pgvector import PgvectorEmbeddingRetriever
from haystack import Pipeline

from dotenv import load_dotenv
load_dotenv()

generator = OpenAIChatGenerator(
    api_key=Secret.from_token("VLLM-PLACEHOLDER-API-KEY"), # api_key is needed for compatibility for OpenAI API
    model="allenai/OLMo-2-1124-13B-Instruct",
    api_base_url="http://localhost:8000/v1",
    generation_kwargs={"max_tokens": 512}
)

prompt_builder = ChatPromptBuilder(
    template = [
        ChatMessage.from_system(
            """You are ALEX, a helpful AI assistant designed to provide information about Humboldt-related documents."""
        ),
        ChatMessage.from_user(
            """
Create a concise and informative answer (no more than 50 words) for a given query based solely on the given documents.
You must only use information from the given documents. Use an unbiased and journalistic tone. Do not repeat text.

Given the following information, answer the query.

Context:
{% for document in documents %}
   {{ document.context }} 
{% endfor %}

Query: {{query}}
Answer:
"""
        )
    ]
)

document_store = PgvectorDocumentStore(
    connection_string=Secret.from_token(os.getenv("PG_CONN_STR")),
    table_name="haystack_docs",
    embedding_dimension=768,
    vector_function="cosine_similarity",
    recreate_table=False,
    search_strategy="hnsw"
)

retriever = PgvectorEmbeddingRetriever(document_store=document_store)

chat_pipeline = Pipeline()
chat_pipeline.add_component("embedder", SentenceTransformersTextEmbedder())
chat_pipeline.add_component("retriever", retriever)
chat_pipeline.add_component("prompt_builder", prompt_builder)
chat_pipeline.add_component("llm", generator)

chat_pipeline.connect("embedder.embedding", "retriever.query_embedding")
chat_pipeline.connect("retriever", "prompt_builder")
chat_pipeline.connect("prompt_builder.prompt", "llm.messages")

query = "What are the impacts of ammonium phosphate-based fire retardants on cyanobacteria growth?"

response = chat_pipeline.run({"embedder": {"text": query}, "prompt_builder": {"query": query}})
print(response)
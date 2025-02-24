import os

from haystack.utils import Secret
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.components.builders import ChatPromptBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.routers import ConditionalRouter
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
from haystack_integrations.components.retrievers.pgvector import PgvectorEmbeddingRetriever
from haystack.tools import Tool
from haystack import Pipeline
from rag_classifier import RAGClassifier

from typing import Dict

class RagPipeline:
    def __init__(self):
        pass
        # Initialize LLM generators for different parts of the pipeline - they will all call the same API endpoint
        rag_classifier_llm = self._llm_component()
        rag_llm = self._llm_component()
        chat_llm = self._llm_component()

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

        # Template for classifying query as chat or rag
        chat_rag_template = [
            ChatMessage.from_user(
                '''
                You are ALEX, a helpful AI assistant designed to provide information about Humboldt. Given a message, output 'chat' if the message is about you. Otherwise, output 'rag'.

                Examples:
                    In response to 'Who are you?', you should output 'chat'
                    In response to 'What is the population of Humboldt County?', you should output 'rag'

                Output 'chat' or 'rag' from the information present in this passage: {{passage}}.
                Only use information that is present in the passage.
                {% if invalid_replies and error_message %}
                You already created the following output in a previous attempt: {{invalid_replies}}
                However, this doesn't comply with the format requirements from above and triggered this Python exception: {{error_message}}
                Correct the output and try again. Just return the corrected output without any extra explanations.
                {% endif %}  
                '''
            )
        ]
        prompt_builder_for_rag_classifier = ChatPromptBuilder(template=chat_rag_template)
        rag_classifier = RAGClassifier()

        # Based on classifier response, follow different route in pipeline
        rag_classifier_routes = [
            {
                "condition": "{{'chat' in valid_reply}}",
                "output": "{{query}}",
                "output_name": "go_to_chat",
                "output_type": str,
            },
            {
                "condition": "{{'rag' in valid_reply}}",
                "output": "{{query}}",
                "output_name": "go_to_rag",
                "output_type": str,
            }
        ]

        rag_classifier_router = ConditionalRouter(rag_classifier_routes)

        # If LLM chooses rag, make a template for utilizing the retrieved documents in the LLM context
        rag_template = [
            ChatMessage.from_system(
                """
                You are ALEX, a helpful AI assistant designed to provide information about Humboldt. Respond to the input as a friendly AI assistant, generating human-like text, and follow the instructions in the input if applicable. Keep the response concise and engaging, using Markdown when appropriate. Use a conversational tone and provide helpful and informative responses, utilizing only the context provided to formulate answers.
                """
            ),
            ChatMessage.from_user(
                """
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

        rag_prompt_builder = ChatPromptBuilder(template=rag_template)

        # If the LLM does not choose rag, just respond as a chatbot with no documents
        chat_template = [
            ChatMessage.from_system(
                """
                You are ALEX, a helpful AI assistant designed to provide information about Humboldt. Respond to the input as a friendly AI assistant, generating human-like text, and follow the instructions in the input if applicable. Keep the response concise and engaging, using Markdown when appropriate. Use a conversational tone and provide helpful and informative responses. You do not have access to outside information. If a user's query requires it, ask them to rephrase their question.
                """
            ),
            ChatMessage.from_user(
                """
                {{query}}
                """
            )
        ]
        chat_prompt_builder = ChatPromptBuilder(template=chat_template)

        self.pipeline = Pipeline(max_runs_per_component=5)
        self.pipeline.add_component("prompt_builder_for_rag_classifier", prompt_builder_for_rag_classifier) # Decide whether prompt is chat-based or RAG-based
        self.pipeline.add_component("rag_classifier_llm", rag_classifier_llm) # Pass prompt to LLM
        self.pipeline.add_component("rag_classifier", rag_classifier) 
        self.pipeline.add_component("rag_classifier_router", rag_classifier_router) # Route query based on chat or RAG

        ##### RAG #####
        self.pipeline.add_component("embedder", SentenceTransformersTextEmbedder()) # Get query vector embedding
        self.pipeline.add_component("retriever", retriever) # Retrieve similar documents
        self.pipeline.add_component("rag_prompt_builder", rag_prompt_builder) # Build prompt using query and retrieved documents
        self.pipeline.add_component("rag_llm", rag_llm) # Pass prompt to LLM

        ##### CHAT #####
        self.pipeline.add_component("chat_prompt_builder", chat_prompt_builder) # Build chat prompt from query
        self.pipeline.add_component("chat_llm", chat_llm) # Pass prompt to LLM

        self.pipeline.connect("prompt_builder_for_rag_classifier.prompt", "rag_classifier_llm.messages")
        self.pipeline.connect("rag_classifier_llm.replies", "rag_classifier")
        self.pipeline.connect("rag_classifier.invalid_replies", "prompt_builder_for_rag_classifier.invalid_replies")
        self.pipeline.connect("rag_classifier.error_message", "prompt_builder_for_rag_classifier.error_message")
        self.pipeline.connect("rag_classifier.valid_reply", "rag_classifier_router.valid_reply")

        self.pipeline.connect("rag_classifier_router.go_to_chat", "chat_prompt_builder.query")
        self.pipeline.connect("rag_classifier_router.go_to_rag", "embedder.text")
        self.pipeline.connect("rag_classifier_router.go_to_rag", "rag_prompt_builder.query")
        self.pipeline.connect("embedder.embedding", "retriever.query_embedding")
        self.pipeline.connect("retriever", "rag_prompt_builder.documents")
        self.pipeline.connect("rag_prompt_builder.prompt", "rag_llm.messages")
        self.pipeline.connect("chat_prompt_builder.prompt", "chat_llm.messages")

    def run(self, query: str):
        res = self.pipeline.run({"prompt_builder_for_rag_classifier": {"passage": query}, "rag_classifier_router": {"query": query}})
        return res

    def _llm_component(
        self,
        api_key: Secret = Secret.from_token("PLACEHOLDER_KEY"),
        model: str = "allenai/OLMo-2-1124-13B-Instruct",
        api_base_url: str = "http://localhost:8000/v1",
        generation_kwargs: Dict = {"max_tokens": 512}
    ) -> OpenAIChatGenerator:
        '''
        Returns an OpenAIChatGenerator object with an endpoint to our self-hosted LLM.

        Args:
            api_key: A Secret containing an OpenAI API key.
            model: The HuggingFace name of the model.
            api_base_url: The base URL of the LLM endpoint.
            generation_kwargs: Additional keyword arguments to pass for LLM generation.
        '''
        llm = OpenAIChatGenerator(
            api_key=api_key, # Placeholder api_key is needed for compatibility for OpenAI API
            model=model,
            api_base_url=api_base_url,
            generation_kwargs=generation_kwargs
        )
        return llm
    



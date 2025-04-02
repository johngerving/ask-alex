import os

from haystack.utils import Secret
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack_integrations.components.generators.google_ai import GoogleAIGeminiChatGenerator
from haystack.dataclasses import ChatMessage, ChatRole
from haystack.components.builders import ChatPromptBuilder, AnswerBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.routers import ConditionalRouter
from haystack.components.converters import OutputAdapter
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
from haystack_integrations.components.retrievers.pgvector import (
    PgvectorEmbeddingRetriever,
)
from haystack import Pipeline
from intent_classifier import IntentClassifier

from typing import Dict


class RagPipeline:
    def __init__(self):
        # Initialize LLM generators for different parts of the pipeline - they will all call the same API endpoint
        intent_classifier_llm = self._llm_component()
        rag_llm = self._llm_component()
        chat_llm = self._llm_component()

        # Get the document store in Postgres
        document_store = PgvectorDocumentStore(
            connection_string=Secret.from_token(os.getenv("PG_CONN_STR")),
            table_name="haystack_docs",
            embedding_dimension=768,
            vector_function="cosine_similarity",
            recreate_table=False,
            search_strategy="hnsw",
        )

        # Retriever to get embedding vectors based on query
        retriever = PgvectorEmbeddingRetriever(document_store=document_store, top_k=8)

        classes = {
            "(a)": "use for conversational messages that don't require outside information or previous context to answer.",
            "(b)": "use for messages that require retrieving information to answer the question.",
        }
        available_classes_string = ""
        for c in classes:
            available_classes_string += " ".join([c, classes[c]])

        # Template for classifying query intent
        intent_template = [
            ChatMessage.from_user(
                """
                You are ALEX, a helpful AI assistant designed to provide information about Humboldt. Given a message, classify the intent of the user.

                Available classes:
                """
                + available_classes_string
                + """

                Examples:
                    U: Who are you?
                    Intention: (a)

                    U: How's it going?
                    Intention: (a)

                    U: What is the population of Humboldt County?
                    Intention: (b)

                    U: What are the symptoms of COVID-19 according to the latest research?
                    Intention: (b)

                    U: Can you tell me a joke?
                    Intention: (a)

                    U: What are the top 5 best-selling books of 2024?
                    Intention: (b)

                    U: Tell me more.
                    Intention: (b)


                {% if invalid_replies and error_message %}
                You already created the following output in a previous attempt: {{invalid_replies}}
                However, this doesn't comply with the format requirements from above and triggered this Python exception: {{error_message}}
                Correct the output and try again. Just return the corrected output without any extra explanations.
                {% endif %} 

                U: {{ passage }}
                Intention: 
                """
            )
        ]
        prompt_builder_for_intent_classifier = ChatPromptBuilder(
            template=intent_template
        )

        intent_classifier = IntentClassifier(classes)

        # Based on classifier response, follow different route in pipeline
        intent_classifier_routes = [
            {
                "condition": "{{'(a)' in valid_reply}}",
                "output": "{{ memories[-1] }}",
                "output_name": "go_to_chat",
                "output_type": str,
            },
            {
                "condition": "{{'(b)' in valid_reply}}",
                "output": "{{ memories }}",
                "output_name": "go_to_retrieval",
                "output_type": list[str],
            },
        ]

        intent_classifier_router = ConditionalRouter(intent_classifier_routes)

        rag_contextualizer_prompt_builder = ChatPromptBuilder(
            [
                ChatMessage.from_user(
                    """
                    Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
                    
                    Conversation history:
                    {% for memory in memories %}
                        {{ memory }}
                    {% endfor %}

                    Rewritten Query:
                    """
                ),
            ]
        )

        # If LLM chooses rag, make a template for utilizing the retrieved documents in the LLM context
        rag_template = [
            ChatMessage.from_system(
                """
                You are ALEX, a helpful AI assistant designed to provide information about Humboldt. Respond to the input as a friendly AI assistant, generating human-like text, and follow the instructions in the input if applicable. Keep the response concise and engaging, using Markdown when appropriate. Use a conversational tone and provide helpful and informative responses, utilizing only the context provided to formulate answers.
                """
            ),
            ChatMessage.from_user(
                """
                Please provide an answer based solely on the provided sources. 
                When referencing information from a source, you must create an inline citation using the corresponding source number. Your citation is presented as [i], where i corresponds to the number of the provided source.
                Every answer should include at least one source citation. Only cite a source when you are explicitly referencing it. If none of the sources are helpful, you should indicate that.
                If you cite multiple sources in one sentence, write the citation as [i][j], where i is the number of one source and j is the number of another source.
                
                For example:
                    [1]:
                    The sky is red in the evening and blue in the morning.
                    
                    [2]:
                    Water is wet when the sky is red.

                    Query: When is water wet?
                    Answer: Water will be wet when the sky is red [2], which occurs in the evening [1].
                End of example.

                Now it's your turn. Below are several numbered sources of information.
                {% for document in documents %}
                    [{{ loop.index }}]:
                    {{ document.content }} 
                {% endfor %}

                Query: {{query}}
                Answer:
                """
            ),
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
            ),
        ]
        chat_prompt_builder = ChatPromptBuilder(template=chat_template)

        self.pipeline = Pipeline(max_runs_per_component=5)
        self.pipeline.add_component(
            "prompt_builder_for_intent_classifier", prompt_builder_for_intent_classifier
        )  # Decide whether prompt is chat-based or RAG-based
        self.pipeline.add_component(
            "intent_classifier_llm", intent_classifier_llm
        )  # Pass prompt to LLM
        self.pipeline.add_component("intent_classifier", intent_classifier)
        self.pipeline.add_component(
            "intent_classifier_router", intent_classifier_router
        )  # Route query based on chat or RAG

        ##### RAG #####
        self.pipeline.add_component(
            "rag_contextualizer_prompt_builder", rag_contextualizer_prompt_builder
        )
        self.pipeline.add_component("rag_contextualizer", self._llm_component())
        self.pipeline.add_component(
            "list_to_str_adapter",
            OutputAdapter(template="{{ replies[0].text }}", output_type=str),
        )
        self.pipeline.add_component(
            "embedder", SentenceTransformersTextEmbedder()
        )  # Get query vector embedding
        self.pipeline.add_component(
            "retriever", retriever
        )  # Retrieve similar documents
        self.pipeline.add_component(
            "rag_prompt_builder", rag_prompt_builder
        )  # Build prompt using query and retrieved documents
        self.pipeline.add_component("rag_llm", rag_llm)  # Pass prompt to LLM
        self.pipeline.add_component(
            "rag_answer_builder",
            AnswerBuilder(reference_pattern="\\[(?:(\\d+),?\\s*)+\\]"),
        )

        ##### CHAT #####
        self.pipeline.add_component(
            "chat_prompt_builder", chat_prompt_builder
        )  # Build chat prompt from query
        self.pipeline.add_component("chat_llm", chat_llm)  # Pass prompt to LLM

        self.pipeline.connect(
            "prompt_builder_for_intent_classifier.prompt",
            "intent_classifier_llm.messages",
        )
        self.pipeline.connect("intent_classifier_llm.replies", "intent_classifier")
        self.pipeline.connect(
            "intent_classifier.invalid_replies",
            "prompt_builder_for_intent_classifier.invalid_replies",
        )
        self.pipeline.connect(
            "intent_classifier.error_message",
            "prompt_builder_for_intent_classifier.error_message",
        )
        self.pipeline.connect(
            "intent_classifier.valid_reply", "intent_classifier_router.valid_reply"
        )

        self.pipeline.connect(
            "intent_classifier_router.go_to_chat", "chat_prompt_builder.query"
        )
        self.pipeline.connect(
            "intent_classifier_router.go_to_retrieval",
            "rag_contextualizer_prompt_builder.memories",
        )
        self.pipeline.connect(
            "rag_contextualizer_prompt_builder.prompt", "rag_contextualizer"
        )
        self.pipeline.connect("rag_contextualizer.replies", "list_to_str_adapter")
        self.pipeline.connect("list_to_str_adapter", "embedder.text")
        self.pipeline.connect("list_to_str_adapter", "rag_prompt_builder.query")
        self.pipeline.connect("list_to_str_adapter", "rag_answer_builder.query")

        self.pipeline.connect("embedder.embedding", "retriever.query_embedding")
        self.pipeline.connect("retriever", "rag_prompt_builder.documents")
        self.pipeline.connect("rag_prompt_builder.prompt", "rag_llm.messages")
        self.pipeline.connect("rag_llm.replies", "rag_answer_builder.replies")
        self.pipeline.connect("chat_prompt_builder.prompt", "chat_llm.messages")

    def run(self, messages: list[ChatMessage]):
        import logging
        from haystack import tracing
        from haystack.tracing.logging_tracer import LoggingTracer

        logging.basicConfig(
            format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING
        )
        logging.getLogger("haystack").setLevel(logging.DEBUG)

        tracing.tracer.is_content_tracing_enabled = (
            True  # to enable tracing/logging content (inputs/outputs)
        )
        tracing.enable_tracing(
            LoggingTracer(
                tags_color_strings={
                    "haystack.component.input": "\x1b[1;31m",
                    "haystack.component.name": "\x1b[1;34m",
                }
            )
        )

        memories = []
        for message in messages:
            memory = ""
            if message.is_from(ChatRole.ASSISTANT):
                memory += "Assistant: "
            else:
                memory += "User: "

            memory += message.text + "\n"

            memories.append(memory)

        res = self.pipeline.run(
            {
                "prompt_builder_for_intent_classifier": {"passage": messages[-1].text},
                "intent_classifier_router": {"memories": memories},
            },
            include_outputs_from={"retriever"},
        )
        # logger = logging.getLogger("ray.serve")
        # for doc in res["retriever"]["documents"]:
        #     logger.info(doc.content)

        return res

    def _llm_component(
        self,
        api_key: Secret = Secret.from_token("PLACEHOLDER_KEY"),
        model: str = "allenai/OLMo-2-1124-13B-Instruct",
        api_base_url: str = "http://localhost:8000/v1",
        generation_kwargs: Dict = {"max_tokens": 512},
    ) -> GoogleAIGeminiChatGenerator:
        """
        Returns an OpenAIChatGenerator object with an endpoint to our self-hosted LLM.

        Args:
            api_key: A Secret containing an OpenAI API key.
            model: The HuggingFace name of the model.
            api_base_url: The base URL of the LLM endpoint.
            generation_kwargs: Additional keyword arguments to pass for LLM generation.
        """
        # llm = OpenAIChatGenerator(
        #     api_key=api_key,  # Placeholder api_key is needed for compatibility for OpenAI API
        #     model=model,
        #     api_base_url=api_base_url,
        #     generation_kwargs=generation_kwargs,
        # )
        llm = GoogleAIGeminiChatGenerator(
            model="gemini-2.0-flash",
        )
        return llm

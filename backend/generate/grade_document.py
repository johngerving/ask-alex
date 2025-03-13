import asyncio
from concurrent.futures import ThreadPoolExecutor
from haystack import component
from haystack.utils import Secret
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.components.builders import ChatPromptBuilder
from haystack import Document, Pipeline

from intent_classifier import IntentClassifier

from typing import Dict, List


@component
class DocumentRelevancyFilter:
    '''
    A custom Haystack component to filter out irrelevant documents using an LLM.
    '''

    def __init__(
        self,
        api_key: Secret = Secret.from_token("PLACEHOLDER_KEY"),
        model: str = "allenai/OLMo-2-1124-13B-Instruct",
        api_base_url: str = "http://localhost:8000/v1",
        generation_kwargs: Dict = {"max_tokens": 512}
    ):
        llm = OpenAIChatGenerator(
            api_key=api_key, # Placeholder api_key is needed for compatibility for OpenAI API
            model=model,
            api_base_url=api_base_url,
            generation_kwargs=generation_kwargs
        )

        self.classes = {
            "final answer: relevant": "the document is relevant to the query.",
            "final answer: irrelevant": "the document is not relevant to the query."
        }

        classifier = IntentClassifier(self.classes)

        prompt_template = [
            ChatMessage.from_system(
                """
                You are a grader assessing relevance of a retrieved document to a user question. 
                Given a user query and a candidate document excerpt, carefully evaluate the relevance of the document to the query. 
                Use a skeptical, step-by-step reasoning process, questioning assumptions and considering alternative interpretations. 
                At the end of your reasoning, output one word: either "Relevant" or "Irrelevant".

                Example 1:

                User Query: "What are the effects of climate change on polar bear populations?"
                Document Excerpt: "Recent research indicates that as Arctic ice diminishes, polar bears are forced to travel longer distances, which increases their energy expenditure and impacts their ability to hunt prey."

                Chain-of-Thought Reasoning:

                The query focuses on the effects of climate change on polar bears.
                The document explicitly discusses the reduction of Arctic ice—a direct effect of climate change—and connects it to changes in polar bear behavior and energy expenditure.
                I question if there’s any alternative interpretation: Is this just about polar bear behavior, or does it relate to climate change? Clearly, the diminishing ice is a climate change effect.
                The connection between habitat changes and polar bear survival is evident.

                Final Answer: Relevant

                ---

                Example 2:

                User Query: "How do quantum computers affect cryptography?"

                Document Excerpt: "The latest graphics processing units (GPUs) have revolutionized real-time rendering in modern video games by significantly boosting frame rates."

                Chain-of-Thought Reasoning:

                The query is about the intersection of quantum computing and cryptography.
                The document, however, talks exclusively about GPUs and their role in video game rendering.
                There is no mention of quantum computing, cryptography, or related security impacts.
                I remain skeptical: Could there be an indirect connection? Not in this case, as GPUs and quantum computers are distinct topics.

                Final Answer: Irrelevant

                ---

                Example 3:

                User Query: "What is the relationship between social media usage and mental health?"

                Document Excerpt: "Several studies have found that while social media platforms can help users maintain connections, excessive usage is often linked to increased feelings of loneliness and anxiety."

                Chain-of-Thought Reasoning:

                The query is investigating the link between social media and mental health outcomes.
                The document discusses both positive (maintaining connections) and negative (loneliness and anxiety) aspects related to social media usage.
                I critically assess: Does the document address the query’s focus? Yes, it directly considers mental health implications tied to social media behavior.
                Although the excerpt covers multiple dimensions, the central relationship is clearly addressed.

                Final Answer: Relevant
                """
            ),
            ChatMessage.from_user(
                """
                User Query: "{{ query }}"

                Document Excerpt: "{{ document }}"

                {% if invalid_replies and error_message %}
                You already created the following output in a previous attempt: {{invalid_replies}}
                However, this doesn't comply with the format requirements from above and triggered this Python exception: {{error_message}}
                Correct the output and try again. Just return the corrected output without any extra explanations.
                {% endif %} 

                Chain-of-Thought Reasoning:
                """
            )
        ]

        prompt_builder = ChatPromptBuilder(
            template = prompt_template,
            required_variables=['document', 'query']
        )

        self.pipeline = Pipeline()
        
        self.pipeline.add_component("prompt_builder_for_intent_classifier", prompt_builder)
        self.pipeline.add_component("intent_classifier_llm", llm) # Pass prompt to LLM
        self.pipeline.add_component("intent_classifier", classifier) 

        self.pipeline.connect("prompt_builder_for_intent_classifier.prompt", "intent_classifier_llm.messages")
        self.pipeline.connect("intent_classifier_llm.replies", "intent_classifier")
        self.pipeline.connect("intent_classifier.invalid_replies", "prompt_builder_for_intent_classifier.invalid_replies")
        self.pipeline.connect("intent_classifier.error_message", "prompt_builder_for_intent_classifier.error_message")


    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document], query: str):
        filtered_documents = self._pfilter(lambda docs: self._is_relevant(docs, query), documents, len(documents))

        return {
            "documents": filtered_documents
        }

    def _pfilter(self, filter_func: int, arr: int, max_workers: int):
        with ThreadPoolExecutor(max_workers) as e:
            booleans = e.map(filter_func, arr)
            return [x for x, b in zip(arr, booleans) if b]
    
    def _is_relevant(self, document: Document, query: str) -> bool:
        '''
        Uses an LLM to determine whether a document is relevant to a query or not.
        '''

        import logging
        logger = logging.getLogger("ray.serve")
        logger.info(document)

        res = self.pipeline.run({"prompt_builder_for_intent_classifier": {"document": document.content, "query": query}})

        if not "valid_reply" in res["intent_classifier"]:
            raise Exception("Field 'valid_reply' not found in intent_classifier response")
        
        reply = res["intent_classifier"]["valid_reply"]

        if reply not in self.classes.keys():
            raise Exception(f"Reply {reply} is not a valid class")

        return reply == 'final answer: relevant'

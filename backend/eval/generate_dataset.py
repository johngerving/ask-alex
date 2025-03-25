from concurrent.futures import ThreadPoolExecutor
import json
import os
import re
from typing import Any, Dict, List
from dotenv import load_dotenv
from haystack import Document, Pipeline
import psycopg
from pydantic import BaseModel
from transformers import AutoTokenizer
import pandas as pd
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from output_validator import OutputValidator

from prompts import (
    question_groundedness_critique_prompt,
    question_relevance_critique_prompt,
    question_standalone_critique_prompt,
)

load_dotenv()

MAX_TOKEN_LENGTH = 3500 - 512
NUM_QUESTIONS = 100

conn_str = os.getenv("PG_CONN_STR")
aws_endpoint_url = os.getenv("AWS_ENDPOINT_URL")

if conn_str is None:
    raise Exception("PG_CONN_STR not found")
if aws_endpoint_url is None:
    raise Exception("AWS_ENDPOINT_URL not found")

documents: List[Document] = []

tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-13B-Instruct")

with psycopg.connect(conn_str) as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT document FROM documents ORDER BY random() LIMIT 1000")
        results = cur.fetchall()

        for result in results:
            obj = json.loads(result[0])

            document = Document.from_dict(obj)

            num_tokens = len(tokenizer.tokenize(document.content))

            if num_tokens <= MAX_TOKEN_LENGTH:
                documents.append(document)

            if len(documents) >= NUM_QUESTIONS:
                break

question_prompt_builder = ChatPromptBuilder(
    template=[
        ChatMessage.from_user(
            """
        Your task is to write a factoid question and an answer given a context.
        Your factoid question should be answerable with a specific, concise piece of factual information from the context.
        Your factoid question should be formulated in the same style as questions users could ask in a search engine.
        This means that your factoid question MUST NOT mention something like "according to the passage" or "context".
        DO NOT answer the question. Only output the question on its own.

        Provide your answer as follows:

        <example>
        Output:
        Factoid question: (your factoid question)
        </example>

        Now here is the context.

        Context: {{ document.content }} 

        Output:
        Factoid question:"""
        )
    ]
)

question_generator = OpenAIChatGenerator(
    api_key=Secret.from_token("PLACEHOLDER_KEY"),
    model="allenai/OLMo-2-1124-13B-Instruct",
    api_base_url="http://localhost:8000/v1",
    generation_kwargs={"max_tokens": 512},
)

question_pipeline = Pipeline()

question_pipeline.add_component("prompt_builder", question_prompt_builder)
question_pipeline.add_component("generator", question_generator)

question_pipeline.connect("prompt_builder", "generator")


answer_prompt_builder = ChatPromptBuilder(
    template=[
        ChatMessage.from_system(
            """
        You are ALEX, a helpful AI assistant designed to provide information about Humboldt. Respond to the input as a friendly AI assistant, generating human-like text, and follow the instructions in the input if applicable. Keep the response concise and engaging, using Markdown when appropriate. Use a conversational tone and provide helpful and informative responses, utilizing only the context provided to formulate answers.
        """
        ),
        ChatMessage.from_user(
            """
        Please provide an answer based solely on the provided source, in no more than 50 words. 
               
        Below is a document: 

        {{ document.content }} 

        Answer the following query:

        Query: {{query}}
        Answer:
        """
        ),
    ]
)

answer_generator = OpenAIChatGenerator(
    api_key=Secret.from_token("PLACEHOLDER_KEY"),
    model="allenai/OLMo-2-1124-13B-Instruct",
    api_base_url="http://localhost:8000/v1",
    generation_kwargs={"max_tokens": 512},
)

answer_pipeline = Pipeline()

answer_pipeline.add_component("prompt_builder", answer_prompt_builder)
answer_pipeline.add_component("generator", answer_generator)

answer_pipeline.connect("prompt_builder", "generator")


class Evaluation(BaseModel):
    explanation: str
    rating: int


json_schema = json.dumps(Evaluation.model_json_schema(), indent=4)

evaluation_prompt_builder = ChatPromptBuilder(
    variables=["invalid_replies", "error_message"]
)

evaluation_generator = OpenAIChatGenerator(
    api_key=Secret.from_token("PLACEHOLDER_KEY"),
    model="allenai/OLMo-2-1124-13B-Instruct",
    api_base_url="http://localhost:8000/v1",
    generation_kwargs={"max_tokens": 512},
)

output_validator = OutputValidator(pydantic_model=Evaluation)

evaluation_pipeline = Pipeline(max_runs_per_component=5)

evaluation_pipeline.add_component("prompt_builder", evaluation_prompt_builder)
evaluation_pipeline.add_component("generator", evaluation_generator)
evaluation_pipeline.add_component("output_validator", output_validator)

evaluation_pipeline.connect("prompt_builder.prompt", "generator.messages")
evaluation_pipeline.connect("generator.replies", "output_validator")

evaluation_pipeline.connect(
    "output_validator.invalid_replies", "prompt_builder.invalid_replies"
)
evaluation_pipeline.connect(
    "output_validator.error_message", "prompt_builder.error_message"
)


def evaluate_question(question: Dict[str, Any]):
    new_question = question.copy()

    try:
        groundedness = json.loads(
            evaluation_pipeline.run(
                {
                    "prompt_builder": {
                        "template": [
                            ChatMessage.from_user(question_groundedness_critique_prompt)
                        ],
                        "template_variables": {
                            "question": question["question"],
                            "context": question["context"],
                            "schema": json_schema,
                        },
                    }
                }
            )["output_validator"]["valid_replies"][0].text
        )

        relevance = json.loads(
            evaluation_pipeline.run(
                {
                    "prompt_builder": {
                        "template": [
                            ChatMessage.from_user(question_relevance_critique_prompt)
                        ],
                        "template_variables": {
                            "question": question["question"],
                            "schema": json_schema,
                        },
                    }
                }
            )["output_validator"]["valid_replies"][0].text
        )

        standalone = json.loads(
            evaluation_pipeline.run(
                {
                    "prompt_builder": {
                        "template": [
                            ChatMessage.from_user(question_standalone_critique_prompt)
                        ],
                        "template_variables": {
                            "question": question["question"],
                            "schema": json_schema,
                        },
                    }
                }
            )["output_validator"]["valid_replies"][0].text
        )

        evaluations = {
            "groundedness": groundedness,
            "relevance": relevance,
            "standalone": standalone,
        }

        for criterion, evaluation in evaluations.items():
            score, eval = (evaluation["rating"], evaluation["explanation"])

            new_question.update(
                {
                    f"{criterion}_score": score,
                    f"{criterion}_eval": eval,
                }
            )
    except Exception as e:
        print(e)

    return new_question


with ThreadPoolExecutor(max_workers=16) as e:
    questions = list(
        e.map(
            lambda doc: question_pipeline.run({"prompt_builder": {"document": doc}})[
                "generator"
            ]["replies"][0].text,
            documents,
        )
    )

    question_regex = re.compile(r"^.*\?", re.I)
    questions = [
        question_regex.search(q)[0] if question_regex.search(q) is not None else ""
        for q in questions
    ]

    outputs = [
        {"question": q, "context": c.content} for q, c in zip(questions, documents)
    ]

    outputs = list(e.map(evaluate_question, outputs))

    outputs = [
        output
        for output in outputs
        if output["groundedness_score"] >= 3
        and output["relevance_score"] >= 3
        and output["standalone_score"] >= 3
    ]

    # generated_questions = pd.DataFrame.from_dict(outputs)
    # generated_questions.dropna(inplace=True)
    # generated_questions = generated_questions.loc[
    #     (generated_questions["groundedness_score"] >= 3)
    #     & (generated_questions["relevance_score"] >= 3)
    #     & (generated_questions["standalone_score"] >= 3)
    # ]

    answers = list(
        e.map(
            lambda doc, q: answer_pipeline.run(
                {"prompt_builder": {"document": doc, "query": q}}
            )["generator"]["replies"][0].text,
            [output["context"] for output in outputs],
            [output["question"] for output in outputs],
        )
    )

    questions_answers = pd.DataFrame.from_dict(outputs)
    questions_answers["answers"] = answers
    questions_answers.dropna(inplace=True)

    questions_answers.to_csv("qa.csv")

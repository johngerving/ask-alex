from haystack.components.evaluators.document_mrr import DocumentMRREvaluator
from haystack.components.evaluators.faithfulness import FaithfulnessEvaluator
from haystack.components.evaluators.sas_evaluator import SASEvaluator
from haystack.evaluation.eval_run_result import EvaluationRunResult

from concurrent.futures import ThreadPoolExecutor
import os
from haystack import Pipeline
import pandas as pd
import requests

rag_endpoint = os.getenv("RAG_ENDPOINT")
if rag_endpoint is None:
    raise Exception("Missing environment variable RAG_ENDPOINT")

# Get QA dataset
dataset = pd.read_csv("qa.csv")
dataset.drop(columns=["Unnamed: 0"], inplace=True)
dataset = dataset.loc[dataset["question"] != ""]
questions = dataset["question"].to_list()
ground_truth_answers = dataset["answers"].to_list()


def run_pipeline(question: str) -> str:
    """
    Runs the RAG pipeline on a question.

    Args:
        question: The question to run through the RAG pipeline
    """
    try:
        # Make request to RAG endpoint
        res = requests.post(
            rag_endpoint, json={"messages": [{"content": question, "type": "user"}]}
        )
        try:
            answer = res.json()["response"]
        except Exception as e:
            print("There was an error with the following question:")
            print(f"\t{question}")
            print(f"\tResponse: {res.text}")
            raise
    except Exception as e:
        print(f"\tError: {e}")
        answer = ""
    return answer


# Get answers from pipeline
with ThreadPoolExecutor(max_workers=len(questions)) as e:
    rag_answers = list(e.map(run_pipeline, questions))

# Evaluation pipeline
eval_pipeline = Pipeline()
eval_pipeline = Pipeline()
eval_pipeline.add_component(
    "sas_evaluator", SASEvaluator(model="sentence-transformers/all-MiniLM-L6-v2")
)

# Evaluate answers to ground truth
results = eval_pipeline.run(
    {
        "sas_evaluator": {
            "predicted_answers": rag_answers,
            "ground_truth_answers": list(ground_truth_answers),
        },
    }
)


inputs = {
    "question": list(questions),
    "answer": list(ground_truth_answers),
    "predicted_answer": rag_answers,
}

# Generate evaluation report
evaluation_result = EvaluationRunResult(
    run_name="pubmed_rag_pipeline", inputs=inputs, results=results
)
print(evaluation_result.aggregated_report())

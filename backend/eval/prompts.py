question_groundedness_critique_prompt = """
You will be given a context and a question.
Your task is to provide a 'total rating' scoring how well one can answer the given question unambiguously with the given context.
Give your answer on a scale of 1 to 5, where 1 means that the question is not answerable at all given the context, and 5 means that the question is clearly and unambiguously answerable with the context.

Follow this JSON schema, but only return the actual instance without any additional schema definition:
{{ schema }}

Examples:
    Context: The Eiffel Tower is one of Paris's most iconic landmarks, standing tall in the heart of the city.
    Question: Where is the Eiffel Tower located?
    Output:
    {
        "explanation": "The context explicitly states the location, making the answer unambiguous.",
        "rating": 5
    }

    Context: The latest report from the weather center indicates a high probability of rain in several cities, though detailed forecasts are pending.
    Question: Is it going to rain in New York today?
    Output:
    {
        "explanation": "The context does not mention New York specifically, making the question hard to answer with confidence.",
        "rating": 1
    }

    Context: Which monument in Paris is the most famous?
    Question: Which monument in Paris is the most famous?
    Output:
    {
        "explanation": "The context is vague about which monument is most famous, leading to ambiguity.",
        "rating": 3
    }
End of examples.

{% if invalid_replies and error_message %}
  You already created the following output in a previous attempt: {{invalid_replies}}
  However, this doesn't comply with the format requirements from above and triggered this Python exception: {{error_message}}
  Correct the output and try again. Just return the corrected output without any extra explanations.
{% endif %}

Question: {{ question }}\n
Context: {{ context }}\n
Output:
"""

question_relevance_critique_prompt = """
You will be given a question.
Your task is to provide a 'total rating' representing how useful this question can be to users trying to find information about documents relevant to Humboldt County.
Give your answer on a scale of 1 to 5, where 1 means that the question is not useful at all, and 5 means that the question is extremely useful.

For each input question, provide:
    1. A brief justification explaining why the document is relevant or not, noting any ambiguities or mismatches.
    2. A relevance rating (1 to 5):
        1: Completely irrelevant
        3: Moderately relevant
        5: Highly relevant

Follow this JSON schema, but only return the actual instance without any additional schema definition:
{{ schema }}

Examples:
    Question: Where can I find the official meeting minutes for Humboldt County Board meetings?
    Output:
    {
        "explanation": "The question clearly targets official documents and meeting records, which is exactly what users looking for Humboldt County documents would need.",
        "rating": 5
    }

    Question: How did the Gold Rush impact local communities in California?
    Output:
    {
        "explanation": "While this touches on California history, the relevance to Humboldt County is indirect unless further context suggests a specific local focus.",
        "rating": 2
    }
End of examples.

{% if invalid_replies and error_message %}
  You already created the following output in a previous attempt: {{invalid_replies}}
  However, this doesn't comply with the format requirements from above and triggered this Python exception: {{error_message}}
  Correct the output and try again. Just return the corrected output without any extra explanations.
{% endif %}

Question: {{ question }}
Output:
"""

question_standalone_critique_prompt = """
You will be given a question.
Your task is to provide a 'total rating' representing how context-independent this question is.
Give your answer on a scale of 1 to 5, where 1 means that the question depends on additional information to be understood, and 5 means that the question makes sense by itself.
For instance, if the question refers to a particular setting, like 'in the context' or 'in the document', the rating must be 1.
The questions can contain obscure technical nouns or acronyms like Humboldt County, Arcata, Gist Hall and still be a 5: it must simply be clear to an operator with access to documentation what the question is about.

You will be given a question. Your task is to evaluate how context-independent the question is by assigning a rating on a scale of 1 to 5, where 1 means the question is highly dependent on additional context (e.g., it includes ambiguous pronouns or unclear references), and 5 means the question is fully self-contained and can be understood without any additional context.
Only use information that is present in the passage. Follow this JSON schema, but only return the actual instances without any additional schema definition:
{{schema}}

Examples:
    Question: What is the capital of France?
    Output:
    {
        "explanation": "The question clearly asks for a factual detail without relying on any external context.",
        "rating": 5
    }

    Question: Do you think it's a good idea?
    Output:
    {
        "explanation": "The question contains an ambiguous reference ("it") that makes it unclear what is being discussed.",
        "rating": 1
    }

    Question: When was the Declaration of Independence signed?
    Output:
    {
        "explanation": "The question is precise and asks for a specific historical fact without needing additional context.",
        "rating": 5
    }

    Question: How does this feature improve our product?
    Output:
    {
        "explanation": "The question refers to "this feature" and "our product" without specifying which feature or product, indicating a dependence on additional context.",
        "rating": 2
    }

    Question: Is the new policy effective compared to previous guidelines?
    Output:
    {
        "explanation": "The question makes a comparison that assumes prior knowledge of both the new policy and the previous guidelines, making it moderately context-dependent.",
        "rating": 3
    }
End of examples.

{% if invalid_replies and error_message %}
  You already created the following output in a previous attempt: {{invalid_replies}}
  However, this doesn't comply with the format requirements from above and triggered this Python exception: {{error_message}}
  Correct the output and try again. Just return the corrected output without any extra explanations.
{% endif %}

Question: {{ question }}
Output:
"""


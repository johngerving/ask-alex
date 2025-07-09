from textwrap import dedent
from typing import List
from llama_index.llms.openrouter import OpenRouter
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import FunctionTool
import os

import psycopg


def call_metadata_agent(prompt: str) -> str:
    """Call the metadata agent to get metadata for a given prompt. Use this for determining the proper metadata to use to search for documents.

    Args:
        prompt (str): The prompt to send to the metadata agent. This should be a detailed, standalone prompt requesting a kind of document and should reflect the user's query.

    Returns:
        str: The response from the metadata agent, which should be a JSON string containing the metadata.
    """

    llm = OpenRouter(
        model="mistralai/mistral-small-3.2-24b-instruct",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        context_window=128000,
        max_tokens=4000,
        is_chat_model=True,
        is_function_calling_model=True,
    )

    with psycopg.connect(os.getenv("PG_CONN_STR")) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT collection FROM collections_distinct ORDER BY collection"
            )
            rows = cur.fetchall()
            collections: List[str] = [row[0] for row in rows]
            collections_str = "\n".join(collections)

            cur.execute(
                "SELECT department FROM departments_distinct ORDER BY department"
            )
            rows = cur.fetchall()
            departments: List[str] = [row[0] for row in rows]
            departments_str = "\n".join(departments)

    response = llm.chat(
        messages=[
            ChatMessage(
                role="system",
                content=dedent(
                    f"""\
                    You are a metadata agent tasked with generating metadata for document retrieval in Cal Poly Humboldt's Digital Commons repository based on user queries.
                    You can utilize the following metadata fields:

                    - query: A Tantivy query string that represents the key words or phrases the user is interested in.
                    - title: The title of the document to search for.
                    - department: The department that published the document. This is also sometimes referred to as the "subject".
                    - collection: The collection that the document belongs to.
                    - start_year: The start of the date range to search for documents.
                    - end_year: The end of the date range to search for documents.

                    You do not have to use all of these fields. Only use the ones that are relevant to the query.

                    ## The Query Field
                    The query field is a Tantivy query string that represents the key words or phrases the user is interested in.
                    This field is used to search for documents that contain the specified keywords in their text.

                    The query can include logical operators such as AND, OR, and NOT (-). You can match phrases by putting them in quotes.

                    If the user is asking for a specific phrase, you MUST use the phrase search operator by surrounding the phrase in quotes. If the user is asking for multiple keywords, you can use the AND operator to combine them.

                    Examples: 

                    "fat AND rat" matches documents that contain both "fat" and "rat" in their text.
                    "fat OR rat" matches documents that contain either "fat" or "rat" in their text.
                    "fat AND rat AND -cat" matches documents that contain "fat" and "rat" but do not contain "cat".
                    ""rain of debris"" matches documents that contain the phrase "rain of debris" in that exact order, with no words in between.
                    "moon AND -"bright sun"" matches documents that contain "moon" but do not contain the exact phrase "bright sun".

                    If you think the user is looking for a specific phrase, you should use the phrase search operator by surrounding the phrase in quotes.
                    
                    You can combine queries with parentheses:
                    Examples:

                    "(fat OR rat) AND (cat OR dog)" matches documents that contain either "fat" or "rat" and either "cat" or "dog".
                    "(fat AND rat) OR (cat AND dog)" matches documents that contain both "fat" and "rat" or both "cat" and "dog".

                    ## The Collection Field
                    The collection field represents the collection of documents that the user is interested in.
                    The Cal Poly Humboldt Digital Commons repository contains the following collections:
                    {collections_str}

                    One of the most common collections is "Cal Poly Humboldt theses and projects", which will often be referred to as just "theses".

                    The collection field is case sensitive, so make sure to use the exact name of the collection.
                    Only include a collection in the metadata you generate if the user explicitly asks for a collection or wants theses.

                    ## The Department Field
                    The department field represents the department that published the document.
                    The Cal Poly Humboldt Digital Commons repository contains documents from the following departments:
                    {departments_str}

                    The department field is case sensitive, so make sure to use the exact name of the department.

                    ## Publication Date
                    The start_year and end_year fields represent the publication date of the document.
                    When the user asks for documents published in a specific year or range of years, you should use these fields to filter the documents.

                    Format your response as a JSON string with the appropriate metadata fields included. If the user asks for a type of document you believe to be invalid, such as a collection that doesn't exist, instead respond saying so and do not return any metadata.

                    Example Inputs/Outputs:

                    User: I'm looking for documents mentioning wildlife conservation.
                    Output: {{ "query": ""wildlife conservation"" }}

                    User: I'm looking for theses mentioning conservation and sustainability.
                    Output: {{ "query": "conservation AND sustainability", "collection": "Cal Poly Humboldt theses and projects" }}

                    User: I'm looking for documents that mention "climate change" published in 2020.
                    Output: {{ "query": ""climate change"", "start_year": 2020, "end_year": 2020 }}

                    User: I'm looking for articles with the subject wildlife.
                    Output: {{ "department": "Wildlife" }}
                    """
                ),
            ),
            ChatMessage(role="user", content=prompt),
        ]
    )

    return response.message.content


tool = FunctionTool.from_defaults(
    fn=call_metadata_agent,
)

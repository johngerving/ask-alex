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

                    - query: A PostgreSQL tsquery string to search for in the document text.
                    - title: The title of the document to search for.
                    - department: The department that published the document.
                    - collection: The collection that the document belongs to.
                    - start_year: The start of the date range to search for documents.
                    - end_year: The end of the date range to search for documents.

                    You do not have to use all of these fields. Only use the ones that are relevant to the query.

                    ## The Query Field
                    The query field is a PostgreSQL tsquery string that represents the keywords the user is interested in.
                    This field is used to search for documents that contain the specified keywords in their text.

                    A tsquery value stores lexemes that are to be searched for, and can combine them using the Boolean operators & (AND), | (OR), and ! (NOT), as well as the phrase search operator <-> (FOLLOWED BY) 

                    Examples: 

                    "fat & rat" matches documents that contain both "fat" and "rat" in their text.
                    "fat | rat" matches documents that contain either "fat" or "rat" in their text.
                    "fat & rat & ! cat" matches documents that contain "fat" and "rat" but do not contain "cat".
                    "rain <-> of <-> debris" matches documents that contain the phrase "rain of debris" in that exact order, with no words in between.

                    If you think the user is looking for a specific phrase, you should use the phrase search operator (<->) to ensure that the phrase is matched exactly.
                    A tsquery cannot have multiple words in a row. This will result in an error. For example, "fat rat" is not a valid tsquery. Instead, you should use "fat & rat" to search for documents that contain both "fat" and "rat", or "fat <-> rat" to search for documents that contain the phrase "fat rat", for example.

                    You can combine queries with parentheses:
                    Examples:

                    "(fat | rat) & (cat | dog)" matches documents that contain either "fat" or "rat" and either "cat" or "dog".
                    "(fat & rat) | (cat & dog)" matches documents that contain both "fat" and "rat" or both "cat" and "dog".

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

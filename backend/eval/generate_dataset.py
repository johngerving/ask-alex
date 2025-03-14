
import json
import os
from typing import List
from dotenv import load_dotenv
from haystack import Document
import psycopg
from transformers import AutoTokenizer
import pandas as pd

load_dotenv()

MAX_TOKEN_LENGTH = 4096 - 512

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

            if len(documents) >= 100:
                break
from pprint import pprint
pprint(documents)
df = pd.DataFrame([document.content for document in documents], columns=['content'])
df.to_csv("documents.csv")
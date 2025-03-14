import os
from dotenv import load_dotenv
import psycopg
import ray
import ray.data
from pyarrow import fs

### Migrate document data from S3 parquet to Postgres table

load_dotenv()

conn_str = os.getenv("PG_CONN_STR")
aws_endpoint_url = os.getenv("AWS_ENDPOINT_URL")

if conn_str is None:
    raise Exception("PG_CONN_STR not found")
if aws_endpoint_url is None:
    raise Exception("AWS_ENDPOINT_URL not found")

# Create the documents table if it doesn't exist
with psycopg.connect(conn_str) as conn:
    conn.cursor().execute("DROP TABLE IF EXISTS documents")
    conn.cursor().execute("CREATE TABLE documents (link TEXT PRIMARY KEY, document TEXT)")

# Create a filesystem to store the converted documents in
filesys = fs.S3FileSystem(endpoint_override=aws_endpoint_url)

# Get dataset from S3
ds = ray.data.read_parquet("s3://documents/", filesystem=filesys)

# Write dataset to Postgres 
ds.write_sql("INSERT INTO documents VALUES(%s, %s)", lambda: psycopg.connect(conn_str))

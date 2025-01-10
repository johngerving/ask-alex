import json
from haystack import Pipeline, Document
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter

import pika
from pika_credentials import get_pika_connection
from pika.adapters.blocking_connection import BlockingChannel
from pika.spec import Basic

document_store = PgvectorDocumentStore(
    table_name="haystack_docs",
    embedding_dimension=768,
    vector_function="cosine_similarity",
    recreate_table=True,
    search_strategy="hnsw"
)

indexing = Pipeline()
indexing.add_component("embedder", SentenceTransformersDocumentEmbedder())
indexing.add_component("writer", DocumentWriter(document_store))
indexing.connect("embedder", "writer")

def on_message(ch: BlockingChannel, method_frame: Basic.Deliver, _header_frame, body: str):
    obj = json.loads(body)

    document = Document.from_dict(obj)

    print(f"Received document with id {document.id}") 
    indexing.run(
        {
            "documents": [document]
        }
    )
    print(f"Finished indexing document with id {document.id}")
    ch.basic_ack(delivery_tag=method_frame.delivery_tag)

connection = get_pika_connection()
channel = connection.channel()
channel.basic_consume("indexing_queue", on_message)

try:
    channel.start_consuming()
except KeyboardInterrupt:
    channel.stop_consuming()
connection.close()


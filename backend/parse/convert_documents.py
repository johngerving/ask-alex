import logging
import re
import psycopg
from psycopg.types.json import Jsonb
import ray
import ray.data

from digitalcommons import dataset_from_digitalcommons
from docling.datamodel.base_models import InputFormat, ConversionStatus
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling_core.types.doc.document import DoclingDocument
from llama_index.core import Document as LIDocument

import json
import uuid

from typing import Any, Dict, List
import numpy as np

import os


class Converter:
    def __init__(self):

        # Use the GPU to parse the PDFs
        accelerator_options = AcceleratorOptions(
            num_threads=16, device=AcceleratorDevice.CUDA
        )

        pipeline_options = PdfPipelineOptions()
        pipeline_options.accelerator_options = accelerator_options

        # Create a Docling converter to convert our PDFs
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        self.logger = logging.getLogger("ray.data")

    def __call__(self, input_batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        print(f'Processing batch of size {len(input_batch["link"])}')

        documents: List[str] = []
        statuses: List[str] = []
        with psycopg.connect(os.getenv("PG_CONN_STR")) as conn:
            with conn.cursor() as cur:
                for link, metadata in zip(input_batch["link"], input_batch["metadata"]):
                    # Check if document already exists in the database
                    cur.execute(
                        "SELECT document FROM documents WHERE link = %s", (link,)
                    )
                    row = cur.fetchone()

                    li_doc = None
                    if row is not None:
                        existing_doc = row[0]
                        li_doc = LIDocument.from_dict(existing_doc)
                        li_doc.metadata = json.loads(metadata)
                    else:
                        # If the document does not exist, we will convert it
                        try:
                            result = self.converter.convert(
                                link,
                                raises_on_error=False,
                                headers={
                                    "User-Agent": "Mozilla/5.0 (X11; Linux i686; rv:124.0) Gecko/20100101 Firefox/124.0"
                                },
                            )

                            if result.status == ConversionStatus.FAILURE:
                                raise Exception(f"{result.errors}")

                            dl_doc = result.document

                            text = json.dumps(dl_doc.export_to_dict())

                            metadata: Dict[Any, Any] = json.loads(metadata)

                            li_doc = LIDocument(
                                doc_id=str(uuid.uuid4()),
                                text=text,
                                metadata=metadata,
                            )

                        except Exception as e:
                            self.logger.error(f"Error converting document {link}: {e}")

                    if li_doc:
                        li_doc.set_content(li_doc.text.replace("\x00", ""))
                        li_doc.set_content(li_doc.text.replace("\\u0000", ""))
                        li_doc.set_content(re.sub(r"/\s\s+/g", " ", li_doc.text))

                    statuses.append("success" if li_doc else "failure")
                    documents.append(json.dumps(li_doc.to_dict() if li_doc else None))

        return {"link": input_batch["link"], "status": statuses, "document": documents}


def is_successful(row: Dict[str, Any]) -> bool:
    return row["status"] == "success"


class ExtractTextContent:
    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Extracts text content from the documents in the batch.
        """
        documents = [LIDocument.from_dict(json.loads(doc)) for doc in batch["document"]]

        dl_docs = [DoclingDocument.model_validate_json(doc.text) for doc in documents]

        texts = [dl_doc.export_to_markdown() for dl_doc in dl_docs]

        return {
            "link": batch["link"],
            "document": batch["document"],
            "text": texts,
        }


class SaveDocuments:
    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        documents_dict = [Jsonb(json.loads(doc)) for doc in batch["document"]]

        values = zip(batch["link"], documents_dict, batch["text"])
        with psycopg.connect(os.getenv("PG_CONN_STR"), autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO documents (link, document, text) VALUES (%s, %s, %s) ON CONFLICT (link) DO UPDATE SET document = EXCLUDED.document, text = EXCLUDED.text",
                    [(link, doc, text) for link, doc, text in values],
                )
        return batch


def convert_documents():
    """
    Converts a dataset containing links to PDFs to markdown documents.

    Args:
        ds: A ray.data.Dataset with a column containing links to PDFs.
    """

    conn_str = os.getenv("PG_CONN_STR")
    if conn_str is None:
        raise Exception("Missing environment variable PG_CONN_STR")

    ds = dataset_from_digitalcommons()

    old_ds = ray.data.read_sql(
        "SELECT link FROM documents",
        lambda: psycopg.connect(conn_str),
    )

    # Remove old documents that are no longer present in the Digital Commons dataset
    remove_old_documents(ds, old_ds)

    # Convert PDFs to markdown documents
    ds = ds.map_batches(
        Converter,
        concurrency=8,  # Run 8 workers
        batch_size=32,  # Send batches of 32 links
        num_gpus=1,  # Use 1 GPU per worker
        num_cpus=1,  # Use 1 CPUs per worker
    )

    ds = ds.filter(fn=is_successful)

    ds = ds.map_batches(
        ExtractTextContent,
        batch_size=128,
        num_cpus=1,
        concurrency=8,
    )

    ds = ds.map_batches(
        SaveDocuments,
        batch_size=128,
        num_cpus=1,
        concurrency=8,
    )

    ds.count()


def remove_old_documents(new_ds: ray.data.Dataset, old_ds: ray.data.Dataset) -> None:
    """
    Removes documents that are no longer present in the Digital Commons dataset.
    """
    old_links = old_ds.join(
        new_ds,
        join_type="left_outer",
        on=("link",),
        num_partitions=8,
    )

    filtered = old_links.filter(lambda row: row["metadata"] is None).take_all()

    with psycopg.connect(os.getenv("PG_CONN_STR"), autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.executemany(
                "DELETE FROM documents WHERE link = %s",
                [(row["link"],) for row in filtered],
            )

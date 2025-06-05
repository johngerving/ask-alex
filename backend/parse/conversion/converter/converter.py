import logging
from typing import Any, Dict, List
from ray import serve
from ray.serve.handle import DeploymentHandle
from fastapi import FastAPI

from docling.datamodel.base_models import InputFormat, ConversionStatus
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.datamodel.base_models import InputFormat, ConversionStatus
from llama_index.core import Document
from starlette.requests import Request


app = FastAPI()

logger = logging.getLogger("ray.serve")

BATCH_SIZE = 32  # Maximum batch size for the converter


@serve.deployment(
    autoscaling_config={
        "min_replicas": 0,
        "max_replicas": 4,
    },
    ray_actor_options={
        "num_cpus": 1,
        "num_gpus": 1,  # Use one GPU per replica
    },
    max_ongoing_requests=BATCH_SIZE,
    max_queued_requests=BATCH_SIZE * 2,  # Allow some queuing
)
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

    @serve.batch(max_batch_size=BATCH_SIZE, batch_wait_timeout_s=5)
    async def __call__(self, requests: List[Request]):
        requests_json = [await request.json() for request in requests]
        links = [body["link"] for body in requests_json]

        logger.info(f"Processing batch of size {len(links)}")
        # Convert all the documents in the input batch - need a User-Agent header to avoid getting blocked
        conversion_results = self.converter.convert_all(
            links,
            raises_on_error=False,
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux i686; rv:124.0) Gecko/20100101 Firefox/124.0"
            },
        )

        return [
            (
                {"status": "success", "document": result.document.export_to_dict()}
                if result.status == ConversionStatus.SUCCESS
                else {"status": "failure", "error": result.errors}
            )
            for result in conversion_results
        ]


app = Converter.bind()

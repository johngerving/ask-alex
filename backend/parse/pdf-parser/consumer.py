import functools
import json
import threading
import pika
from pika.adapters.blocking_connection import BlockingChannel
from pika.spec import Basic
import pika.channel


from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.datamodel.base_models import InputFormat

from docling_converter import DoclingConverter 

from rabbitmq_pipeline import PipelineStep
from typing import Optional

class PDFParser(PipelineStep):
    def __init__(
            self,
            consumer_queue: str,
            publisher_queues: Optional[str] = None,
            prefetch_count: Optional[int] = 1
    ):
        super().__init__(consumer_queue, publisher_queues, prefetch_count)

        # Use GPU to accelerate converter
        accelerator_options = AcceleratorOptions(
            num_threads=16, device=AcceleratorDevice.CUDA
        )

        pipeline_options = PdfPipelineOptions()
        pipeline_options.accelerator_options = accelerator_options

        dl_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )

        self.converter = DoclingConverter(
            converter=dl_converter
        )

    def work(self, delivery_tag: int, body: str) -> list[str]:
        print(f"got work - {body}")
        links = json.loads(body)

        if not isinstance(links, list):
            raise ValueError("Object is not a list")

        print(f"Processing document batch of size {len(links)}...")

        documents = self.converter.run(links)

        objects = []
        for document in documents:
            obj = json.dumps(document.to_dict())
            print(f"Processed document of length {len(obj)}")
            objects.append(obj)

        return objects 

    def publish_document(self, ch: BlockingChannel, body: str):
        if ch.is_open:
            ch.basic_publish(exchange="", routing_key="indexing_queue", body=body)
        else:
            pass
 
    def ack_message(self, delivery_tag: int, result: list[str]):
        if self.channel.is_open:
            for document in result:
                self.channel.basic_publish(exchange="", routing_key="indexing_queue", body=document)
            self.channel.basic_ack(delivery_tag)
        else:
            pass

    def reject_message(self, delivery_tag: int):
        if self.channel.is_open:
            self.channel.basic_reject(delivery_tag)
        else:
            pass

def main():
    parser = PDFParser(
        consumer_queue="links_queue",
        publisher_queues=["indexing_queue"],
        prefetch_count=1
    )
    parser.run()

if __name__ == "__main__":
    main()

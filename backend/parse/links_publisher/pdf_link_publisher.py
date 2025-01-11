import json
import os

import pika
from ask_alex_utils.rabbitmq_pipeline import PipelineStep

import requests
from requests.adapters import HTTPAdapter, Retry


class PDFLinkPublisher(PipelineStep):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        '''
        Initialize link publisher.
        '''
        super().__init__(
            *args,
            **kwargs
        )

        # List to hold batches before publishing to the queue
        self.batch = []
        self.batch_size = 2 
    
    def ack_message(self, delivery_tag, result):
        pass 
    def reject_message(self, delivery_tag):
        pass
    def work(self, delivery_tag, body):
        pass 

    def run(self):
        '''Get a list of PDF links from Digital Commons and publish to a RabbitMQ queue.'''
        # Initialize HTTP session
        session = requests.Session()

        # Configure retries
        retries = Retry(total=5,
                        backoff_factor=0.1,
                        status_forcelist=[ 401, 403, 503 ],
                        raise_on_status=True)

        session.mount('https://', HTTPAdapter(max_retries=retries))

        start = 0
        LIMIT = 500
        while True:
            # Get list of documents
            resp = session.get(f"https://content-out.bepress.com/v2/digitalcommons.humboldt.edu/query?download_format=pdf&start={start}&limit={LIMIT}", headers={"Authorization": os.getenv("API_TOKEN")})
            body = resp.json()

            # Check to make sure response fields are populated

            if "results" not in body:
                raise Exception("Document results not found")

            if not isinstance(body["results"], list):
                raise Exception(f"Invalid type {type(body['results'])} for document results")

            results = body["results"]

            # Stop if no results
            if len(results) == 0:
                break

            for result in results:
                # Check to make sure result is valid
                if not isinstance(result, dict):
                    raise Exception(f"Invalid type {type(result)} for document result element")

                if "download_link" not in result or len(result["download_link"]) == 0:
                    raise Exception("Document result does not contain 'download_link' field")
                
                # Add the link to the list
                download_link = result["download_link"]
                self.batch.append(download_link)

                # If we have a full batch, push it to the queue
                if len(self.batch) >= self.batch_size:
                    body = json.dumps(self.batch)

                    self.channel.basic_publish(exchange="", routing_key="links_queue", body=body)
                    self.batch = []

            start += LIMIT

        # Publish remaining links to queue
        if len(self.batch) > 0:
            body = json.dumps(self.batch)

            self.channel.basic_publish(exchange="", routing_key="links_queue", body=body)


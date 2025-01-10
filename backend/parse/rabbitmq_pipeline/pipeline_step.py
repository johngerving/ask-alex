import threading

import pika
from pika.adapters.blocking_connection import BlockingChannel
from pika.spec import Basic
from pika_credentials import get_pika_connection

from abc import ABC, abstractmethod
from typing import Optional, Any
import functools


class PipelineStep(ABC):
    def __init__(
        self,
        consumer_queue: str,
        publisher_queues: Optional[str] = None, 
        prefetch_count: Optional[int] = 1 
    ):
        # Initialize RabbitMQ connection
        self.connection = get_pika_connection()
        self.channel = self.connection.channel()

        # Set prefetch count
        print(f"Initializing with prefetch count {prefetch_count}")
        self.channel.basic_qos(prefetch_count=prefetch_count)

        # Declare queues
        self.channel.queue_declare(queue=consumer_queue)

        # Start consuming the queue
        self.channel.basic_consume(consumer_queue, self._on_message)

        # Declare queues to publish to if provided
        if publisher_queues is not None:
            for queue in publisher_queues:
                self.channel.queue_declare(queue=queue)

        # Create list of threads to perform work on
        self._threads = []

    def _on_message(self, _channel: BlockingChannel, method: Basic.Deliver, _header_frame, body: str):
        '''
        Receive a message from the consumer queue. Start performing work on a new thread. 
        '''
        
        delivery_tag = method.delivery_tag
        
        # Start a new thread with the _do_work callback
        t = threading.Thread(target=self._do_work, args=(delivery_tag, body))
        t.start()
        self._threads.append(t) 

    def _do_work(self, delivery_tag: int, body: str):
        '''
        Perform work. Add threadsafe callback depending on whether the work was successful or not.
        
        Args:
            delivery_tag: The delivery tag of the queue task.
            body: The body of the queue tag to be processed.
        '''

        try:
            result = self.work(delivery_tag, body)
        except Exception as e:
            print(e)
            # Failure state
            callback = functools.partial(self.reject_message, delivery_tag)
        else:
            # Success state - pass result on
            callback = functools.partial(self.ack_message, delivery_tag, result)  

        self.connection.add_callback_threadsafe(callback)

    @abstractmethod
    def work(
        self,
        delivery_tag: int,
        body: str
    ) -> Any:
        '''
        Perform work on a queue task.

        Args:
            delivery_tag: The delivery tag of the queue task.
            body: The body of the queue task to be processed.

        Returns:
            The result of the work, of type Any. Raises an error if the work was unsuccessful.
        '''
        raise NotImplementedError

    @abstractmethod
    def ack_message(
        self,
        delivery_tag: int,
        result: Any,
    ):
        '''
        Callback executed when a task is processed successfully. 

        Args:
            delivery_tag: The delivery tag of the queue task.
            result: The result of the work done with the queue task.
        '''
        raise NotImplementedError

    @abstractmethod
    def reject_message(
        self,
        delivery_tag: int,
    ):
        '''
        Callback executed when a task fails.

        Args:
            delivery_tag: The delivery tag of the queue task.
        '''
        raise NotImplementedError

    def run(self):
        try:
            self.channel.start_consuming()
        except KeyboardInterrupt:
            self.channel.stop_consuming()
    
    def __del__(self):
        '''Cleanup when object is destroyed'''
        for thread in self._threads:
            thread.join()

        if hasattr(self, "connection") and self.connection.is_open:
            self.connection.close()

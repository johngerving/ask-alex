import pika

from dotenv import load_dotenv
import os

def get_pika_connection() -> pika.BlockingConnection:
    '''
    Returns a pika.BlockingConnection given the following environment variables:
        RABBITMQ_HOST
        RABBITMQ_PORT
        RABBITMQ_USER
        RABBITMQ_PASS
    '''
    load_dotenv()

    # Get environment variables
    RABBITMQ_HOST = os.getenv("RABBITMQ_HOST")
    RABBITMQ_PORT = os.getenv("RABBITMQ_PORT")
    RABBITMQ_USER = os.getenv("RABBITMQ_USER")
    RABBITMQ_PASS = os.getenv("RABBITMQ_PASS")

    # Raise exception if an environment variable doesn't exist
    if RABBITMQ_HOST is None: 
        raise Exception("RABBITMQ_HOST environment variable is empty")
    if RABBITMQ_PORT is None: 
        raise Exception("RABBITMQ_PORT environment variable is empty")
    if RABBITMQ_USER is None: 
        raise Exception("RABBITMQ_USER environment variable is empty")
    if RABBITMQ_PASS is None: 
        raise Exception("RABBITMQ_PASS environment variable is empty")

    # Create credentials and connection from environment variables
    credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
    connection = pika.BlockingConnection(pika.ConnectionParameters(RABBITMQ_HOST, RABBITMQ_PORT, credentials=credentials))

    return connection
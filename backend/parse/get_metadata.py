from datetime import datetime
import json
from dateutil import parser
import os

from typing import Any, Dict, List

import psycopg
import requests
from requests.adapters import HTTPAdapter, Retry

import numpy as np
import pandas as pd

import ray
import ray.data

import logging


def get_metadata() -> ray.data.Dataset:
    """
    Gets a ray.data.Dataset containing a list of valid links to PDFs from Digital Commons.

    Returns:
        ray.data.Dataset: A dataset with a column "link" containing links to valid PDFs, and a column "metadata" containing the metadata for a given document.
    """

    # Get the initial dataset
    ds = dataset_from_digitalcommons()

    # Filter out the invalid links
    ds = ds.filter(
        pdf_row_is_valid,
        concurrency=100,  # Start 100 concurrent processes
        num_cpus=0.2,  # 0.2 CPUs per process
    )

    ds = ds.materialize()  # Load the dataset into memory
    return ds


def dataset_from_digitalcommons() -> ray.data.Dataset:
    """
    Retrieves a list of links to PDFs from Digital Commons.

    Returns:
        ray.data.Dataset: A dataset containing links to PDFs from Digital Commons.
    """
    import numpy as np

    import requests
    from requests.adapters import HTTPAdapter, Retry

    from pyarrow import fs

    import logging

    logger = logging.getLogger("ray")

    # Create a filesystem to store the converted documents in
    filesys = fs.S3FileSystem(endpoint_override=os.getenv("AWS_ENDPOINT_URL"))

    # Get the existing dataset if it exists
    try:
        conn_str = os.getenv("PG_CONN_STR")
        if conn_str is None:
            raise Exception("Missing environment variable PG_CONN_STR")

        # Read full documents from Postgres database
        existing_ds = (
            ray.data.read_sql(
                "SELECT * FROM documents", lambda: psycopg.connect(conn_str)
            )
            .select_columns(["link"])
            .to_random_access_dataset(key="link")
        )

        # Sort the dataset by link for fast lookup
        ds_exists = True
        logger.info(f"Dataset found.")
    except Exception as e:
        logger.error(e)
        ds_exists = False
        logger.info(f"Dataset not found.")

    # Initialize HTTP session
    session = requests.Session()

    # Configure retries
    retries = Retry(
        total=5,
        backoff_factor=0.1,
        status_forcelist=[401, 403, 503],  # Force a retry on these status codes
        raise_on_status=True,
    )

    session.mount("https://", HTTPAdapter(max_retries=retries))

    start = 0
    LIMIT = 500

    ds_items: List[Dict[str, Any]] = []

    while True:
        # Get list of documents
        resp = session.get(
            f"https://content-out.bepress.com/v2/digitalcommons.humboldt.edu/query?download_format=pdf&select_fields=all&start={start}&limit={LIMIT}",
            headers={"Authorization": os.getenv("API_TOKEN")},
        )
        body = resp.json()

        # Check to make sure response fields are populated
        if "results" not in body:
            raise Exception("Document results not found")

        if not isinstance(body["results"], list):
            raise Exception(
                f"Invalid type {type(body['results'])} for document results"
            )

        results = body["results"]

        # Stop if no results
        if len(results) == 0:
            break

        for result in results:
            # Check to make sure result is valid
            if not isinstance(result, dict):
                raise Exception(
                    f"Invalid type {type(result)} for document result element"
                )

            if "download_link" not in result or len(result["download_link"]) == 0:
                raise Exception(
                    "Document result does not contain 'download_link' field"
                )

            # If the dataset already exists, search it see if the link already exists
            if ds_exists:
                # Add the item if it doesn't already exist in the dataset
                link = ray.get(existing_ds.get_async(result["download_link"]))
                print(link)
                if link is None:
                    logger.info(
                        f"Document {result['download_link']} not found in dataset. Adding to the new dataset."
                    )
                    ds_items.append(
                        {
                            "link": result["download_link"],
                            "metadata": json.dumps(result),
                        }
                    )
                else:
                    logger.info(
                        f"Document {result['download_link']} found in dataset. Ignoring."
                    )
            else:
                # Add the item anyway if the dataset doesn't exist already
                ds_items.append(
                    {"link": result["download_link"], "metadata": json.dumps(result)}
                )

        start += LIMIT

    if len(ds_items) == 0:
        raise Exception("dataset_items is empty")

    # Create a dataset from the numpy array
    dataset = ray.data.from_items(ds_items)
    dataset = dataset.repartition(
        200
    ).materialize()  # Repartition to avoid being processed in one chunk
    return dataset


def pdf_row_is_valid(row: Dict[str, str]) -> bool:
    """
    Checks if a PDF link in a ray.data.Dataset row is valid.

    Args:
        row: A Dict[str, str] representing a row in a ray.data.Dataset. Must contain the key "link".

    Returns:
        bool: False if any errors were raised in getting the PDF link, otherwise True.
    """

    import requests
    import logging

    # Get the Ray logger to log events
    logger = logging.getLogger("ray")

    session = requests.Session()

    link = row["link"]
    logger.info(f"Getting PDF at URL {link}")

    try:
        # Get the PDF - raise an error if the status of the response is bad
        r = session.get(
            link,
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux i686; rv:124.0) Gecko/20100101 Firefox/124.0"
            },
            timeout=10,
        )
        r.raise_for_status()
    except Exception as e:
        logger.error(f"Could not get link {link}: {e}")
        return False

    return True

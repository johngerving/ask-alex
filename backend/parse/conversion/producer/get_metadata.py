import asyncio
import os

from typing import Any, Dict, List

import psycopg
from pydantic import BaseModel
import requests
from requests.adapters import HTTPAdapter, Retry

from logging import Logger

logger = Logger(__name__)


class DatasetItem(BaseModel):
    """
    Represents a single item in the dataset, containing a link to a PDF and its metadata.
    """

    link: str
    metadata: Dict[str, Any]


def get_metadata() -> List[DatasetItem]:
    """
    Gets a list of valid links to PDFs from Digital Commons.

    Returns:
        List[DatasetItem]: A list of DatasetItem objects, each containing a link to a valid PDF and its associated metadata.
    """

    # Get the initial dataset
    ds = dataset_from_digitalcommons()

    ds = asyncio.run(filter_links_async(ds))

    return ds


async def filter_links_async(ds: List[DatasetItem]) -> List[DatasetItem]:
    filter_tasks = [pdf_row_is_valid(item) for item in ds]

    results = await asyncio.gather(*filter_tasks)

    ds = [item for item, is_valid in zip(ds, results) if is_valid]

    return ds


def dataset_from_digitalcommons() -> List[DatasetItem]:
    """
    Retrieves a list of links and associated metadata to PDFs from Digital Commons.

    Returns:
        List[DatasetItem]: A list of DatasetItem objects, each containing a link to a PDF and its associated metadata.
    """
    # Get the existing dataset if it exists
    conn_str = os.getenv("PG_CONN_STR")
    if conn_str is None:
        raise Exception("Missing environment variable PG_CONN_STR")

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

    ds_items: List[DatasetItem] = []

    with psycopg.connect(conn_str) as conn:
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

                # Add the item if it doesn't already exist in the dataset
                with conn.cursor() as cur:
                    # Check if row already exists in the dataset
                    cur.execute(
                        "SELECT link FROM documents WHERE link = %s",
                        (result["download_link"],),
                    )
                    records = cur.fetchone()

                if records is None:
                    logger.debug(
                        f"Document {result['download_link']} not found in dataset. Adding to the new dataset."
                    )
                    ds_items.append(
                        DatasetItem(
                            link=result["download_link"],
                            metadata=result,
                        )
                    )
                else:
                    logger.debug(
                        f"Document {result['download_link']} found in dataset. Ignoring."
                    )

            start += LIMIT

    if len(ds_items) == 0:
        raise Exception("Dataset is empty")

    return ds_items


async def pdf_row_is_valid(item: DatasetItem) -> bool:
    """
    Checks if a PDF link in a DatasetItem is valid.

    Args:
        item: A DatasetItem representing a row in a Dataset.

    Returns:
        bool: False if any errors were raised in getting the PDF link, otherwise True.
    """
    session = requests.Session()

    link = item.link
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
        logger.debug(f"Could not get link {link}: {e}")
        return False

    return True

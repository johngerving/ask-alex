import json
import os

from typing import Any, Dict, List

import requests
from requests.adapters import HTTPAdapter, Retry

import numpy as np
import pandas as pd

import ray
import ray.data

import logging

from metadata_mapping.get_department_mappings import get_department_mappings

department_mappings = get_department_mappings()


def dataset_from_digitalcommons() -> ray.data.Dataset:
    """
    Retrieves a list of links to PDFs from Digital Commons.

    Returns:
        ray.data.Dataset: A dataset containing links to PDFs from Digital Commons.
    """
    logger = logging.getLogger("ray")

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

            metadata = clean_metadata(result)

            ds_items.append(
                {
                    "link": result["download_link"],
                    "metadata": json.dumps(metadata),
                }
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


def clean_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Cleans and formats metadata for a document.

    Args:
        metadata (Dict[str, Any]): The raw metadata dictionary.

    Returns:
        Dict[str, Any]: The cleaned and formatted metadata.
    """
    # Extract metadata from Digital Commons API
    metadata_keys = [
        "title",
        "url",
        "download_link",
        "author",
        "publication_date",
        "discipline",
        "abstract",
        "publication_title",
        "configured_field_t_department",
    ]
    metadata = {key: metadata.get(key) for key in metadata_keys}

    metadata_mappings = {
        "configured_field_t_department": "department",
        "publication_title": "collection",
    }

    for key in metadata_mappings:
        metadata[metadata_mappings[key]] = metadata.pop(key)

    for key in metadata:
        if isinstance(metadata[key], list):
            metadata[key] = list(set(metadata[key]))  # Remove duplicates

    departments = metadata.get("department", [])
    if not isinstance(departments, list):
        departments = []

    # Clean up department names
    metadata["department"] = [
        department_mappings[dept] if dept in department_mappings else dept
        for dept in departments
    ]

    return metadata

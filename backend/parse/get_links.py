import os

from typing import Dict

import requests
from requests.adapters import HTTPAdapter, Retry

import numpy as np

import ray
import ray.data

# get_links returns a dataset containing a list of links to PDFs from Digital Commons
@ray.remote
def get_links() -> ray.data.Dataset:
    '''
    Retrieves a list of links to PDFs from Digital Commons.

    Returns:
        ray.data.Dataset: A dataset containing links to PDFs from Digital Commons.
    '''
    import numpy as np

    import requests
    from requests.adapters import HTTPAdapter, Retry

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
    links = []
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
            links.append(download_link)

        start += LIMIT

    # Create a dataset from the numpy array
    dataset = ray.data.from_numpy(np.array(links))
    dataset = dataset.repartition(200).materialize() # Repartition to avoid being processed in one chunk
    return dataset

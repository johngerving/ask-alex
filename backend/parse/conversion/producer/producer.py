from logging import Logger
import os
from typing import List, Dict, Any
from dotenv import load_dotenv

import psycopg
from psycopg.types.json import Jsonb

from get_metadata import DatasetItem, get_metadata

load_dotenv()
logger = Logger(__name__)


def create_tasks(ds_items: List[DatasetItem]):
    """Create tasks for processing each item in the dataset.

    Args:
        ds_items (List[DatasetItem]): A list of DatasetItem objects, each containing a link to a PDF and its associated metadata.
    """

    logger.info(f"Creating tasks for {len(ds_items)} items.")
    with psycopg.connect(os.getenv("PG_CONN_STR"), autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO tasks (payload) VALUES (%s)",
                [
                    (
                        Jsonb(
                            {
                                "task_type": "conversion",
                                "link": item.link,
                                "metadata": item.metadata,
                            }
                        ),
                    )
                    for item in ds_items
                ],
            )


if __name__ == "__main__":
    # Get metadata from Digital Commons
    ds_items = get_metadata()

    create_tasks(ds_items)

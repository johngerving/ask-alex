import asyncio
import contextlib
import json
import os
from typing import Any, Dict, List, Literal, Optional
import uuid
import httpx
from psycopg_pool import AsyncConnectionPool
from psycopg.types.json import Jsonb
import logging
from llama_index.core import Document
from pydantic import BaseModel

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

HEARTBEAT_EVERY = 30


class TaskPayload(BaseModel):
    link: str
    metadata: Dict[str, Any]


class Task(BaseModel):
    id: int
    payload: TaskPayload


class TaskResult(BaseModel):
    id: int
    payload: TaskPayload
    status: Literal["success", "failure"]
    document: Optional[Dict[str, Any]]


async def send_heartbeat(pool: AsyncConnectionPool, task_id: int, stop: asyncio.Event):
    """Send a heartbeat to the database to keep the task alive."""

    # Loop until the stop event is set
    while not stop.is_set():
        await asyncio.sleep(HEARTBEAT_EVERY)
        async with pool.connection() as conn:
            await conn.execute(
                """UPDATE tasks SET heartbeat_at = now() WHERE id = %s""", (task_id,)
            )


async def post_one(
    db_pool: AsyncConnectionPool, http_client: httpx.AsyncClient, task: Task
) -> TaskResult:
    """Post a single payload to the converter API."""

    stop = asyncio.Event()
    hb_task = asyncio.create_task(send_heartbeat(db_pool, task.id, stop))

    try:
        r = await http_client.post(
            os.getenv("CONVERTER_API_URL"), json={"link": task.payload.link}
        )
        r.raise_for_status()

        data = r.json()
        if not isinstance(data, dict):
            raise ValueError("Response is not a valid JSON object")

        if "status" not in data:
            raise ValueError("Response does not contain 'status' field")

        if data["status"] != "success":
            raise ValueError(
                f"Conversion failed: {data.get('message', 'No message provided')}"
            )

        if data["status"] == "success" and "document" not in data:
            raise ValueError("Response does not contain 'document' field")

        document_obj = data["document"]

        text = json.dumps(document_obj)

        # Extract metadata from Digital Commons API
        metadata_keys = [
            "title",
            "url",
            "download_link",
            "author",
            "publication_date",
            "discipline",
            "abstract",
        ]
        metadata = {key: task.payload.metadata.get(key) for key in metadata_keys}

        li_doc = Document(
            doc_id=str(uuid.uuid4()),
            text=text,
            metadata=metadata,
        )

        return TaskResult(
            id=task.id,
            payload=task.payload,
            status="success",
            document=li_doc.to_dict(),
        )
    except httpx.HTTPStatusError as e:
        logger.error(
            f"HTTP error occurred: {e.response.status_code} - {e.response.text}"
        )
    except ValueError as e:
        logger.error(f"Value error occurred: {str(e)}")
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
    finally:
        stop.set()
        hb_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await hb_task

    return TaskResult(
        id=task.id,
        payload=task.payload,
        status="failure",
        document=None,
    )


async def handle_tasks(
    db_pool: AsyncConnectionPool, http_client: httpx.AsyncClient, batch: List[Task]
) -> List[TaskResult]:
    """Handle a batch of tasks."""
    logger.info(f"Handling job with length {len(batch)}")
    results = await asyncio.gather(
        *(post_one(db_pool, http_client, task) for task in batch),
        return_exceptions=True,
    )
    return results


async def claim_batch(pool: AsyncConnectionPool) -> List[Task]:
    """Claim a batch of tasks from the database.

    Args:
        pool (AsyncConnectionPool): The connection pool to the PostgreSQL database.

    Returns:
        List[Task]: A list of claimed tasks.
    """
    async with pool.connection() as conn:
        async with conn.transaction(), conn.cursor() as cur:
            task_rows = await (
                await cur.execute(
                    "SELECT id, payload FROM tasks WHERE status = 'pending' AND payload->>'task_type' = 'conversion' FOR UPDATE SKIP LOCKED LIMIT 32"
                )
            ).fetchall()

            tasks: List[Task] = []

            if task_rows:
                tasks = [
                    Task(id=row[0], payload=TaskPayload(**row[1])) for row in task_rows
                ]

                await cur.executemany(
                    """
                    UPDATE tasks SET status = 'in_progress', updated_at = now() WHERE id = %s
                    """,
                    [(t.id,) for t in tasks],
                )

            return tasks


async def worker(pool: AsyncConnectionPool, idx: int):
    """Worker function that processes tasks from the database.

    Args:
        pool (AsyncConnectionPool): The connection pool to the PostgreSQL database.
        idx (int): The index of the worker.
    """
    async with httpx.AsyncClient(timeout=60 * 1) as client:
        while True:
            tasks = await claim_batch(pool)
            if not tasks:
                await asyncio.sleep(10)
                continue

            results = await handle_tasks(pool, client, tasks)

            successes: List[TaskResult] = []
            failures: List[TaskResult] = []

            for r in results:
                if r.status == "success":
                    successes.append(r)
                else:
                    failures.append(r)

            async with pool.connection() as conn:
                async with conn.transaction(), conn.cursor() as cur:
                    if successes:
                        await cur.executemany(
                            """
                            INSERT INTO documents (link, document) VALUES (%s, %s) 
                            """,
                            [(s.payload.link, Jsonb(s.document)) for s in successes],
                        )

                        await cur.executemany(
                            """
                            UPDATE tasks SET status = 'success', updated_at = now() WHERE id = %s
                            """,
                            [(s.id,) for s in successes],
                        )
                    if failures:
                        await cur.executemany(
                            """
                            UPDATE tasks SET status = 'failure', updated_at = now() WHERE id = %s
                            """,
                            [(f.id,) for f in failures],
                        )


async def main():
    async with AsyncConnectionPool(
        conninfo=os.getenv("PG_CONN_STR"),
        min_size=8,
        max_size=32,
    ) as pool:
        await asyncio.gather(*(worker(pool, i) for i in range(8)))


if __name__ == "__main__":
    asyncio.run(main())

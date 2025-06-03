import asyncio
import json
import os
from typing import Any, Dict, List
from psycopg_pool import AsyncConnectionPool
from logging import Logger


logger = Logger(__name__)


async def handle_tasks(payload: List[Dict[str, Any]]):
    """Handle a batch of tasks."""
    print(f"Handling job with length {len(payload)}")


async def worker(pool: AsyncConnectionPool, idx: int):
    """Worker function that processes tasks from the database.

    Args:
        pool (AsyncConnectionPool): The connection pool to the PostgreSQL database.
        idx (int): The index of the worker.
    """
    async with pool.connection() as conn:
        while True:
            async with conn.transaction():
                async with conn.cursor("fetch_tasks") as cur:
                    tasks = await (
                        await cur.execute(
                            "SELECT id, payload FROM tasks WHERE status = 'pending' AND payload->>'task_type' = 'conversion' FOR UPDATE SKIP LOCKED LIMIT 32"
                        )
                    ).fetchall()

                    if tasks:
                        await handle_tasks([task[1] for task in tasks])
                        continue

                    await conn.wait(interval=10)


async def main():
    async with AsyncConnectionPool(
        conninfo=os.getenv("PG_CONN_STR"),
        min_size=8,
        max_size=8,
        open=True,
    ) as pool:
        await asyncio.gather(*(worker(pool, i) for i in range(8)))


asyncio.run(main())

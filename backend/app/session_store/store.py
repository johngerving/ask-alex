from typing import Optional
from psycopg.types.json import Jsonb
import secrets
import datetime
from psycopg_pool import ConnectionPool


class SessionStore:
    def __init__(self, conn_str: str, max_age: Optional[int] = 60 * 10):
        if not conn_str:
            raise ValueError("conn_str must be set")
        if not max_age:
            raise ValueError("max_age must be set")

        self.max_age = max_age
        self.pool = ConnectionPool(conn_str)

    def new(self, data: dict) -> str:
        """Create a new session and return the resulting key."""

        key = secrets.token_bytes(32)
        expires_on = self._get_expires_on()

        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO http_sessions (key, data, expires_on)
                    VALUES (%s, %s, %s)
                    """,
                    (key, Jsonb(data), expires_on),
                )

        return key.hex()

    def get(self, key: str) -> Optional[dict]:
        """Get session data for the given key."""

        try:
            key_bytes = bytes.fromhex(key)
        except ValueError:
            return None  # Invalid hex key format

        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT data
                    FROM http_sessions
                    WHERE key = %s
                    AND expires_on > now()
                    """,
                    (key_bytes,),
                )
                res = cur.fetchone()

                return res[0] if res else None

    def delete(self, key: str):
        """Delete the session for the given key."""
        try:
            key_bytes = bytes.fromhex(key)
        except ValueError:
            return None  # Invalid hex key format

        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM http_sessions
                    WHERE key = %s
                    """,
                    (key_bytes,),
                )

    def _get_expires_on(self) -> datetime.datetime:
        """Calculates the expiration date for the session."""
        return datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(
            seconds=self.max_age
        )

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.pool.close()

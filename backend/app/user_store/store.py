from psycopg_pool import ConnectionPool
from pydantic import BaseModel


class User(BaseModel):
    id: int
    email: str


class UserStore:
    def __init__(self, conn_str: str):
        if not conn_str:
            raise ValueError("conn_str must be set")

        self.pool = ConnectionPool(conn_str)

    def create(self, email: str) -> User:
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO users (email)
                    VALUES (%s)
                    ON CONFLICT (email) DO UPDATE SET email = EXCLUDED.email
                    RETURNING id, email
                    """,
                    (email,),
                )
                res = cur.fetchone()
                return User(id=res[0], email=res[1])

    def find_by_email(self, email: str) -> User | None:
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, email
                    FROM users
                    WHERE email = %s
                    """,
                    (email,),
                )
                res = cur.fetchone()
                if res:
                    return User(id=res[0], email=res[1])
                else:
                    return None

    def find_by_id(self, id: int) -> User | None:
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, email
                    FROM users
                    WHERE id = %s
                    """,
                    (id,),
                )
                res = cur.fetchone()
                if res:
                    return User(id=res[0], email=res[1])
                else:
                    return None

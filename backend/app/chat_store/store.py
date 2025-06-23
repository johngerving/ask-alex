from datetime import datetime
from typing import List
from psycopg_pool import ConnectionPool
from pydantic import BaseModel
from llama_index.core.workflow.context import Context

from app.user_store.store import User
from app.agent.agent import Agent


class Chat:
    def __init__(self, id: int, user_id: int, context: Context, updated_at: datetime):
        self.id: int = id
        self.user_id: int = user_id
        self.context: Context = context
        self.updated_at: datetime = updated_at


class ChatStore:
    def __init__(self, conn_str: str):
        if not conn_str:
            raise ValueError("conn_str is required")

        self.conn_str = conn_str

        self.pool = ConnectionPool(conn_str)

    def get_chats(self, user: User) -> List[int]:
        """Returns a list of chat ids for the given user

        Args:
            user (User): The user to get chats for

        Returns:
            List[int]: A list of chat ids
        """
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                rows = cur.execute(
                    "SELECT id FROM chats WHERE user_id = %s ORDER BY updated_at DESC",
                    (user.id,),
                ).fetchall()

                return [row[0] for row in rows]

    def create(self, user: User) -> int:
        """Creates a new chat for the given user

        Args:
            user (User): The user to create the chat for

        Returns:
            int: The id of the created chat
        """
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO chats (user_id) VALUES (%s) RETURNING id, user_id, context, updated_at",
                    (user.id,),
                )
                row = cur.fetchone()[0]
                conn.commit()

                return row

    def find_by_id(self, chat_id: int, user: User) -> Chat:
        """Finds a chat by id

        Args:
            chat_id (int): The id of the chat to find
            user (User): The user who owns the chat

        Returns:
            Chat: The found chat
        """
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, user_id, context, updated_at FROM chats WHERE id = %s AND user_id = %s",
                    (chat_id, user.id),
                )
                row = cur.fetchone()
                if row is None:
                    raise ValueError(f"Chat with id {chat_id} not found")

                if row[2] is None:
                    context = None
                else:
                    workflow = Agent(logger=None)
                    context = Context.from_dict(workflow, data=row[2])

                return Chat(
                    id=row[0],
                    user_id=row[1],
                    context=context,
                    updated_at=row[3],
                )

    def delete(self, chat_id: int, user: User):
        """Deletes a chat by id

        Args:
            chat_id (int): The id of the chat to delete
            user (User): The user who owns the chat

        Raises:
            ValueError: If the chat with the given id does not exist
        """
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM chats WHERE id = %s AND user_id = %s RETURNING id",
                    (chat_id, user.id),
                )

                if cur.fetchone() is None:
                    raise ValueError(f"Chat with id {chat_id} not found")
                conn.commit()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.pool.close()

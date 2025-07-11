from datetime import datetime
from typing import List
from psycopg_pool import ConnectionPool
from psycopg.types.json import Jsonb
from pydantic import BaseModel
from llama_index.core.workflow.context import Context
from llama_index.core.workflow import JsonSerializer, JsonPickleSerializer

from app.user_store.store import User


class Chat:
    def __init__(self, id: int, user_id: int, context: dict, updated_at: datetime):
        self.id: int = id
        self.user_id: int = user_id
        self.context: dict = context
        self.updated_at: datetime = updated_at


class ChatStore:
    def __init__(self, conn_str: str):
        if not conn_str:
            raise ValueError("conn_str is required")

        self.conn_str = conn_str

        self.pool = ConnectionPool(conn_str)

    def get_chats(self, user: User) -> List[Chat]:
        """Returns a list of chat ids for the given user

        Args:
            user (User): The user to get chats for

        Returns:
            List[int]: A list of chat ids
        """
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                rows = cur.execute(
                    "SELECT id, user_id, context, updated_at FROM chats WHERE user_id = %s ORDER BY updated_at DESC",
                    (user.id,),
                ).fetchall()

                return [
                    Chat(
                        id=row[0],
                        user_id=row[1],
                        context=row[2],
                        updated_at=row[3],
                    )
                    for row in rows
                ]

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

                return Chat(
                    id=row[0],
                    user_id=row[1],
                    context=row[2],
                    updated_at=row[3],
                )

    def set_context(self, chat_id: int, context: Context, user: User):
        """Sets the context for a chat

        Args:
            chat_id (int): The id of the chat to set the context for
            context (Context): The context to set
            user (User): The user who owns the chat

        Raises:
            ValueError: If the chat with the given id does not exist
        """
        with self.pool.connection() as conn:
            with conn.cursor() as cur:

                try:
                    ctx_dict = context.to_dict(serializer=JsonSerializer())
                except Exception as e:
                    raise ValueError(f"Failed to serialize context: {e}")

                cur.execute(
                    "UPDATE chats SET context = %s, updated_at = NOW() WHERE id = %s AND user_id = %s RETURNING id",
                    (Jsonb(ctx_dict), chat_id, user.id),
                )

                if cur.fetchone() is None:
                    raise ValueError(f"Chat with id {chat_id} not found")
                conn.commit()

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

                cur.execute(
                    "DELETE FROM llama_index_memory WHERE key = %s RETURNING id",
                    (f"{user.id}-{chat_id}",),
                )

                if cur.fetchone() is None:
                    print(f"No memory found for chat {chat_id} and user {user.id}")

                conn.commit()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.pool.close()

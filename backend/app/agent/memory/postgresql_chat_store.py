import time
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import (
    JSON,
    Column,
    Integer,
    MetaData,
    String,
    Table,
    TIMESTAMP,
    delete,
    select,
    insert,
    update,
)
import sqlalchemy
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    create_async_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker

from llama_index.core.async_utils import asyncio_run
from llama_index.core.bridge.pydantic import Field, PrivateAttr, model_serializer
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.storage.chat_store.base_db import AsyncDBChatStore, MessageStatus
from llama_index.core.storage.chat_store.sql import SQLAlchemyChatStore


class PostgreSQLChatStore(SQLAlchemyChatStore):
    """PostgreSQL chat store using SQLAlchemy."""

    async def _setup_tables(self, async_engine: AsyncEngine) -> Table:
        # Create messages table with status column
        self._table = Table(
            f"{self.table_name}",
            self._metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("key", String, nullable=False, index=True),
            Column(
                "timestamp",
                TIMESTAMP,
                nullable=False,
                index=True,
                server_default=sqlalchemy.func.now(),
            ),
            Column("role", String, nullable=False),
            Column(
                "status",
                String,
                nullable=False,
                default=MessageStatus.ACTIVE.value,
                index=True,
            ),
            Column("data", JSON, nullable=False),
        )

        # Create tables in the database
        async with async_engine.begin() as conn:
            await conn.run_sync(self._metadata.create_all)

        return self._table

    async def add_message(
        self,
        key: str,
        message: ChatMessage,
        status: MessageStatus = MessageStatus.ACTIVE,
    ) -> None:
        """Add a message for a key with the specified status (async)."""
        session_factory, table = await self._initialize()

        async with session_factory() as session:
            await session.execute(
                insert(table).values(
                    key=key,
                    role=message.role,
                    status=status.value,
                    data=message.model_dump(mode="json"),
                )
            )
            await session.commit()

    async def add_messages(
        self,
        key: str,
        messages: List[ChatMessage],
        status: MessageStatus = MessageStatus.ACTIVE,
    ) -> None:
        """Add a list of messages in batch for the specified key and status (async)."""
        session_factory, table = await self._initialize()

        async with session_factory() as session:
            await session.execute(
                insert(table).values(
                    [
                        {
                            "key": key,
                            "role": message.role,
                            "status": status.value,
                            "data": message.model_dump(mode="json"),
                        }
                        for message in messages
                    ]
                )
            )
            await session.commit()

    async def set_messages(
        self,
        key: str,
        messages: List[ChatMessage],
        status: MessageStatus = MessageStatus.ACTIVE,
    ) -> None:
        """Set all messages for a key (replacing existing ones) with the specified status (async)."""
        session_factory, table = await self._initialize()

        # First delete all existing messages
        await self.delete_messages(key)

        async with session_factory() as session:
            for i, message in enumerate(messages):
                await session.execute(
                    insert(table).values(
                        key=key,
                        role=message.role,
                        status=status.value,
                        data=message.model_dump(mode="json"),
                    )
                )
            await session.commit()

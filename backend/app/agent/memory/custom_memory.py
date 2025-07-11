from llama_index.core.memory.memory import (
    Memory,
    BaseMemoryBlock,
    InsertMethod,
    generate_chat_store_key,
)
import asyncio
import uuid
from abc import abstractmethod
from enum import Enum
from sqlalchemy.ext.asyncio import AsyncEngine
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    TypeVar,
    Generic,
    cast,
)

from llama_index.core.async_utils import asyncio_run
from llama_index.core.base.llms.types import (
    ChatMessage,
    ContentBlock,
    TextBlock,
    AudioBlock,
    ImageBlock,
    DocumentBlock,
    CachePoint,
    CitableBlock,
    CitationBlock,
)
from llama_index.core.bridge.pydantic import (
    BaseModel,
    Field,
    model_validator,
    ConfigDict,
)
from llama_index.core.memory.types import BaseMemory
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.utils import get_tokenizer

from app.agent.memory.postgresql_chat_store import PostgreSQLChatStore

# Define type variable for memory block content
T = TypeVar("T", str, List[ContentBlock], List[ChatMessage])

DEFAULT_TOKEN_LIMIT = 30000
DEFAULT_FLUSH_SIZE = int(DEFAULT_TOKEN_LIMIT * 0.1)
DEFAULT_MEMORY_BLOCKS_TEMPLATE = RichPromptTemplate(
    """
<memory>
{% for (block_name, block_content) in memory_blocks %}
<{{ block_name }}>
  {% for block in block_content %}
    {% if block.block_type == "text" %}
{{ block.text }}
    {% elif block.block_type == "image" %}
      {% if block.url %}
        {{ (block.url | string) | image }}
      {% elif block.path %}
        {{ (block.path | string) | image }}
      {% endif %}
    {% elif block.block_type == "audio" %}
      {% if block.url %}
        {{ (block.url | string) | audio }}
      {% elif block.path %}
        {{ (block.path | string) | audio }}
      {% endif %}
    {% endif %}
  {% endfor %}
</{{ block_name }}>
{% endfor %}
</memory>
"""
)


class CustomMemory(Memory):
    """Custom memory class that extends Memory with a new from_defaults method."""

    @classmethod
    def from_defaults(  # type: ignore[override]
        cls,
        session_id: Optional[str] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        token_limit: int = DEFAULT_TOKEN_LIMIT,
        memory_blocks: Optional[List[BaseMemoryBlock[Any]]] = None,
        tokenizer_fn: Optional[Callable[[str], List]] = None,
        chat_history_token_ratio: float = 0.7,
        token_flush_size: int = DEFAULT_FLUSH_SIZE,
        memory_blocks_template: RichPromptTemplate = DEFAULT_MEMORY_BLOCKS_TEMPLATE,
        insert_method: InsertMethod = InsertMethod.SYSTEM,
        image_token_size_estimate: int = 256,
        audio_token_size_estimate: int = 256,
        # SQLAlchemyChatStore parameters
        table_name: str = "llama_index_memory",
        async_database_uri: Optional[str] = None,
        async_engine: Optional[AsyncEngine] = None,
    ) -> "CustomMemory":
        """Initialize Memory."""
        session_id = session_id or generate_chat_store_key()

        sql_store = PostgreSQLChatStore(
            table_name=table_name,
            async_database_uri=async_database_uri,
            async_engine=async_engine,
        )

        if chat_history is not None:
            asyncio_run(sql_store.set_messages(session_id, chat_history))

        if token_flush_size > token_limit:
            token_flush_size = int(token_limit * 0.7)

        return cls(
            token_limit=token_limit,
            tokenizer_fn=tokenizer_fn or get_tokenizer(),
            sql_store=sql_store,
            session_id=session_id,
            memory_blocks=memory_blocks or [],
            chat_history_token_ratio=chat_history_token_ratio,
            token_flush_size=token_flush_size,
            memory_blocks_template=memory_blocks_template,
            insert_method=insert_method,
            image_token_size_estimate=image_token_size_estimate,
            audio_token_size_estimate=audio_token_size_estimate,
        )

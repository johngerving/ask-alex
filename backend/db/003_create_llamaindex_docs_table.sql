CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS data_llamaindex_docs (
    id BIGSERIAL PRIMARY KEY,
    text VARCHAR NOT NULL,
    metadata_ JSON,
    node_id VARCHAR,
    embedding vector(768),
    text_search_tsv tsvector GENERATED ALWAYS AS (
        to_tsvector('english', text)
    ) STORED
);

CREATE INDEX IF NOT EXISTS data_llamaindex_docs_embedding_idx ON data_llamaindex_docs USING hnsw ( embedding vector_cosine_ops ) WITH (m='16', ef_construction='64' );
CREATE INDEX IF NOT EXISTS llamaindex_docs_idx ON data_llamaindex_docs USING gin ( text_search_tsv );

---- create above / drop below ----

DROP TABLE data_llamaindex_docs;
CREATE TABLE IF NOT EXISTS documents (
    link TEXT NOT NULL UNIQUE,
    document JSONB NOT NULL
);

---- create above / drop below ----

DROP TABLE documents;
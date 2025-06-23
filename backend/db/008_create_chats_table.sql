CREATE TABLE chats (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    context JSONB,
    updated_at TIMESTAMPTZ DEFAULT now()
);

---- create above / drop below ----

DROP TABLE chats;